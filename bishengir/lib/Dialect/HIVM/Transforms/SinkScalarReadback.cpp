//===--------------------- SinkScalarReadback.cpp -------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Sink a scalar "readback chain" down to its earliest consumer.
//
// The chain starts at a `tensor.extract` reading a single element of a tensor
// produced by a vector function call, and continues through the pure scalar
// computation that depends on it (e.g. `arith.divf` for 1/(1+exp(-x))).
//
// The scalar pipe issues in order.  After bufferization and sync solving, a
// V->S wait sits right after the producer call, so the scalar core blocks
// there before it has issued the following vector calls, leaving the vector
// pipe idle.  Moving the *readback* (producer stays in place) down to the
// consumer lets the vector calls be issued first.
//
// The move happens in the tensor/SSA stage, before bufferization and before
// graph-sync-solver.  Every op on the chain is memory-effect-free, so no alias
// analysis is needed; GSS sees the final order and derives the V->S flags from
// the consumer position on its own.
//
// See ir_sinkwait/sink_scalar_readback_design.md (v3).
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hivm-sink-scalar-readback"

namespace mlir {
#define GEN_PASS_DEF_SINKSCALARREADBACK
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct SinkPlan {
  Operation *target = nullptr;
  SmallVector<Operation *> chain; // in program order
  int64_t numCrossed = 0;
};

bool isScalarType(Type type) {
  return type.isIntOrFloat() || isa<IndexType>(type);
}

bool hasOnlyScalarResults(Operation *op) {
  for (Type type : op->getResultTypes())
    if (!isScalarType(type))
      return false;
  return true;
}

/// Whether `value` reads the same data no matter where it is read.  A value
/// backed by a mutable buffer does not: `bufferization.to_tensor` exposes such
/// a view, and slicing or casting the view keeps the same storage.  Views of
/// an `alloc_tensor` buffer are mutable too: it may be written via a memref.
/// Everything else is rejected unless it is known to be value-semantic.  In
/// particular `arith.select` forwards the storage of one of its operands, and
/// an unknown op may do the same without copying.
bool isImmutableTensorValue(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return true; // block argument
  if (isa<bufferization::ToTensorOp>(defOp))
    return false;
  if (isa<bufferization::AllocTensorOp>(defOp))
    return false;
  if (auto select = dyn_cast<arith::SelectOp>(defOp))
    return isImmutableTensorValue(select.getTrueValue()) &&
           isImmutableTensorValue(select.getFalseValue());
  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(defOp))
    return isImmutableTensorValue(slice.getSource());
  if (auto cast = dyn_cast<tensor::CastOp>(defOp))
    return isImmutableTensorValue(cast.getSource());
  if (auto reshape = dyn_cast<tensor::CollapseShapeOp>(defOp))
    return isImmutableTensorValue(reshape.getSrc());
  if (auto reshape = dyn_cast<tensor::ExpandShapeOp>(defOp))
    return isImmutableTensorValue(reshape.getSrc());
  if (isa<func::CallOp>(defOp))
    return isVFCall(defOp);
  return false;
}

/// The readback source must be an immutable SSA value defined in the seed's
/// block.
bool isValueSemanticProducer(Value source, Block *block) {
  Operation *defOp = source.getDefiningOp();
  if (!defOp || defOp->getBlock() != block)
    return false;
  if (!isImmutableTensorValue(source))
    return false;
  if (isa<func::CallOp>(defOp))
    return isVFCall(defOp);
  return isMemoryEffectFree(defOp);
}

/// Whether `op` may be moved together with the readback chain.  Conservative:
/// anything that is not a pure, region-free scalar computation bounds the
/// chain.
bool isPureScalarOp(Operation *op) {
  if (!isPure(op))
    return false;
  if (op->getNumResults() == 0 || !hasOnlyScalarResults(op))
    return false;
  // A chained readback moves with the chain, so it must observe immutable data
  // too; otherwise the later read could see an intermediate write.
  if (auto extract = dyn_cast<tensor::ExtractOp>(op))
    if (!isImmutableTensorValue(extract.getTensor()))
      return false;
  StringRef dialect = op->getName().getDialectNamespace();
  return dialect == "arith" || dialect == "math" || dialect == "index" ||
         isa<tensor::ExtractOp>(op);
}

/// The seed itself must be a constant-index read of a single-element tensor
/// produced by a value-semantic op in the same block.
bool isSinkableSeed(tensor::ExtractOp seed, Value source, Block *block) {
  if (!isScalarType(seed.getResult().getType()))
    return false;
  if (!utils::isOneSizeShape(source))
    return false;
  for (Value index : seed.getIndices())
    if (!getConstantIntValue(index))
      return false;
  return isValueSemanticProducer(source, block);
}

SmallVector<Operation *> getUsersOf(Operation *op) {
  SmallVector<Operation *> users;
  for (Value result : op->getResults()) {
    auto range = result.getUsers();
    users.append(range.begin(), range.end());
  }
  return users;
}

/// Collect the pure scalar closure of `seed` inside `block`.  A use in another
/// block is rejected outright (fail closed); a non-pure op just bounds the
/// chain.
bool collectChain(tensor::ExtractOp seed, Block *block,
                  DenseSet<Operation *> &inChain) {
  bool crossBlock = false;
  auto filter = [&](Operation *op) {
    if (op->getBlock() != block) {
      crossBlock = true;
      return false;
    }
    return isPureScalarOp(op);
  };

  SetVector<Operation *> slice;
  ForwardSliceOptions options;
  options.filter = filter;
  getForwardSlice(seed.getOperation(), &slice, options);
  if (crossBlock)
    return false;

  inChain.insert(seed);
  for (Operation *op : slice)
    inChain.insert(op);
  return true;
}

/// Users of the chain that live outside it.  Returns false when a user sits in
/// another block, in which case the chain cannot be moved.
bool collectOutsideUsers(const DenseSet<Operation *> &inChain, Block *block,
                         DenseSet<Operation *> &outsideUsers) {
  for (Operation *op : inChain)
    for (Operation *user : getUsersOf(op)) {
      if (inChain.contains(user))
        continue;
      if (user->getBlock() != block)
        return false;
      outsideUsers.insert(user);
    }
  return true;
}

/// Earliest operation among `outsideUsers`, compared in block order.
Operation *findEarliestUser(const DenseSet<Operation *> &outsideUsers) {
  Operation *earliest = nullptr;
  for (Operation *user : outsideUsers)
    if (!earliest || user->isBeforeInBlock(earliest))
      earliest = user;
  return earliest;
}

/// Record the chain ops that precede `target` in program order and count the
/// ops they would jump over.  Returns false when no chain op precedes the
/// target or when the prefix already abuts it.  Chain ops are never located
/// before the seed, so the scan starts there instead of at the block entry.
bool collectChainOps(Operation *seed, Operation *target,
                     const DenseSet<Operation *> &inChain, SinkPlan &plan) {
  bool sawChainOp = false;
  for (Operation &op :
       llvm::make_range(seed->getIterator(), target->getIterator())) {
    if (inChain.contains(&op)) {
      plan.chain.push_back(&op);
      sawChainOp = true;
    } else if (sawChainOp) {
      ++plan.numCrossed;
    }
  }
  return !plan.chain.empty() && plan.numCrossed > 0;
}

/// Describe the move to perform for `seed`, or nothing when the chain must
/// stay where it is.
std::optional<SinkPlan> buildPlan(tensor::ExtractOp seed) {
  Block *block = seed->getBlock();
  Value source = seed.getTensor();
  if (!isSinkableSeed(seed, source, block))
    return std::nullopt;

  DenseSet<Operation *> inChain;
  if (!collectChain(seed, block, inChain))
    return std::nullopt;

  DenseSet<Operation *> outsideUsers;
  if (!collectOutsideUsers(inChain, block, outsideUsers))
    return std::nullopt;
  if (outsideUsers.empty())
    return std::nullopt;

  Operation *target = findEarliestUser(outsideUsers);
  if (!target)
    return std::nullopt;
  // Only the prefix that precedes the landing point moves; chain ops after it
  // stay in place and keep consuming the moved prefix.  Dominance holds for
  // both: an operand is defined before the chain op (hence before the target)
  // or dominates the whole block, and a chain op that stays put is still after
  // everything it depends on.
  SinkPlan plan;
  plan.target = target;
  if (!collectChainOps(seed, target, inChain, plan))
    return std::nullopt;
  return plan;
}

void moveChain(const SinkPlan &plan) {
  for (Operation *op : plan.chain)
    op->moveBefore(plan.target);
}

struct SinkScalarReadbackPass
    : public impl::SinkScalarReadbackBase<SinkScalarReadbackPass> {
  using SinkScalarReadbackBase<SinkScalarReadbackPass>::SinkScalarReadbackBase;

public:
  void runOnOperation() override;
};

} // end anonymous namespace

void SinkScalarReadbackPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Visit every seed once.  Moving a chain changes the landing point of the
  // chains around it, so a fixed-point iteration is tempting but not safe:
  // two chains can leapfrog each other forever, each one reporting progress.
  // A single ordered scan terminates and still lands every chain next to its
  // earliest consumer.
  SmallVector<tensor::ExtractOp> seeds;
  funcOp.walk([&](tensor::ExtractOp extractOp) { seeds.push_back(extractOp); });

  for (tensor::ExtractOp seed : seeds) {
    std::optional<SinkPlan> plan = buildPlan(seed);
    if (!plan)
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] sinking scalar readback chain of "
               << plan->chain.size() << " op(s) across " << plan->numCrossed
               << " op(s) before '" << plan->target->getName() << "'\n");
    moveChain(*plan);
  }
}

std::unique_ptr<Pass> mlir::hivm::createSinkScalarReadbackPass() {
  return std::make_unique<SinkScalarReadbackPass>();
}
