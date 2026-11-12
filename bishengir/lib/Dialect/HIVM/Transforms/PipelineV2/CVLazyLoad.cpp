//===- CVLazyLoad.cpp - Reload tensors in consuming preload stages --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Util.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {

// Do not chase loop-carried pointers to their initial values: the address may
// change on a backedge even when the initial arguments are disjoint.
static Value getViewRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(op))
      value = view.getViewSource();
    else if (isa<memref::CastOp, memref::MemorySpaceCastOp>(op))
      value = op->getOperand(0);
    else if (auto scope = dyn_cast<scope::ScopeOp>(op)) {
      auto ret =
          cast<scope::ReturnOp>(scope.getRegion().front().getTerminator());
      value = ret.getOperand(cast<OpResult>(value).getResultNumber());
    } else
      break;
  }
  return value;
}

// Treat distinct GM roots as independent for this optimization. Reject a reload
// when the loop stores through the same root, even via a different view or in
// another iteration. Do not infer read-only storage merely from tt.tensor_kind.
static bool isReadOnlyInLoop(Value source, scf::ForOp loop) {
  Value root = getViewRoot(source);
  auto arg = dyn_cast<BlockArgument>(root);
  auto func = loop->getParentOfType<func::FuncOp>();
  if (!arg || arg.getOwner() != &func.getBody().front())
    return false;
  WalkResult result = loop.walk([&](Operation *op) {
    if (isa<annotation::MarkOp>(op))
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const auto &effect : instances) {
      if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
        continue;
      Value value = effect.getValue();
      if (!value)
        return WalkResult::interrupt();
      if (isa<TensorType>(value.getType()))
        continue;
      Value other = getViewRoot(value);
      if (other == root)
        return WalkResult::interrupt();
      auto otherArg = dyn_cast<BlockArgument>(other);
      if (otherArg && otherArg.getOwner() == arg.getOwner())
        continue;
      if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp,
                           bishengir::memref_ext::AllocWorkspaceOp>(
              other.getDefiningOp()))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

// Clone only a GM load into a private allocation and its tensor views. Do not
// rematerialize arbitrary tensor computations, mutable scalar loads or control
// flow. The original load stays available to the early Cube publication.
class LoadSlice {
public:
  LoadSlice(scope::ScopeOp producer, scf::ForOp loop)
      : producer(producer), loop(loop) {}

  bool collect(Value value) {
    Operation *op = value.getDefiningOp();
    if (!op || !producer->isAncestor(op))
      return true;
    if (ops.contains(op))
      return true;
    if (op->getBlock() != &producer.getRegion().front() || op->getNumRegions())
      return false;
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op)) {
      auto alloc = toTensor.getOperand().getDefiningOp<memref::AllocOp>();
      if (!alloc || !toTensor.getRestrict())
        return false;
      LoadOp writer;
      SmallVector<Value> worklist{alloc.getResult()};
      DenseSet<Value> visited;
      while (!worklist.empty()) {
        Value buffer = worklist.pop_back_val();
        if (!visited.insert(buffer).second)
          continue;
        for (Operation *user : buffer.getUsers()) {
          if (user->getBlock() != op->getBlock())
            return false;
          if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
            worklist.append(user->result_begin(), user->result_end());
            continue;
          }
          if (isa<bufferization::ToTensorOp>(user))
            continue;
          auto load = dyn_cast<LoadOp>(user);
          if (!load || load.getDst() != buffer || (writer && writer != load))
            return false;
          writer = load;
        }
      }
      if (!writer || !writer->isBeforeInBlock(op) ||
          !isReadOnlyInLoop(writer.getSrc(), loop))
        return false;
      for (Value operand : writer->getOperands())
        if (!collect(operand))
          return false;
      ops.insert(writer);
      hasLoad = true;
    } else if (isa<VTransposeOp, VCastOp>(op)) {
      if (!llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return isa<RankedTensorType>(type); }))
        return false;
    } else if (!isa<memref::AllocOp>(op)) {
      // Only cheap scalar/address operations and tensor views are duplicable.
      if (!isMemoryEffectFree(op) ||
          (llvm::any_of(op->getResultTypes(),
                        [](Type type) { return isa<TensorType>(type); }) &&
           !isa<tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
                tensor::ExtractSliceOp, tensor::CastOp>(op)))
        return false;
    }
    for (Value operand : op->getOperands())
      if (!collect(operand))
        return false;
    ops.insert(op);
    return true;
  }

  Value clone(Value value, scope::ScopeOp consumer, OpResult scopeResult) {
    // Keep the reload late within the consumer as well, so its local buffer
    // does not overlap unrelated temporaries at the start of the stage.
    Block &body = consumer.getRegion().front();
    Operation *firstUse = body.getTerminator();
    for (Operation *user : scopeResult.getUsers()) {
      Operation *inBody = body.findAncestorOpInBlock(*user);
      if (inBody && inBody->isBeforeInBlock(firstUse))
        firstUse = inBody;
    }
    OpBuilder builder(firstUse);
    IRMapping mapping;
    for (Operation *op : ops)
      builder.clone(*op, mapping);
    return mapping.lookup(value);
  }

  bool hasLoad = false;

private:
  scope::ScopeOp producer;
  scf::ForOp loop;
  llvm::SetVector<Operation *> ops;
};

} // namespace

static void rematerializeLateLoads(scf::ForOp loop) {
  SmallVector<scope::ScopeOp> producers(
      loop.getBody()->getOps<scope::ScopeOp>());
  for (scope::ScopeOp producer : producers) {
    if (getScopeCoreType(producer) != TCoreType::VECTOR)
      continue;
    auto ret =
        cast<scope::ReturnOp>(producer.getRegion().front().getTerminator());
    bool changed = false;
    for (OpResult result : producer.getResults()) {
      if (!isa<RankedTensorType>(result.getType()) || result.use_empty())
        continue;
      auto producerStage =
          producer->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
      if (!producerStage)
        continue;
      llvm::SetVector<scope::ScopeOp> consumers;
      for (Operation *user : result.getUsers()) {
        auto consumer = user->getParentOfType<scope::ScopeOp>();
        if (!consumer || consumer->getBlock() != producer->getBlock() ||
            getScopeCoreType(consumer) != TCoreType::VECTOR)
          continue;
        auto consumerStage =
            consumer->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
        if (consumerStage && producerStage.getInt() != consumerStage.getInt())
          consumers.insert(consumer);
      }
      if (consumers.empty())
        continue;
      Value returned = ret.getOperand(result.getResultNumber());
      LoadSlice slice(producer, loop);
      if (!slice.collect(returned) || !slice.hasLoad)
        continue;
      for (scope::ScopeOp consumer : consumers) {
        Value replacement = slice.clone(returned, consumer, result);
        changed = true;
        result.replaceUsesWithIf(replacement, [&](OpOperand &use) {
          return consumer->isProperAncestor(use.getOwner());
        });
      }
    }
    if (!changed)
      continue;
    SmallVector<Value> returns;
    SmallVector<OpResult> liveResults;
    for (auto [result, value] :
         llvm::zip(producer.getResults(), ret.getOperands())) {
      if (!result.use_empty()) {
        returns.push_back(value);
        liveResults.push_back(result);
      }
    }
    if (liveResults.size() == producer.getNumResults())
      continue;
    auto replacement = recreateScope(producer, returns);
    for (auto [oldResult, newResult] :
         llvm::zip(liveResults, replacement.getResults()))
      oldResult.replaceAllUsesWith(newResult);
    producer.erase();
  }
}

namespace mlir {
#define GEN_PASS_DEF_CVLAZYLOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

namespace {
struct CVLazyLoadPass : public impl::CVLazyLoadBase<CVLazyLoadPass> {
  void runOnOperation() override {
    getOperation().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      if (loop->hasAttr(LegacyCVPipelineAttr::name))
        return WalkResult::skip();
      if (!loop->hasAttr(CVPipelinedLoopAttr::name))
        return WalkResult::advance();
      rematerializeLateLoads(loop);
      return WalkResult::skip();
    });
  }
};
} // namespace

std::unique_ptr<Pass> hivm::createCVLazyLoadPass() {
  return std::make_unique<CVLazyLoadPass>();
}
} // namespace mlir
