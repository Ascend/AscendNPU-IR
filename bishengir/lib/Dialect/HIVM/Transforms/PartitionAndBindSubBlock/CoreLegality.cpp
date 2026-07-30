//------------------------------CoreLegality.cpp------------------------------//
//
// Checks operand-parallel partition legality.
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
#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/CoreLegality.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hivm-partition-and-bind-sub-block"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "][legality] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hivm {
namespace partition_and_bind {

namespace {

Value getRelabelSource(Operation *op) {
  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(op))
    return toTensor.getMemref();
  if (auto toMemref = dyn_cast<bufferization::ToMemrefOp>(op))
    return toMemref.getTensor();
  if (auto view = dyn_cast<ViewLikeOpInterface>(op))
    return view.getViewSource();
  return Value();
}

bool isShapedValue(Value v) { return isa<ShapedType>(v.getType()); }

bool isNonCubeVectorOp(Operation &op) {
  if (isCubeOrSharedOp(&op))
    return false;
  if (getRelabelSource(&op))
    return false;

  return llvm::any_of(op.getOperands(), isShapedValue);
}

} // namespace

Core CoreLegalityChecker::originOf(
    Operation &op, const llvm::DenseMap<Operation *, Core> &origins) const {

  if (isCubeOrSharedOp(&op))
    return Core::Bottom;

  Core joined = Core::Bottom;
  for (Value operand : op.getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def)
      continue; // block argument => GM / shared.
    auto it = origins.find(def);
    if (it != origins.end())
      joined = join(joined, it->second);
  }
  return joined;
}

LegalityResult CoreLegalityChecker::check() {
  Block &entry = func.getBody().front();

  llvm::DenseMap<Operation *, Core> origins;
  origins.reserve(std::distance(entry.begin(), entry.end()));

  for (Operation &opRef : entry) {
    Operation *op = &opRef;

    if (Value src = getRelabelSource(op)) {
      Core srcOrigin = Core::Bottom;
      if (Operation *srcDef = src.getDefiningOp()) {
        auto it = origins.find(srcDef);
        if (it != origins.end())
          srcOrigin = it->second;
      }
      origins[op] = srcOrigin;
      LDBG("relabel " << op->getName() << " -> origin "
                      << static_cast<int>(srcOrigin));
      continue;
    }

    Core computed = originOf(*op, origins);

    // only check original sub-block scope ops
    Core scopeRootCore = getSubBlockCoreOf(op);
    if (isSingleCore(scopeRootCore) && !isCubeOrSharedOp(op))
      computed = join(computed, scopeRootCore);

    origins[op] = computed;
    LDBG("op " << op->getName() << " -> origin "
               << static_cast<int>(computed));

    if (computed == Core::Both && isNonCubeVectorOp(*op)) {
      if (succeeded(hook.tryResolve(op))) {
        origins[op] = Core::Bottom;
        LDBG("conflict on " << op->getName() << " resolved by hook");
        continue;
      }

      return LegalityResult::failure(op, "legality check failed");
    }
  }

  return LegalityResult::success();
}

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir
