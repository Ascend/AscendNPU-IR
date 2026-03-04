//===- InferFuncFuseKind.cpp -- label host function to a fusion kind ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements fusion kind inferring and labeling
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlock.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlockAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"

#include <numeric>
#include <queue>

#define DEBUG_TYPE "hfusion-infer-func"

namespace mlir {
#define GEN_PASS_DEF_INFERFUNCFUSIONKIND
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {
std::pair<int, SetVector<Operation *>>
getImportantOpInformation(func::FuncOp func) {
  int opCount = 0;
  SetVector<Operation *> ops;
  func.walk([&](Operation *op) {
    if (!opfusion::FusibleHelper::isImportantPattern(op))
      return;

    opCount++;
    ops.insert(op);
  });
  return {opCount, ops};
}

inline Block *getSingleFuncBlock(func::FuncOp func) {
  auto &funcBlocks = func.getBody();
  assert(llvm::hasSingleElement(funcBlocks));
  return &funcBlocks.front();
}

} // namespace

bool tryFusionKind(func::FuncOp func, const FusionKind &fusionKind) {
  // Only functions with a single block is considered fusible.
  if (!llvm::hasSingleElement(func.getBody()))
    return false;

  auto *funcBlock = getSingleFuncBlock(func);
  // Check if all operations here is outlined.
  opfusion::FusibleHelper fusibleHelper(fusionKind, /*bufferToOut=*/false,
                                        /*maxHorizontalFusionSize=*/-1);
  opfusion::FusibleBlockAnalyzer analyzer(*funcBlock, fusibleHelper);
  return analyzer.isFusible();
}

FusionKind hfusion::inferFuncFusionKind(func::FuncOp func) {
  auto [opCount, ops] = getImportantOpInformation(func);

  if (opCount == 0) {
    LLVM_DEBUG(llvm::dbgs() << "No outlinable op found, skipping\n";);
    return FusionKind::AnyPB;
  }

  // Single outlinable corner case
  if (opCount == 1) {
    LLVM_DEBUG(llvm::dbgs() << "Single outlinable function found\n";);
    return opfusion::FusibleHelper::getSingleFusionKind(ops.front());
  }

  for (uint32_t i = 1; i < getMaxEnumValForFusionKind(); i++) {
    // Loop through all this and apply fusion to it
    auto fusionKind = symbolizeFusionKind(i).value();
    if (tryFusionKind(func, fusionKind))
      return fusionKind;
  }
  LLVM_DEBUG(llvm::dbgs() << "This function cannot be labeled\n";);
  return FusionKind::Unknown;
}

LogicalResult hfusion::canFuse(func::FuncOp func) {
  return success(hfusion::inferFuncFusionKind(func) != FusionKind::Unknown);
}

namespace mlir {

using namespace hfusion;
struct InferFuncFusionKindPass
    : public impl::InferFuncFusionKindBase<InferFuncFusionKindPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto wantedFusionKind = tryGetFusionKind(func);
    auto inferredKind = hfusion::inferFuncFusionKind(func);
    if (!wantedFusionKind.has_value()) {
      trySetFusionKind(func, inferredKind);
      return;
    }

    if (wantedFusionKind.value() == inferredKind)
      return;

    func->emitWarning() << "Wanted fusion kind: "
                        << stringifyFusionKind(wantedFusionKind.value())
                        << " is overridden by inferred fusion: "
                        << stringifyFusionKind(inferredKind);
    trySetFusionKind(func, inferredKind);
  }
};

} // namespace mlir

std::unique_ptr<Pass> mlir::hfusion::createInferFuncFusionKind() {
  return std::make_unique<InferFuncFusionKindPass>();
}