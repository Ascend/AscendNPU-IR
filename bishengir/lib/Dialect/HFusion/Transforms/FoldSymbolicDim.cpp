//===- FoldSymbolicDim.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tensor.dim source replacer optimization
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "dim-source-replacer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_FOLDSYMBOLICDIM
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
namespace hfusion {

using namespace opfusion;

namespace {

LogicalResult processTensorEmpty(OpBuilder &builder, tensor::EmptyOp emptyOp) {
  // Get the source tensor and its users
  auto rankedType = getSymbolicTensor(emptyOp.getResult().getType());
  if (!rankedType.has_value())
    return failure();
  auto tmpMixed = emptyOp.getMixedSizes();
  for (auto [emptyOperandIdx, dimUsage] : llvm::enumerate(tmpMixed)) {
    if (getConstantIntValue(dimUsage).has_value())
      continue;
    auto valUsage = dimUsage.get<Value>();
    if (isa_and_nonnull<hfusion::SymbolicDimOp>(valUsage.getDefiningOp()))
      continue;
    if (emptyOperandIdx >= rankedType.value().size()) {
      llvm_unreachable("Constant dim value is greater than tensor symbol size");
    }
    auto wantedSymbol = rankedType.value()[emptyOperandIdx];
    auto symbolAttr = dyn_cast<SymbolRefAttr>(wantedSymbol);
    if (!symbolAttr)
      continue;
    auto parentOp = emptyOp->getParentOp();
    if (!parentOp)
      return failure();
    builder.setInsertionPointToStart(
        &*(parentOp->getRegions().begin()->begin()));
    auto newSymbolicDim =
        builder.create<hfusion::SymbolicDimOp>(emptyOp.getLoc(), symbolAttr);
    valUsage.replaceAllUsesWith(newSymbolicDim);
  }
  return success();
}

LogicalResult processTensorDim(OpBuilder &builder, tensor::DimOp dimOp) {
  // Get the source tensor and its users
  auto dimSrc = dimOp.getSource();
  std::optional<size_t> constantDim = dimOp.getConstantIndex();
  if (!constantDim.has_value())
    return failure();
  auto tensorSymbol = getSymbolicTensor(dimSrc.getType());
  if (!tensorSymbol.has_value())
    return failure();
  if (constantDim.value() >= tensorSymbol->size()) {
    llvm_unreachable("Constant dim value is greater than tensor symbol size");
  }
  auto symbolAttr =
      dyn_cast<SymbolRefAttr>(tensorSymbol.value()[constantDim.value()]);
  if (!symbolAttr)
    return failure();
  auto *parentOp = dimOp->getParentOp();
  if (!parentOp) {
    return emitError(dimOp.getLoc(), "Parent of dimOp doesn't exist");
  }
  builder.setInsertionPointToStart(&parentOp->getRegion(0).getBlocks().front());
  auto newSymbolicDim =
      builder.create<hfusion::SymbolicDimOp>(dimOp->getLoc(), symbolAttr);
  LDBG((dimOp->getParentOp() ? *(dimOp->getParentOp()) : *dimOp));
  dimOp.getResult().replaceAllUsesWith(newSymbolicDim);
  return success();
}

} // namespace
} // namespace hfusion

} // namespace mlir

struct FoldSymbolicDimPass
    : public impl::FoldSymbolicDimBase<FoldSymbolicDimPass> {
  void runOnOperation() override {
    LDBG("Running FoldSymbolicDim");

    auto funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    auto replacedCount = 0;
    // Walk through all tensor.empty operations

    funcOp.walk([&](tensor::DimOp dimOp) {
      auto res = processTensorDim(builder, dimOp);
      replacedCount += res.succeeded();
    });
    funcOp.walk([&](tensor::EmptyOp emptyOp) {
      auto res = processTensorEmpty(builder, emptyOp);
      replacedCount += res.succeeded();
    });
    LDBG("Replaced count " << replacedCount);
  }
};
std::unique_ptr<Pass> mlir::hfusion::createFoldSymbolicDimPass() {
  return std::make_unique<FoldSymbolicDimPass>();
}
