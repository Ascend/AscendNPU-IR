//===- DropSymbols.cpp ----------------------------------------------------===//
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

#define DEBUG_TYPE "hfusion-drop-symbols"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_DROPSYMBOLS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
namespace hfusion {

using namespace opfusion;

namespace {

Type dropSymbolsFromType(Type val) {
  auto rankedTensorTy = dyn_cast<RankedTensorType>(val);
  if (!rankedTensorTy)
    return val;
  auto newRankedTy = RankedTensorType::get(rankedTensorTy.getShape(),
                                           rankedTensorTy.getElementType());
  return newRankedTy;
}

SmallVector<Type> dropSymbolsFromType(TypeRange ty) {
  SmallVector<Type> ret;
  for (auto res : ty)
    ret.emplace_back(dropSymbolsFromType(res));
  return ret;
}

void dropSymbolsFromValue(Value val) {
  val.setType(dropSymbolsFromType(val.getType()));
}

void dropSymbolsFromValue(ValueRange val) {
  for (auto res : val)
    dropSymbolsFromValue(res);
}

/// or decode the argument’s shape.
void dropSymbolsFromFunc(func::FuncOp funcOp) {
  dropSymbolsFromValue(funcOp.getArguments());
  for (auto &block : funcOp.getBlocks())
    dropSymbolsFromValue(block.getArguments());
  for (auto &op : funcOp.getOps())
    dropSymbolsFromValue(op.getResults());
  auto oldFuncType = funcOp.getFunctionType();
  auto newFuncType = FunctionType::get(
      funcOp.getContext(), dropSymbolsFromType(oldFuncType.getInputs()),
      dropSymbolsFromType(oldFuncType.getResults()));
  funcOp.setFunctionType(newFuncType);
}

} // namespace
} // namespace hfusion

} // namespace mlir

struct DropSymbolsPass : public impl::DropSymbolsBase<DropSymbolsPass> {
  void runOnOperation() override {
    LDBG("Running DropSymbols");
    auto funcOp = getOperation();
    hfusion::dropSymbolsFromFunc(funcOp);
  }
};
std::unique_ptr<Pass> mlir::hfusion::createDropSymbolsPass() {
  return std::make_unique<DropSymbolsPass>();
}
