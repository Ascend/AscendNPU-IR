//===- UnfoldSymbolicInt.cpp ----------------------------------------------===//
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
// This pass replaces all uses of symbolic_int and bind_symbolic_shape with
// tensor.dim and affine.apply
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Dialect/Symbol/Transforms/Passes.h"
#include "bishengir/Dialect/Symbol/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#define DEBUG_TYPE "unfold-symbolic-int"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_UNFOLDSYMBOLICINT
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::symbol;

namespace mlir::symbol {
struct UnfoldSymbolicIntPass
    : public mlir::impl::UnfoldSymbolicIntBase<UnfoldSymbolicIntPass> {
  // NOTE: the first bind_symbolic_shape is identity affine_map and symbols
  // are valued from bind_symbolic_shape
  void runOnOperation() override {
    LDBG("Running UnfoldSymbolicIntPass");

    auto funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    const auto unfoldingResult = funcOp->walk([this, &builder](Operation *op) {
      if (auto symbolicIntOp = dyn_cast<SymbolicIntOp>(op)) {
        if (failed(unfoldSymbolicInt(builder, symbolicIntOp)))
          return WalkResult::interrupt();
      } else if (auto bindSymbolicShapeOp = dyn_cast<BindSymbolicShapeOp>(op)) {
        if (failed(unfoldBindSymbolicShape(builder, bindSymbolicShapeOp))) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (unfoldingResult.wasInterrupted())
      return signalPassFailure();

    LDBG("Finish UnfoldSymbolicIntPass");
  }

private:
  LogicalResult unfoldSymbolicInt(OpBuilder &builder,
                                  SymbolicIntOp symbolicIntOp) const {
    AffineMapAttr affineMap = symbolicIntOp.getIntExpressionsAttr();
    if (!affineMap || affineMap.getAffineMap().isEmpty())
      return success();
    LDBG("unfolding symbol " << symbolicIntOp.getSymbolName());

    // apply the multiplication
    builder.setInsertionPoint(symbolicIntOp);
    auto applyOp = builder.create<affine::AffineApplyOp>(
        symbolicIntOp.getLoc(), affineMap, symbolicIntOp.getIntSymbols());

    // replace symbolic_int
    symbolicIntOp->replaceAllUsesWith(applyOp);
    symbolicIntOp->erase();

    LDBG("finish unfolding symbol " << symbolicIntOp.getSymbolName());
    return success();
  }

  // create dimOp from the source of the bind_symbolic_shape
  LogicalResult
  unfoldBindSymbolicShape(OpBuilder &builder,
                          BindSymbolicShapeOp bindSymbolicShapeOp) const {
    LDBG("Unfolding bind_symbolic_shape " << *bindSymbolicShapeOp);
    // list all used symbols in bind_symbolic_shape op
    DenseSet<size_t> replaceableInputIdx;
    auto inputs = bindSymbolicShapeOp.getShapeSymbolsMutable();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      auto defOp = input.get().getDefiningOp<SymbolicIntOp>();
      if (!defOp)
        continue;
      replaceableInputIdx.insert(idx);
    }

    Value src = bindSymbolicShapeOp.getOperand();
    ArrayRef<AffineExpr> results =
        bindSymbolicShapeOp.getShapeExpressions().getAffineMap().getResults();

    for (auto [resultIdx, result] : llvm::enumerate(results)) {
      // only handle indentity affine_map
      if (!result.isSymbolicOrConstant())
        continue;

      auto symExpr = dyn_cast<AffineSymbolExpr>(result);
      // skip constant result
      if (!symExpr)
        continue;

      // get the inputArg index of the symbol
      size_t inputIdx = symExpr.getPosition();
      if (!replaceableInputIdx.contains(inputIdx))
        continue;

      SymbolicIntOp symbolicIntOp =
          inputs[inputIdx].get().getDefiningOp<SymbolicIntOp>();
      // only process symbolic, skip other ops like tensor.dim or affine.apply
      if (!symbolicIntOp)
        continue;

      LDBG("unfolding symbol " << symbolicIntOp.getSymbolName());
      // get insertion point either after the first operation of the block or
      // after the defining op
      builder.setInsertionPointAfterValue(src);
      auto dimOp = builder.create<tensor::DimOp>(
          symbol::utils::getValueLocation(src), src, resultIdx);

      // remove unused symbolic int op
      symbolicIntOp->replaceAllUsesWith(dimOp);
      symbolicIntOp.erase();

      replaceableInputIdx.erase(inputIdx);
      LDBG("finish unfolding symbol");
    }

    // bind symbolic shape is no longer used
    bindSymbolicShapeOp->erase();

    LDBG("Finish unfolding bind_symbolic_shape op");
    if (!replaceableInputIdx.empty()) {
      llvm::report_fatal_error(
          "all replaceable symbols should be unfolded on the first usage");
      return failure();
    }
    return success();
  }
};
} // namespace mlir::symbol

std::unique_ptr<mlir::Pass> mlir::symbol::createUnfoldSymbolicIntPass() {
  return std::make_unique<UnfoldSymbolicIntPass>();
}
