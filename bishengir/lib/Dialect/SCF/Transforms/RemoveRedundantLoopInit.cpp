//===--- RemoveRedundantInitIterArg.cpp - remove redundant init iter arg --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iterator>

namespace mlir {
#define GEN_PASS_DEF_REMOVEREDUNDANTLOOPINIT
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define DEBUG_TYPE "scf-remove-redundant-loop-init"

using namespace mlir;

namespace {
int getUserNum(Value v) {
  auto users = v.getUsers();
  return std::distance(users.begin(), users.end());
}

bool isConstOne(OpFoldResult result) {
  if (isa<Attribute>(result)) {
    auto attr = cast<Attribute>(result);
    auto integerAttr = dyn_cast_if_present<IntegerAttr>(attr);
    if (integerAttr) {
      if (integerAttr.getInt() == 1) {
        return true;
      } else {
        return false;
      }
    }
  }

  auto resultValue = cast<Value>(result);
  auto arithConstOp =
      dyn_cast_if_present<arith::ConstantOp>(resultValue.getDefiningOp());
  if (arithConstOp == nullptr) {
    return false;
  }

  return utils::isConst<IntegerAttr, int>(arithConstOp.getValue(), 1);
}

bool isSame(Value lhs, Value rhs) {
  auto lhsOwningOp = lhs.getDefiningOp();
  if (lhsOwningOp) {
    if (auto indexCastOp =
            dyn_cast_if_present<arith::IndexCastOp>(lhsOwningOp)) {
      return isSame(indexCastOp.getIn(), rhs);
    }
    return false;
  }

  auto rhsOwningOp = rhs.getDefiningOp();
  if (rhsOwningOp) {
    if (auto indexCastOp =
            dyn_cast_if_present<arith::IndexCastOp>(rhsOwningOp)) {
      return isSame(indexCastOp.getIn(), lhs);
    }
    return false;
  }

  return lhs == rhs;
}

bool isSame(OpFoldResult lhs, OpFoldResult rhs) {
  // attr and attr cmp
  if (isa<Attribute>(lhs) && isa<Attribute>(rhs)) {
    auto lhsAttr = cast<Attribute>(lhs);
    auto rhsAttr = cast<Attribute>(rhs);
    if (isa<IntegerAttr>(lhsAttr) && isa<IntegerAttr>(rhsAttr)) {
      return cast<IntegerAttr>(lhsAttr).getInt() ==
             cast<IntegerAttr>(rhsAttr).getInt();
    }
    return lhs == rhs;
  }
  // value and value cmp
  if (isa<Value>(lhs) && isa<Value>(rhs)) {
    auto lhsValue = cast<Value>(lhs);
    auto rhsValue = cast<Value>(rhs);
    return isSame(lhsValue, rhsValue);
  }
  // attr and value cmp
  auto valueOne = cast<Value>((isa<Value>(lhs) ? lhs : rhs));
  auto attrOne = cast<Attribute>((isa<Attribute>(lhs) ? lhs : rhs));
  auto valueConstOp =
      dyn_cast_if_present<arith::ConstantOp>(valueOne.getDefiningOp());
  if (valueConstOp == nullptr) {
    return false;
  }

  if (auto integerAttr = dyn_cast_if_present<IntegerAttr>(attrOne)) {
    auto constValueAttr =
        dyn_cast_if_present<IntegerAttr>(valueConstOp.getValue());
    return constValueAttr && integerAttr.getInt() == constValueAttr.getInt();
  }

  return false;
}

std::optional<int> getUniqueSliceDim(mlir::PatternRewriter &rewriter,
                                     tensor::InsertSliceOp op) {
  auto SubsetOpInterfaceOp = cast<SubsetOpInterface>(op.getOperation());
  auto hyperRectangularSlice =
      SubsetOpInterfaceOp.getAccessedHyperrectangularSlice().value();
  if (!llvm::all_of(hyperRectangularSlice.getMixedStrides(),
                    [](OpFoldResult stride) { return isConstOne(stride); })) {
    return std::nullopt;
  }

  int64_t uniqueSliceDim = -1;
  for (auto [indx, sizePair] : llvm::enumerate(llvm::zip(
           hyperRectangularSlice.getMixedSizes(),
           tensor::getMixedSizes(rewriter, op.getLoc(), op.getDest())))) {
    auto [sliceSize, size] = sizePair;
    if (isSame(sliceSize, size)) {
      continue;
    }

    if (uniqueSliceDim == -1) {
      uniqueSliceDim = static_cast<int64_t>(indx);
    } else {
      return std::nullopt;
    }
  }

  if (uniqueSliceDim == -1) {
    return std::nullopt;
  }

  return uniqueSliceDim;
}

bool isRedundantInit(mlir::PatternRewriter &rewriter, scf::ForOp op,
                     int initIdx) {
  auto initValue = op.getInitArgs()[initIdx];
  if (llvm::any_of(initValue.getUsers(), [&](Operation *user) {
        return op->isProperAncestor(user);
      })) {
    // init is used in for body, not redundant init
    return false;
  }

  // check if loop result does not depend on the content of init value
  // namely,
  // for i 0 to %loop_size step %step  %iter_arg = %init
  //   %t = insert_slice %iter_arg [i][%step][1]
  //   yield %t
  // here %init content does not matter

  // step 1: check if it is loop iter_arg -> insert_slice -> yield chain
  auto iterArg = op.getRegionIterArg(initIdx);
  if (getUserNum(iterArg) != 1) {
    return false;
  }

  auto insertSliceOp =
      dyn_cast_if_present<tensor::InsertSliceOp>(*iterArg.getUsers().begin());
  if (insertSliceOp == nullptr) {
    return false;
  }

  if (getUserNum(insertSliceOp.getResult()) != 1) {
    return false;
  }

  auto yieldDefOp =
      op.getYieldedValues()[initIdx].getDefiningOp<tensor::InsertSliceOp>();
  if (yieldDefOp != insertSliceOp) {
    return false;
  }

  // step 2: check if the insert slice cover the full size
  auto maybeUniqueDim = getUniqueSliceDim(rewriter, insertSliceOp);
  if (!maybeUniqueDim.has_value()) {
    return false;
  }

  auto insertSliceIndx =
      insertSliceOp.getMixedOffsets()[maybeUniqueDim.value()];
  if (!isSame(op.getInductionVar(), insertSliceIndx)) {
    return false;
  }

  auto insertSliceSize = insertSliceOp.getMixedSizes()[maybeUniqueDim.value()];
  if (!isSame(insertSliceSize, op.getStep())) {
    return false;
  }

  if (!isSame(op.getLowerBound(),
              rewriter.getIntegerAttr(op.getLowerBound().getType(), 0))) {
    return false;
  }

  auto insertDstSizes = tensor::getMixedSizes(rewriter, insertSliceOp->getLoc(),
                                              insertSliceOp.getDest());
  if (!isSame(op.getUpperBound(), insertDstSizes[maybeUniqueDim.value()])) {
    return false;
  }
  return true;
}
} // namespace

struct RemoveRedundantLoopInitPattern : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inits = op.getInits();
    if (inits.empty()) {
      return failure();
    }

    bool matched = false;
    for (size_t idx = 0; idx < op.getYieldedValues().size(); idx++) {
      auto yieldValue = op.getYieldedValues()[idx];
      if (!isa<TensorType>(op.getInitsMutable()[idx].get().getType())) {
        continue;
      }

      auto initDefOp = op.getInitsMutable()[idx].get().getDefiningOp();
      if (initDefOp && isa<tensor::EmptyOp>(initDefOp)) {
        continue;
      }

      auto yieldDefOp = yieldValue.getDefiningOp();
      if (!llvm::isa_and_nonnull<tensor::InsertSliceOp>(yieldDefOp)) {
        continue;
      }

      if (!isRedundantInit(rewriter, op, idx)) {
        LDBG("not redudant init");
        continue;
      }

      matched = true;
      // replace the init by tensor empty.
      auto emptyInit = utils::createEmptyOp(
          rewriter, op.getInitsMutable()[idx].get().getLoc(),
          op.getInitsMutable()[idx].get());
      rewriter.modifyOpInPlace(
          op, [&]() { op.getInitsMutable()[idx].assign(emptyInit); });
    }

    return matched ? success() : failure();
  }
};

struct RemoveRedundantLoopInitPass
    : public impl::RemoveRedundantLoopInitBase<RemoveRedundantLoopInitPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<RemoveRedundantLoopInitPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> scf::createRemoveRedundantLoopInitPass() {
  return std::make_unique<RemoveRedundantLoopInitPass>();
}