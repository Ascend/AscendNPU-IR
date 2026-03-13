//===--------------- RemoveMaskFromUnalignedReductionLoop.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
#define GEN_PASS_DEF_REMOVEMASKFROMUNALIGNEDREDUCTIONLOOP
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "remove-mask-from-unaligned-reduction-loop"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct RemoveMaskFromUnalignedReductionLoopPass
    : public impl::RemoveMaskFromUnalignedReductionLoopBase<
        RemoveMaskFromUnalignedReductionLoopPass> {
public:
  void runOnOperation() override;
};

static void insertSelectBeforeReductionOp(IRRewriter& rewriter, Operation *reductionOp,
                                          scf::ForOp reductionLoop) {
  Value lhs = reductionOp->getOperand(0);
  Value rhs = reductionOp->getOperand(1);
  // reduction op has two operand, the one is the original data, and the other is the
  // iteration variable, we need to select the original data. Here we find it whether
  // the lhs or the rhs is the original data.
  std::optional<bool> isLhsSelected = std::nullopt;
  Value selectMask;
  Value reductionIterArg = reductionLoop.getRegionIterArg(0);
  if (auto lhsMaskOp = dyn_cast<vector::MaskOp>(lhs.getDefiningOp())) {
    Operation *lhsMaskedOp = lhsMaskOp.getMaskableOp();
    selectMask = lhsMaskOp.getMask();
    if (auto lhsReadOp = dyn_cast<vector::TransferReadOp>(lhsMaskedOp)) {
      if (auto lhsExtractSliceOp = dyn_cast<tensor::ExtractSliceOp>(
              lhsReadOp.getSource().getDefiningOp())) {
        isLhsSelected = lhsExtractSliceOp.getSource() != reductionIterArg;
      }
    }
  }
  if (!isLhsSelected.has_value()) {
    if (auto rhsMaskOp = dyn_cast<vector::MaskOp>(rhs.getDefiningOp())) {
      Operation *rhsMaskedOp = rhsMaskOp.getMaskableOp();
      selectMask = rhsMaskOp.getMask();
      if (auto rhsReadOp = dyn_cast<vector::TransferReadOp>(rhsMaskedOp)) {
        if (auto rhsExtractSliceOp = dyn_cast<tensor::ExtractSliceOp>(
                rhsReadOp.getSource().getDefiningOp())) {
          isLhsSelected = rhsExtractSliceOp.getSource() == reductionIterArg;
        }
      }
    }
  }
  if (selectMask && isLhsSelected.has_value()) {
    vector::TransferWriteOp writeOp;
    for (Value initArg : reductionLoop.getInitArgs()) {
      if (!initArg.getDefiningOp())
        continue;
      auto *defOp = initArg.getDefiningOp();
      if (isa<vector::TransferWriteOp>(defOp)) {
        writeOp = dyn_cast<vector::TransferWriteOp>(defOp);
        break;
      }
    }
    if (writeOp) {
      rewriter.setInsertionPoint(reductionOp);
      Value select = rewriter.create<arith::SelectOp>(
          reductionOp->getLoc(), selectMask, *isLhsSelected ? lhs : rhs, writeOp.getVector());
      reductionOp->setOperand(*isLhsSelected ? 0 : 1, select);
    }
  }
}

static void removeMaskForReadOrWriteOp(IRRewriter& rewriter, vector::MaskOp maskOp,
                                       scf::ForOp reductionLoop) {
  Operation *maskedOp = maskOp.getMaskableOp();
  Value reductionIterArg = reductionLoop.getRegionIterArg(0);
  if (auto readOp = dyn_cast<vector::TransferReadOp>(maskedOp)) {
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(
            readOp.getSource().getDefiningOp())) {
      if (extractSliceOp.getSource() == reductionIterArg) {
        rewriter.setInsertionPointAfter(maskOp);
        auto newReadOp = rewriter.create<vector::TransferReadOp>(
            readOp.getLoc(), readOp.getVectorType(), reductionIterArg,
            readOp.getIndices(), readOp.getPermutationMap(), readOp.getPadding(),
            Value(), readOp.getInBounds());
        rewriter.replaceOp(maskOp, newReadOp);
      }
    }
  } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(maskedOp)) {
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(
            writeOp.getSource().getDefiningOp())) {
      if (extractSliceOp.getSource() == reductionIterArg) {
        Operation *insertSliceOp = *maskOp->getUsers().begin();
        if (isa<tensor::InsertSliceOp>(insertSliceOp)) {
          Operation *yieldOp = *insertSliceOp->getUsers().begin();
          if (isa<scf::YieldOp>(yieldOp)) {
            rewriter.setInsertionPointAfter(maskOp);
            auto newWriteOp = rewriter.create<vector::TransferWriteOp>(
                writeOp.getLoc(), reductionIterArg.getType(), writeOp.getVector(),
                reductionIterArg, writeOp.getIndices(), writeOp.getPermutationMap(),
                Value(), writeOp.getInBounds());
            yieldOp->setOperand(0, newWriteOp.getResult());
            rewriter.eraseOp(insertSliceOp);
            rewriter.eraseOp(maskOp);
          }
        }
      }
    }
  }
}

/// For reduction loop with tail block, transfer_read and transfer_write cannot be optimized
/// away because of the mask. Here, we add arith.select before reduction op to remove the
/// mask so that transfer_read and transfer_write will be optimized away by pass
/// LoopInvariantSubsetHoisting and ConvertArithToAffine.
/// Before:
///   %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
///   %init = vector.transfer_write %cst
///   scf.for iter_args(%arg = %init)
///     %extracted_slice = tensor.extract_slice %arg
///     %mask = vector.create_mask
///     %lhs = vector.mask %mask { vector.transfer_read }
///     %rhs = vector.mask %mask { vector.transfer_read %extract_slice }
///     %reduction = arith.addf %lhs, %rhs {isReductionOp}
///     %write = vector.mask %mask { vector.transfer_write %reduction, %extract_slice }
///     %inserted_slice = tensor.insert_slice %write into %arg
///     scf.yield %inserted_slice
/// After:
///   %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
///   %init = vector.transfer_write %cst
///   scf.for iter_args(%arg = %init)
///     %mask = vector.create_mask
///     %lhs = vector.mask %mask { vector.transfer_read }
///     %rhs = vector.transfer_read %arg
///     %select = arith.select %mask, %lhs, %cst
///     %reduction = arith.addf %select, %rhs {isReductionOp}
///     %write = vector.transfer_write %reduction, %arg
///     scf.yield %write
void RemoveMaskFromUnalignedReductionLoopPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(hivm::VectorFunctionAttr::name))
    return;
  IRRewriter rewriter(func.getContext());
  func.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("reductionLoop"))
      return;
    auto lb = getConstantIntValue(forOp.getLowerBound());
    auto ub = getConstantIntValue(forOp.getUpperBound());
    auto step = getConstantIntValue(forOp.getStep());
    if (!lb || !ub || !step || (*ub - *lb) % *step == 0)
      return;

    SmallVector<vector::MaskOp> maskOps;
    forOp.walk([&](Operation *op) {
      if (vector::MaskOp maskOp = dyn_cast<vector::MaskOp>(op)) {
        maskOps.push_back(maskOp);
      } else if (op->hasAttr("reductionOp")) {
        insertSelectBeforeReductionOp(rewriter, op, forOp);
      }
    });
    for (vector::MaskOp maskOp : maskOps) {
      removeMaskForReadOrWriteOp(rewriter, maskOp, forOp);
    }
  });
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createRemoveMaskFromUnalignedReductionLoopPass() {
  return std::make_unique<RemoveMaskFromUnalignedReductionLoopPass>();
}
