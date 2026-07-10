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

static void removeMaskOfReadOp(IRRewriter &rewriter, Value operand, bool operandIsLhs,
                               scf::ForOp reductionLoop, std::optional<bool> &isLhsSelected,
                               Value &selectMask, vector::TransferWriteOp &reductionInitWriteOp,
                               int &reductionInitIdxInLoopIterArgs) {
  auto maskOp = dyn_cast<vector::MaskOp>(operand.getDefiningOp());
  if (!maskOp)
      return;
  Operation *maskedOp = maskOp.getMaskableOp();
  auto readOp = dyn_cast<vector::TransferReadOp>(maskedOp);
  if (!readOp)
    return;
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(readOp.getSource().getDefiningOp());
  if (!extractSliceOp)
    return;
  auto loopIterArgs = reductionLoop.getRegionIterArgs();
  auto it = llvm::find(loopIterArgs, extractSliceOp.getSource());
  if (it == loopIterArgs.end())
    return;
  reductionInitIdxInLoopIterArgs = std::distance(loopIterArgs.begin(), it);
  Value reductionInitArg = reductionLoop.getInitArgs()[reductionInitIdxInLoopIterArgs];
  auto writeOp = dyn_cast_or_null<vector::TransferWriteOp>(reductionInitArg.getDefiningOp());
  if (!writeOp)
    return;
  isLhsSelected = !operandIsLhs;
  selectMask = maskOp.getMask();
  reductionInitWriteOp = writeOp;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(maskOp);
  auto newReadOp = rewriter.create<vector::TransferReadOp>(
      readOp.getLoc(), readOp.getVectorType(), loopIterArgs[reductionInitIdxInLoopIterArgs],
      readOp.getIndices(), readOp.getPermutationMap(), readOp.getPadding(), Value(),
      readOp.getInBounds());
  rewriter.replaceOp(maskOp, newReadOp);
}

static void insertSelectBeforeReductionAndRemoveMask(
    IRRewriter &rewriter, Operation *reductionOp, scf::ForOp reductionLoop) {
  Value lhs = reductionOp->getOperand(0);
  Value rhs = reductionOp->getOperand(1);
  // reduction op has two operand, the one is the original data, and the other is the
  // iteration variable, we need to select the original data. Here we find it whether
  // the lhs or the rhs is the original data.
  std::optional<bool> isLhsSelected = std::nullopt;
  Value selectMask;
  vector::TransferWriteOp reductionInitWriteOp;
  int reductionInitIdxInLoopIterArgs = -1;
  removeMaskOfReadOp(rewriter, lhs, /*operandIsLhs*/true, reductionLoop, isLhsSelected,
                     selectMask, reductionInitWriteOp, reductionInitIdxInLoopIterArgs);
  if (!isLhsSelected.has_value())
    removeMaskOfReadOp(rewriter, rhs, /*operandIsLhs*/false, reductionLoop, isLhsSelected,
                       selectMask, reductionInitWriteOp, reductionInitIdxInLoopIterArgs);
  assert(isLhsSelected.has_value());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(reductionOp);
  Value select = rewriter.create<arith::SelectOp>(
      reductionOp->getLoc(), selectMask, *isLhsSelected ? lhs : rhs,
      reductionInitWriteOp.getVector());
  reductionOp->setOperand(*isLhsSelected ? 0 : 1, select);

  Operation *userOp = *reductionOp->getUsers().begin();
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(userOp)) {
    if (writeOp.isMasked()) {
      auto maskOp = writeOp.getMaskingOp();
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(
              writeOp.getSource().getDefiningOp())) {
        auto loopIterArgs = reductionLoop.getRegionIterArgs();
        assert(extractSliceOp.getSource() == loopIterArgs[reductionInitIdxInLoopIterArgs]);
        Operation *insertSliceOp = *maskOp->getUsers().begin();
        if (isa<tensor::InsertSliceOp>(insertSliceOp)) {
          Operation *yieldOp = *insertSliceOp->getUsers().begin();
          if (isa<scf::YieldOp>(yieldOp)) {
            rewriter.setInsertionPointAfter(maskOp);
            auto newWriteOp = rewriter.create<vector::TransferWriteOp>(
                writeOp.getLoc(), loopIterArgs[reductionInitIdxInLoopIterArgs].getType(),
                writeOp.getVector(), loopIterArgs[reductionInitIdxInLoopIterArgs],
                writeOp.getIndices(), writeOp.getPermutationMap(), Value(),
                writeOp.getInBounds());
            yieldOp->setOperand(reductionInitIdxInLoopIterArgs, newWriteOp.getResult());
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
/// If the reductionLoop loops only once, this loop will be optimized away, then the reductionOp
/// will not be enclosed by a reductionLoop. In such scenarios, we can also remove redundant
/// transfer_read and transfer_write by arith.select.
/// Before:
///   %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
///   %init = vector.transfer_write %cst
///   %extracted_slice = tensor.extract_slice %init
///   %mask = vector.create_mask
///   %lhs = vector.mask %mask { vector.transfer_read }
///   %rhs = vector.mask %mask { vector.transfer_read %extract_slice }
///   %reduction = arith.addf %lhs, %rhs {isReductionOp}
///   %write = vector.mask %mask { vector.transfer_write %reduction, %extract_slice }
///   %inserted_slice = tensor.insert_slice %write into %init
///   %read_vcadd = vector.transfer_read %inserted_slice
///   %read_acc = vector.transfer_read
///   %multi_reduction = vector.multi_reduction <add>, %read_vcadd, %read_acc {withoutInitMergeOp}
/// After:
///   %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
///   %mask = vector.create_mask
///   %lhs = vector.mask %mask { vector.transfer_read }
///   %select = arith.select %mask, %lhs, %cst
///   %reduction = arith.addf %select, %cst {isReductionOp}
///   %read_acc = vector.transfer_read
///   %multi_reduction = vector.multi_reduction <add>, %reduction, %read_acc {withoutInitMergeOp}
void RemoveMaskFromUnalignedReductionLoopPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(hivm::VectorFunctionAttr::name))
    return;
  IRRewriter rewriter(func.getContext());
  // handle those reductionOp enclosed by reductionLoop
  func.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("reductionLoop"))
      return;
    auto lb = getConstantIntValue(forOp.getLowerBound());
    auto ub = getConstantIntValue(forOp.getUpperBound());
    auto step = getConstantIntValue(forOp.getStep());
    if (!lb || !ub || !step || (*ub - *lb) % *step == 0)
      return;

    SmallVector<Operation *> reductionOps;
    forOp.walk([&](Operation *op) {
      if (op->hasAttr("reductionOp")) {
        reductionOps.push_back(op);
      }
    });
    for (Operation *reductionOp : reductionOps) {
      insertSelectBeforeReductionAndRemoveMask(rewriter, reductionOp, forOp);
    }
  });
  // handle those reductionOp not enclosed by reductionLoop
  func.walk([&](Operation *reductionOp) {
    if (!reductionOp->hasAttr("reductionOp"))
      return;
    if (reductionOp->getParentOp()->hasAttr("reductionLoop"))
      return;
    Value lhs = reductionOp->getOperand(0);
    Value rhs = reductionOp->getOperand(1);
    if (auto lhsMaskOp = dyn_cast_or_null<vector::MaskOp>(lhs.getDefiningOp())) {
      Operation *lhsMaskedOp = lhsMaskOp.getMaskableOp();
      Value selectMask = lhsMaskOp.getMask();
      if (auto lhsReadOp = dyn_cast_or_null<vector::TransferReadOp>(lhsMaskedOp)) {
        if (auto lhsExtractSliceOp = dyn_cast_or_null<tensor::ExtractSliceOp>(
                lhsReadOp.getSource().getDefiningOp())) {
          if (auto lhsWriteOp = dyn_cast_or_null<vector::TransferWriteOp>(
                  lhsExtractSliceOp.getSource().getDefiningOp())) {
            if (isa<arith::ConstantOp>(lhsWriteOp.getVector().getDefiningOp())) {
              rewriter.setInsertionPoint(reductionOp);
              Value select = rewriter.create<arith::SelectOp>(
                  reductionOp->getLoc(), selectMask, rhs, lhsWriteOp.getVector());
              reductionOp->setOperand(0, lhsWriteOp.getVector());
              reductionOp->setOperand(1, select);
            }
          }
        }
      }
    }
    if (auto rhsMaskOp = dyn_cast_or_null<vector::MaskOp>(rhs.getDefiningOp())) {
      Operation *rhsMaskedOp = rhsMaskOp.getMaskableOp();
      Value selectMask = rhsMaskOp.getMask();
      if (auto rhsReadOp = dyn_cast_or_null<vector::TransferReadOp>(rhsMaskedOp)) {
        if (auto rhsExtractSliceOp = dyn_cast_or_null<tensor::ExtractSliceOp>(
                rhsReadOp.getSource().getDefiningOp())) {
          if (auto rhsWriteOp = dyn_cast_or_null<vector::TransferWriteOp>(
                  rhsExtractSliceOp.getSource().getDefiningOp())) {
            if (isa<arith::ConstantOp>(rhsWriteOp.getVector().getDefiningOp())) {
              rewriter.setInsertionPoint(reductionOp);
              Value select = rewriter.create<arith::SelectOp>(
                  reductionOp->getLoc(), selectMask, lhs, rhsWriteOp.getVector());
              reductionOp->setOperand(1, rhsWriteOp.getVector());
              reductionOp->setOperand(0, select);
            }
          }
        }
      }
    }
    Operation *user = *reductionOp->getUsers().begin();
    if (auto userWriteOp = dyn_cast_or_null<vector::TransferWriteOp>(user)) {
      if (userWriteOp.isMasked()) {
        user = *userWriteOp.getMaskingOp()->getUsers().begin();
        if (isa<tensor::InsertSliceOp>(user)) {
          user = *user->getUsers().begin();
          if (isa<vector::TransferReadOp>(user)) {
            for (OpOperand &useOperand : user->getResult(0).getUses()) {
              user = useOperand.getOwner();
              if (user->hasAttr("withoutInitMergeOp")) {
                user->setOperand(useOperand.getOperandNumber(),
                                 reductionOp->getResult(0));
                break;
              }
            }
          }
        }
      }
    }
  });
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createRemoveMaskFromUnalignedReductionLoopPass() {
  return std::make_unique<RemoveMaskFromUnalignedReductionLoopPass>();
}
