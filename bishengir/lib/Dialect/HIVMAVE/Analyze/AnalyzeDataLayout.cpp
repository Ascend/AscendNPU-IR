//===------------------------AnalyzeDataLayout.cpp--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "data-layout-analyze"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
#define GEN_PASS_DEF_ANALYZEDATALAYOUT
#include "bishengir/Dialect/HIVMAVE/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivmave;

namespace {
struct DataLayoutAnalysisPass
    : public impl::AnalyzeDataLayoutBase<DataLayoutAnalysisPass> {
  using AnalyzeDataLayoutBase<DataLayoutAnalysisPass>::AnalyzeDataLayoutBase;

public:
  void runOnOperation() override;
};
} // namespace

static bool isSparseUnfriendlyOp(const Operation &op) {
  // If maskedStore op try to store long vector data,
  // which makes vector length is a limitation for alignment bitwidth.
  if (auto storeOp = dyn_cast<hivmave::VFMaskedStoreOp>(op)) {
    auto mask = storeOp.getMask();
    if (auto pltOp = mask.getDefiningOp<hivmave::VFPltOp>()) {
      auto trueShape = pltOp.getTrueShape();
      if (auto cstOp = trueShape.getDefiningOp<arith::ConstantOp>()) {
        auto cst = cast<IntegerAttr>(cstOp.getValue()).getInt();
        return cst > hivm::util::VL_BITS / 32;
      }
    }
    auto vecVal = storeOp.getVal();
    if (auto vbr = vecVal.getDefiningOp<hivmave::VFBroadcastScalarMaskOp>()) {
      return true;
    }
  }
  if (auto vgatherOp = dyn_cast<hivmave::VFGatherOp>(op)) {
    for (auto user : vgatherOp->getUsers()) {
      if (isa<hivmave::VFExtFOp, hivmave::VFExtSIOp, hivmave::VFTruncFOp,
              hivmave::VFTruncIOp>(user)) {
        return true;
      }
    }
  }
  return isa<hivmave::VFInterleaveOp, hivmave::VFDeInterleaveOp,
             hivmave::VFStoreWithStrideOp, hivmave::VFSlideOp>(op);
}

/// If there is a SparseUnfriendlyOp, we should set element alignment of its
/// direct parent forOp and indirect parent forOp defining its operand as -1
/// (.i.e in a compact manner), for example:
/// clang-format off
/// for          // Element Alignment = -1
///   load reg
///   for        // Element Alignment = -1
///     use reg  // use is SparseUnfriendlyOp
/// clang-format on
static void
handleSparseUnfriendlyOp(Operation &op, IRRewriter &rewriter,
                         llvm::SmallSet<Operation *, 4> &markedParentOps) {
  Operation *parentOp = op.getParentOp();
  auto compactMannerAttr =
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), -1);
  assert(parentOp != nullptr);
  parentOp->setAttr(mlir::utils::elementAlignmentBitWidth, compactMannerAttr);
  if (isa<func::FuncOp>(parentOp))
    return;
  for (auto operand : op.getOperands()) {
    assert(operand.getDefiningOp() != nullptr);
    Operation *operandParentOp = operand.getDefiningOp()->getParentOp();
    // FIXME: here we only handle operandParentOp is ForOp.
    // If operandParentOp is FuncOp, since there may be other ForOp inside
    // this FuncOp, we need a more sophisticated way to mark.
    if (isa<scf::ForOp>(operandParentOp) && operandParentOp != parentOp) {
      assert(operandParentOp != nullptr);
      operandParentOp->setAttr(mlir::utils::elementAlignmentBitWidth,
                               compactMannerAttr);
      markedParentOps.insert(operandParentOp);
    }
  }
}

/// This pass get and mark element alignment of every block.
/// Element alignment is the max bit Width of all element type in current block
/// and all data and mask should be arranged according to the element alignment
/// (.i.e in a sparse manner). But if there is sparse-unfriendly(such as
/// interleave, deinterleave, slide) op in a block, the data should be arranged
/// according to their own type(.i.e in a compact manner) and we set element
/// alignment of current block as -1.
void DataLayoutAnalysisPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(hivm::VectorFunctionAttr::name))
    return;
  IRRewriter rewriter(func.getContext());
  llvm::SmallSet<Operation *, 4> markedParentOps;
  llvm::SmallSet<Operation *, 4> markedForOps;
  func.walk([&](Block *block) {
    Operation *parentOp = block->getParentOp();
    assert(isa<func::FuncOp>(parentOp) || isa<scf::ForOp>(parentOp));
    if (markedParentOps.count(parentOp) > 0)
      return;
    int maxElementTypeBitWidth = -1;
    for (auto &op : block->getOperations()) {
      if (isa<scf::ForOp>(op)) {
        if (markedForOps.count(&op) > 0) {
          if (auto elementAlignmentAttr = op.getAttrOfType<IntegerAttr>(
                  utils::elementAlignmentBitWidth)) {
            int elementBitWidth = elementAlignmentAttr.getInt();
            maxElementTypeBitWidth =
                std::max(maxElementTypeBitWidth, elementBitWidth);
          }
        }
        continue;
      }
      if (isSparseUnfriendlyOp(op)) {
        handleSparseUnfriendlyOp(op, rewriter, markedParentOps);
        return;
      }
      SmallVector<Type> operandAndResultType(op.getOperandTypes());
      operandAndResultType.insert(operandAndResultType.end(),
                                  op.getResultTypes().begin(),
                                  op.getResultTypes().end());
      for (Type ty : operandAndResultType) {
        if (!isa<VectorType>(ty))
          continue;
        Type elementType = getElementTypeOrSelf(ty);
        if (elementType.isIndex())
          continue;
        /// clang-format off
        /// When do caculation of i64, the data will be split into two 64xi32
        /// register:
        /// 1_low  |  2_low  |  3_low  |  NA   // vector<64xi32>
        /// 1_high |  2_high |  3_high |  NA   // vector<64xi32>
        /// the mask should use plt/pge for b32:
        /// 1 | 1 | 1 | 0   // mask for low register
        /// 1 | 1 | 1 | 0   // mask for high register
        /// so here we set 32 as elementTypeBitWidth of i64 to instruct mask
        /// setting in HIVMAVEToAVEIntrin pass.
        /// clang-format on
        int elementTypeBitWidth = elementType.getIntOrFloatBitWidth() == 64
                                      ? 32
                                      : elementType.getIntOrFloatBitWidth();
        // Update maxElementTypeBitWidth
        if (maxElementTypeBitWidth < elementTypeBitWidth)
          maxElementTypeBitWidth = elementTypeBitWidth;
      }
    }
    maxElementTypeBitWidth =
        maxElementTypeBitWidth <= 8 ? 8 : maxElementTypeBitWidth;
    // Cannot set attr for block, so we set attr for parent op of block.
    auto maxElementTypeBitWidthAttr = rewriter.getIntegerAttr(
        rewriter.getIntegerType(32), maxElementTypeBitWidth);
    parentOp->setAttr(mlir::utils::elementAlignmentBitWidth,
                      maxElementTypeBitWidthAttr);
    if (isa<scf::ForOp>(parentOp)) {
      markedForOps.insert(parentOp);
    }
  });
}

std::unique_ptr<Pass> hivmave::createDataLayoutAnalysisPass() {
  return std::make_unique<DataLayoutAnalysisPass>();
}