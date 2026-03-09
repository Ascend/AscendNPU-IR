//===-------------------- ConvertLayoutUtils.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

#define DEBUG_TYPE "convert-layout-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {

//===----------------------------------------------------------------------===//
// Common Helpers
//===----------------------------------------------------------------------===//

/// Extract block sizes from layout based on fractal layout type
FailureOr<FractalSize> extractBlockSizes(
    DataLayoutAttr layout) {

  auto fractalSizes = layout.getFractalSizesArray();
  if (!fractalSizes.has_value()) {
    LDBG("ERROR: Layout has no fractal sizes");
    return failure();
  }

  SmallVector<int64_t> blockSizes(fractalSizes->begin(), fractalSizes->end());

  LLVM_DEBUG({
      DBGS() << "Fractal block sizes: [";
      llvm::interleaveComma(blockSizes, llvm::dbgs());
      llvm::dbgs() << "]\n";
      });

  if (blockSizes.size() != 2) {
    LDBG("ERROR: wrong block sizes (need to be 2, got "
        << blockSizes.size() << ")");
    return failure();
  }

  return FractalSize((*fractalSizes)[0], (*fractalSizes)[1]);
}

/// Compute batch index bias from rank
int computeBatchIndexBias(size_t rank) {
  return (rank == 3) ? 1 : 0;
}

bool isNDLayout(hivm::DataLayoutAttr layoutAttr) {
  return llvm::is_contained(
      {hivm::DataLayout::DOTA_ND,
       hivm::DataLayout::DOTB_ND,
       hivm::DataLayout::DOTC_ND,
       hivm::DataLayout::ND},
      layoutAttr.getDataLayout());
}

//===----------------------------------------------------------------------===//
// Public API - Unified Target Shape Computation
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<OpFoldResult>> computeMixedTargetLayoutShape(
    ArrayRef<OpFoldResult> currentShape,
    DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout,
    OpBuilder &builder,
    Location loc) {

  LDBG("=== computeMixedTargetLayoutShape ===");

  bool srcIsND = isNDLayout(srcLayout);
  bool dstIsND = isNDLayout(dstLayout);

  // ND -> Fractal conversion
  if (srcIsND && !dstIsND) {
    return computeMixedNDToFractalShape(
        currentShape, srcLayout, dstLayout, builder, loc);
  }

  // Fractal -> ND conversion
  if (!srcIsND && dstIsND) {
    return computeMixedFractalToNDShape(
        currentShape, srcLayout, dstLayout, builder, loc);
  }

  return failure();
}

bool isPropagatingUp(ConvertLayoutOp op) {
  return op.getDstLayout().getDataLayout() == DataLayout::Fractal;
}

bool isPropagatingDown(ConvertLayoutOp op) {
  return !isPropagatingUp(op);
}

bool isLayoutAgnosticOp(Operation *op) {
  // TODO: When propagating is fixed, can remove this following line
  if (!op)
    return false;
  return isa<hivm::VCastOp>(op);
  if (auto vbrcOp = dyn_cast<VBrcOp>(op)) {
    return isScalarLike(vbrcOp.getSrc().getType());
  }
  bool isAllowed = mlir::hivm::detail::isElemwiseNaryOpImpl(op);
  return isAllowed;
}

/// Check if operation is a fixpipe operation
bool isFixpipeOp(Operation *op) {
  return isa_and_present<hivm::FixpipeOp>(op);
}

/// Create a ConvertLayoutOp with the same direction attribute
Value createConvertLayoutLike(PatternRewriter &rewriter,
                              ConvertLayoutOp templateOp,
                              Value input) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  auto converted = cast<ConvertLayoutOp>(rewriter.clone(*templateOp));
  converted->setLoc(input.getLoc());
  converted.getSourceMutable().assign(input);
  auto newReplacedElementType = cast<
    ShapedType>(converted.getResult().getType()).clone(
      getElementTypeOrSelf(input));
  converted.getResult().setType(newReplacedElementType);
  return converted.getResult();
}

Value createConvertLayoutOpposite(PatternRewriter &rewriter,
                                  ConvertLayoutOp templateOp,
                                  Value input) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  auto newReplacedElementType = cast<ShapedType>(
      templateOp.getSource().getType()).clone(getElementTypeOrSelf(input));
  auto converted = rewriter.create<ConvertLayoutOp>(
      input.getLoc(), newReplacedElementType, input,
      templateOp.getDstLayoutAttr(),
      templateOp.getSrcLayoutAttr());
  return converted.getResult();
}
}