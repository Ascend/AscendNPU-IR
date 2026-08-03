//===-------------------- PropagateConvertLayoutInsertSlice.cpp -----------===//
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

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

LogicalResult checkInsertSliceTileAlignment(tensor::InsertSliceOp insertSliceOp,
                                            DataLayoutAttr dstLayout,
                                            PatternRewriter &rewriter,
                                            ConvertLayoutOp convertOp) {
  auto sourceType =
      dyn_cast<RankedTensorType>(insertSliceOp.getSource().getType());
  auto destType = dyn_cast<RankedTensorType>(insertSliceOp.getDest().getType());
  if (!sourceType || !destType || sourceType.getRank() != destType.getRank())
    return rewriter.notifyMatchFailure(
        convertOp, "rank-reduced insert_slice is not supported");

  int64_t rank = destType.getRank();
  if (rank != 2 && rank != 3)
    return rewriter.notifyMatchFailure(
        convertOp, "insert_slice must have rank two or three");

  FailureOr<FractalSize> blockSizes = dstLayout.getFractalBlockSizes();
  if (failed(blockSizes))
    return rewriter.notifyMatchFailure(convertOp,
                                       "failed to get fractal block sizes");

  int64_t spatialStart = rank == 3 ? 1 : 0;
  SmallVector<OpFoldResult> offsets = insertSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = insertSliceOp.getMixedSizes();
  for (int64_t dim = spatialStart; dim < rank; ++dim) {
    int64_t blockSize =
        dim == spatialStart ? blockSizes->first : blockSizes->second;
    std::optional<int64_t> offset = getConstantIntValue(offsets[dim]);
    std::optional<int64_t> size = getConstantIntValue(sizes[dim]);
    if (!offset || !size || *offset % blockSize != 0 || *size % blockSize != 0)
      return rewriter.notifyMatchFailure(
          convertOp,
          "insert_slice offsets and sizes must be statically tile-aligned");
  }
  return success();
}

FailureOr<Value> createUpConvertLayoutForOperand(PatternRewriter &rewriter,
                                                 Location loc,
                                                 ConvertLayoutOp templateOp,
                                                 Value operand) {
  auto srcLayout = templateOp.getSrcLayoutAttr();
  auto dstLayout = templateOp.getDstLayoutAttr();
  auto operandType = cast<RankedTensorType>(operand.getType());
  SmallVector<OpFoldResult> operandShape = llvm::map_to_vector(
      operandType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return getAsIndexOpFoldResult(rewriter.getContext(), dim);
      });

  auto mixedShape = computeMixedTargetLayoutShape(operandShape, srcLayout,
                                                  dstLayout, rewriter, loc);
  if (failed(mixedShape))
    return failure();

  auto convertedType = RankedTensorType::get(
      decomposeMixedValues(*mixedShape).first, operandType.getElementType());
  return rewriter
      .create<ConvertLayoutOp>(loc, convertedType, operand, srcLayout,
                               dstLayout, *mixedShape)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Propagate UP through InsertSlice Operations
//===----------------------------------------------------------------------===//

/// Pattern: Push convert_layout UP through tensor.insert_slice operations
/// Before:
///   %inserted = tensor.insert_slice %source into %dest[off][sz][1,1]
///   %fractal = hivm.hir.convert_layout %inserted {up}  // ND -> Fractal
/// After:
///   %dest_fractal = hivm.hir.convert_layout %dest {up}
///   %source_fractal = hivm.hir.convert_layout %source {up}
///   %inserted_fractal = tensor.insert_slice %source_fractal into %dest_fractal
///       [off'][sz'][1,1,1,1]
struct PropagateConvertLayoutUpThroughInsertSlice
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp convertOp,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertOp))
      return failure();

    auto insertSliceOp =
        convertOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp)
      return failure();

    for (OpFoldResult stride : insertSliceOp.getMixedStrides()) {
      std::optional<int64_t> strideVal = getConstantIntValue(stride);
      if (!strideVal || *strideVal != 1)
        return rewriter.notifyMatchFailure(
            convertOp, "insert_slice has non-unit or dynamic strides");
    }

    Location loc = insertSliceOp.getLoc();
    auto srcLayout = convertOp.getSrcLayoutAttr();
    auto dstLayout = convertOp.getDstLayoutAttr();

    if (failed(checkInsertSliceTileAlignment(insertSliceOp, dstLayout, rewriter,
                                             convertOp)))
      return rewriter.notifyMatchFailure(
          convertOp, "insert_slice offsets or sizes are not tile-aligned");

    rewriter.setInsertionPoint(insertSliceOp);

    FailureOr<Value> destConverted = createUpConvertLayoutForOperand(
        rewriter, loc, convertOp, insertSliceOp.getDest());
    if (failed(destConverted))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to convert dest operand");

    FailureOr<Value> sourceConverted = createUpConvertLayoutForOperand(
        rewriter, loc, convertOp, insertSliceOp.getSource());
    if (failed(sourceConverted))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to convert source operand");

    auto newOffsets = computeTargetLayoutOffset(
        insertSliceOp.getMixedOffsets(), srcLayout, dstLayout, rewriter, loc);
    if (failed(newOffsets))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to compute fractal offsets");

    auto newSizes = computeMixedTargetLayoutShape(
        insertSliceOp.getMixedSizes(), srcLayout, dstLayout, rewriter, loc);
    if (failed(newSizes))
      return rewriter.notifyMatchFailure(convertOp,
                                         "failed to compute fractal sizes");

    int64_t fractalRank =
        cast<RankedTensorType>(destConverted->getType()).getRank();
    SmallVector<OpFoldResult> newStrides(fractalRank, rewriter.getIndexAttr(1));

    auto newInsertSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, *sourceConverted, *destConverted, *newOffsets, *newSizes,
        newStrides);

    rewriter.replaceOp(convertOp, newInsertSlice.getResult());
    return success();
  }
};

} // namespace

void mlir::hivm::populateConvertLayoutInsertSlice(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<PropagateConvertLayoutUpThroughInsertSlice>(context);
}
