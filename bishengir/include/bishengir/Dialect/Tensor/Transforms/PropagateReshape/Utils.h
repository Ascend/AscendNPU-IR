//===- Utils.h ------------------------------------------------------------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_UTILS_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_UTILS_H

namespace mlir {
namespace tensor {
namespace reshape_utils {

/// Update dimensional operations
template <class OpDimTy>
void updateDimensionalOp(OpDimTy op, PatternRewriter &rewriter,
                         ArrayRef<int64_t> newDimensions);

/// Update HIVM broadcast dimensions
void updateHIVMDimensionalOp(hivm::VBrcOp op, PatternRewriter &rewriter,
                             ArrayRef<int64_t> newDimensions);

/// Update HIVM reduce dimensions
void updateHIVMDimensionalOp(hivm::VReduceOp op, PatternRewriter &rewriter,
                             ArrayRef<int64_t> newDimensions);

/// Update defining operation with destination style interface
void updateDefiningOp(Operation *definingOp, PatternRewriter &rewriter,
                      ArrayRef<Value> newOperands);

/// Update defining operation without destination style interface
void updateDefiningOpNonDst(Operation *definingOp, PatternRewriter &rewriter,
                            ArrayRef<Value> newOperands);

/// Update defining operation without destination style interface and have
/// result
void updateDefiningOpNonDst(Operation *definingOp, PatternRewriter &rewriter,
                            ArrayRef<Value> newOperands,
                            ArrayRef<int64_t> collapsedShape);

/// Renumber reassociation indices
void renumberReassociation(SmallVector<ReassociationIndices> &newReassociation);

/// Renumber reassociation indices and get new dimensions
void renumberReassociationAndGetNewDimensions(
    SmallVector<ReassociationIndices> &newReassociation,
    SmallVector<int64_t> &newDimensions);

/// Check if value is an init operand
bool checkValueIsInit(Operation *op, Value val);

/// Create new reshaping operation
template <class ReshapeOpTy, class BuilderTy>
Operation *createNewReshapingOp(BuilderTy &rewriter, Location loc,
                                Value operand,
                                ArrayRef<ReassociationIndices> reassociation,
                                ArrayRef<int64_t> resultShape);

/// Collapse and replace operation
void collapseAndReplace(PatternRewriter &rewriter,
                        MutableArrayRef<ReassociationIndices> reassociation,
                        const SmallVector<int64_t> &outputShape, Value newVal,
                        Operation *userOp);

/// Expand and replace operation
void expandAndReplace(PatternRewriter &rewriter,
                      MutableArrayRef<ReassociationIndices> reassociation,
                      const SmallVector<int64_t> &outputShape, Value newVal,
                      Operation *userOp);

/// Collapse and replace operation with reshape op
template <class ReshapeOpTy>
void collapseAndReplace(PatternRewriter &rewriter, ReshapeOpTy reshapeOp,
                        Type ty, Value newVal, Operation *definingOp);

/// Collapse and replace operation with collapse shape op
void collapseAndReplace(PatternRewriter &rewriter,
                        tensor::CollapseShapeOp collapseOp, Value newVal,
                        Operation *userOp);

/// Get result reassociation indices
SmallVector<ReassociationIndices> getResultReassociation(Operation *op);

void collapseAndReplace(PatternRewriter &rewriter,
                        MutableArrayRef<ReassociationIndices> reassociation,
                        const SmallVector<int64_t> &outputShape,
                        Value replacedVal, Value newUncollapsedVal,
                        Operation *userOp);

LogicalResult computeExpandPad(OpBuilder &rewriter, tensor::PadOp &padOp,
                               ArrayRef<ReassociationIndices> reassociation,
                               DenseMap<uint64_t, uint64_t> &padBodyMapping,
                               SmallVector<OpFoldResult> &newPadLow,
                               SmallVector<OpFoldResult> &newPadHigh,
                               SmallVector<OpFoldResult> &newExpandOutputShape,
                               ArrayRef<OpFoldResult> oldExpandOutputShape,
                               ArrayRef<int64_t> dimensionResult);

void clonePadRegion(OpBuilder &rewriter, tensor::PadOp &padOp,
                    tensor::PadOp &newPadOp,
                    DenseMap<uint64_t, uint64_t> &padBodyMapping);

void clonePadRegion(OpBuilder &rewriter, tensor::PadOp &padOp,
                    tensor::PadOp &newPadOp);

LogicalResult
computeExpandConcat(PatternRewriter &rewriter, tensor::ConcatOp &concatOp,
                    ArrayRef<ReassociationIndices> reassociation,
                    ArrayRef<OpFoldResult> mixedExpandedDimSizeOpr,
                    uint64_t &newConcatDim,
                    SmallVector<OpFoldResult> &newExpandOutputShape,
                    SmallVector<OpFoldResult> &operandsNewDimSize);

tensor::ConcatOp buildNewConcat(OpBuilder &rewriter, tensor::ConcatOp &concatOp,
                                ArrayRef<ReassociationIndices> reassociation,
                                uint64_t &newConcatDim,
                                SmallVector<OpFoldResult> &newExpandOutputShape,
                                ArrayRef<OpFoldResult> operandsNewDimSize);

tensor::ExpandShapeOp
createExpand(PatternRewriter &rewriter, Location loc, Value src,
             ArrayRef<ReassociationIndices> reassociation,
             const SmallVector<OpFoldResult> &newOutputShape);

memref::ExpandShapeOp
createMemrefExpand(PatternRewriter &rewriter, Location loc, Value src,
                   ArrayRef<ReassociationIndices> reassociation,
                   const SmallVector<OpFoldResult> &newOutputShape);

class SliceModifyingOpResult {
public:
  SliceModifyingOpResult() = default;

  // Getters
  ArrayRef<OpFoldResult> getMixedOffsets() const { return mixedOffsets; }
  ArrayRef<OpFoldResult> getMixedSizes() const { return mixedSizes; }
  ArrayRef<OpFoldResult> getMixedStrides() const { return mixedStrides; }
  ArrayRef<OpFoldResult> getSubviewOutputShape() const {
    return subviewOutputShape;
  }
  ArrayRef<OpFoldResult> getSuperviewOutputShape() const {
    return superviewOutputShape;
  }

  // Append methods for building the result
  void appendOffset(OpFoldResult offset) { mixedOffsets.push_back(offset); }
  void appendSize(OpFoldResult size) { mixedSizes.push_back(size); }
  void appendStride(OpFoldResult stride) { mixedStrides.push_back(stride); }
  void appendSubviewShape(OpFoldResult shape) {
    subviewOutputShape.push_back(shape);
  }
  void appendSuperviewShape(OpFoldResult shape) {
    superviewOutputShape.push_back(shape);
  }

  // Method to append all values at once
  void append(OpFoldResult offset, OpFoldResult size, OpFoldResult stride,
              OpFoldResult subviewShape, OpFoldResult superviewShape) {
    appendOffset(offset);
    appendSize(size);
    appendStride(stride);
    appendSubviewShape(subviewShape);
    appendSuperviewShape(superviewShape);
  }

  void replaceBack(OpFoldResult offset, OpFoldResult size, OpFoldResult stride,
                   OpFoldResult subviewShape, OpFoldResult superviewShape) {
    mixedOffsets.back() = offset;
    mixedSizes.back() = size;
    mixedStrides.back() = stride;
    subviewOutputShape.back() = subviewShape;
    superviewOutputShape.back() = superviewShape;
  }

private:
  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
  SmallVector<OpFoldResult> subviewOutputShape;
  SmallVector<OpFoldResult> superviewOutputShape;
};

LogicalResult
getExtractSliceModifyingOp(PatternRewriter &rewriter, ExtractSliceOp slicingOp,
                           ArrayRef<ReassociationIndices> reassociation,
                           ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                           SmallVector<OpFoldResult> &newMixedOffsets,
                           SmallVector<OpFoldResult> &newMixedSizes,
                           SmallVector<OpFoldResult> &newMixedStrides,
                           SmallVector<OpFoldResult> &expandOutputShape);

LogicalResult
getInsertSliceModifyingOp(PatternRewriter &rewriter, InsertSliceOp slicingOp,
                          ArrayRef<ReassociationIndices> reassociation,
                          ArrayRef<OpFoldResult> expandedRef, bool isSubview,
                          SmallVector<OpFoldResult> &newMixedOffsets,
                          SmallVector<OpFoldResult> &newMixedSizes,
                          SmallVector<OpFoldResult> &newMixedStrides,
                          SmallVector<OpFoldResult> &expandSrcOutputShape,
                          SmallVector<OpFoldResult> &expandDestOutputShape);

SmallVector<OpFoldResult> getMixedSizesOrOutputShape(PatternRewriter &rewriter,
                                                     Value val);

void updateHFusionReduceWithIndexDim(PatternRewriter &rewriter,
                                     Operation *reduceWithIndexOp,
                                     const SmallVector<int64_t> &newDimensions);

// utility functions used for reorder tranpose op
void createTransposedReassoc(
    SmallVector<ReassociationIndices, 4> &oldReassociation,
    ArrayRef<int64_t> expandedShape, ArrayRef<int64_t> permutation,
    SmallVector<int64_t, 4> &newExpandedShape,
    SmallVector<ReassociationIndices, 4> &newReassociation);

// utility functions used for reorder tranpose op
void createNewPermutation(
    size_t rank, ArrayRef<int64_t> permutation,
    SmallVector<ReassociationIndices, 4> &newExpandReassociation,
    SmallVector<int64_t, 4> &newPermutation);

// Get the inverse of this permutation
SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> permutation);

bool isNonUnitExpandOrEmptyReassoc(ArrayRef<int64_t> expandedShape,
                     ArrayRef<ReassociationIndices> reassociation);

} // namespace reshape_utils
} // namespace tensor

Operation *createNewExpandOpFromExpandOp(tensor::ExpandShapeOp expandOp,
                                         PatternRewriter &rewriter,
                                         Location loc, Value operand);

Operation *createNewExpandOpFromCollapseOp(Operation *collapseOp,
                                           PatternRewriter &rewriter,
                                           Location loc, Value operand);

SmallVector<Value> getNewOperands(Operation *collapseOp,
                                  PatternRewriter &rewriter, Operation *userOp,
                                  int64_t targetRank);

Value tryExpandOperand(Operation *collapseOp, PatternRewriter &rewriter,
                       Value operand, int64_t targetRank);

} // namespace mlir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PROPAGATERESHAPE_UTILS_H