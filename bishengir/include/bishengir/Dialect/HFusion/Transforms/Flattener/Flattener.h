//===- DimensionAnalyzer.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FLATTENER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FLATTENER_H
#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"

namespace mlir {
namespace hfusion {
namespace detail {

class Flattener : public DimensionAnalyzer {
public:
  /// Constructor that takes the operation to flatten.
  Flattener(Operation *op, mlir::detail::DimensionAnalyzerOptions options = {});

public:
  using CollapseGroup = SmallVector<ReassociationIndices>;

  /// Performs the flattening transformation on the operation.
  LogicalResult flatten(bool multiDynamicShape);

  /// Gets the collapse group for a given value.
private:
  DenseSet<Operation *> newlyAddedCollapse;
  DenseMap<Value, Value> valueReplacement;
  DenseSet<Operation *> markToDelete;

  /// Return whether the current value is flattened by the collapse group
  /// information. This function can be used to skip return operands to add the
  /// expansion because its never collapsed in the first place.
  /// This operation is linear because it uses the argumentList_.
  /// Change to a Set to make it sub linear.
  bool isValueFromHead(Value val) const {
    return llvm::count(argumentList_, val) > 0;
  }
  /// Marks broken connections in dimensions.
  void markBroken(const DimensionIndex &args);

  /// Propagates broken connections.
  void propagateBroken();
  void breakDynamicShapes();

  bool computeMutation(int pos, int dir) const;

  CollapseGroup getCollapseGroup(Value res) const;

  bool hasCollapseGroup(Value res) const;

  /// Gets the flattened mixed sizes for a value.
  SmallVector<OpFoldResult> getFlattenMixedSizes(Value res) const;

  // Get operations to adjust
  void collectOperations(Operation *op, SetVector<Operation *> &operations);

  /// Adjusts operations after flattening.
  void adjustOperations();

  LogicalResult VerifyCollapsedOperand(Operation *op) const;

  /// Collapses block arguments.
  void collapserForArg(Value arg, OpBuilder &builder);

  /// Collapses arguments.
  void collapseTensorArg(Value arg, OpBuilder &builder);
  void collapseMemrefArg(Value arg, OpBuilder &builder);

  /// Adjusts collapse dimensions for certain operations.
  template <class T>
  SmallVector<int64_t> adjustCollapseDimensions(T op,
                                                CollapseGroup indices) const;
  SmallVector<int64_t>
  adjustCollapseDimensionsArray(SmallVector<int64_t> dimensions,
                                CollapseGroup indices) const;

  void adjustArangeOp(hfusion::ArangeOp arangeOp, OpBuilder &builder);
  void adjustFlipOp(hfusion::FlipOp flipOp, OpBuilder &builder);
  void adjustSortOp(hfusion::SortOp sortOp, OpBuilder &builder);

  /// Adjusts indices for tensor.extract operations.
  void adjustExtractOpIndices(tensor::ExtractOp extractOp, OpBuilder &builder);

  /// Adjusts indices for tensor.pad operations.
  void adjustPadOp(tensor::PadOp padOp, OpBuilder &builder);

  /// Adjusts indices for hfusion.gather operations.
  void adjustGatherOp(hfusion::GatherOp gatherOp, OpBuilder &builder);

  MemRefType inferResultMemRefType(memref::SubViewOp subviewOp);

  std::optional<SmallVector<int64_t>> getConstantStrides(MemRefType memrefType);

  void calculateStrides(memref::SubViewOp slicingOp,
                        ReassociationIndices &collapseGroup,
                        SmallVector<OpFoldResult> &newMixedStrides,
                        OpBuilder &builder);
  
  template <class T>
  void calculateOffsets(T slicingOp,
                        ReassociationIndices &collapseGroup,
                        SmallVector<OpFoldResult> &newMixedOffsets,
                        OpBuilder &builder);

  /// Adjusts indices for tensor.concat operations.
  void adjustConcatOp(tensor::ConcatOp concatOp);

  /// Adjusts indices for tensor.insert operation.
  void adjustInsertOpIndices(tensor::InsertOp insertOp,
                             mlir::OpBuilder &builder);

  void adjustInterleaveOp(hfusion::InterleaveOp interleaveOp);

  void adjustDeinterleaveOp(hfusion::DeinterleaveOp deinterleaveOp);

  void adjustResultType(DestinationStyleOpInterface dpsLikeOp);
  void adjustResultTypeFromOperand(Operation *op);

  void adjustBroadcastOp(linalg::BroadcastOp broadcastOp, OpBuilder &builder);

  void adjustTransposeOp(linalg::TransposeOp transposeOp,
                         OpBuilder &builder) const;

  void adjustIfOp(scf::IfOp ifOp, OpBuilder &builder);
  void adjustForOp(scf::ForOp forOp, OpBuilder &builder);
  void adjustYieldOp(scf::YieldOp yieldOp, OpBuilder &builder);
  /// Adjusts dimensions for reduce-like operations.
  template <class T>
  void adjustReduceLikeOpBody(T reduceOp) const;

  /// Adjusts dimensions for cumulative operations.
  template <class T>
  void adjustCumOp(T cumOp, OpBuilder &builder);

  template <class T>
  void computeNewSlicingOperands(T slicingOp,
                                 SmallVector<OpFoldResult> &newMixedOffsets,
                                 SmallVector<OpFoldResult> &newMixedSizes,
                                 SmallVector<OpFoldResult> &newMixedStrides,
                                 OpBuilder &builder);

  void adjustExtractSliceOp(tensor::ExtractSliceOp extractSliceOp,
                            OpBuilder &builder);

  void adjustInsertSliceOp(tensor::InsertSliceOp insertSliceOp,
                           OpBuilder &builder);

  void adjustSubviewOp(memref::SubViewOp subviewOp, OpBuilder &builder);

  void adjustToTensorOp(bufferization::ToTensorOp toTensorOp,
                        OpBuilder &builder);

  void adjustToMemrefOp(bufferization::ToMemrefOp toMemrefOp,
                        OpBuilder &builder);

  void adjustEmbeddingGatherOp(hfusion::EmbeddingGatherOp embeddingGatherOp,
                               OpBuilder &builder);

  void adjustIndirectLoadOp(hfusion::IndirectLoadOp indirectLoadOp,
                            OpBuilder &builder);

  void adjustMemrefLoadOp(memref::LoadOp memrefLoadOp, OpBuilder &builder);

  void adjustMemrefStoreOp(memref::StoreOp memrefStoreOp, OpBuilder &builder);

  /// Collapses operations during adjustment.
  LogicalResult collapser(Operation *op, OpBuilder &builder);

  /// Adjusts return operations after flattening.
  void adjustReturnOp(Operation *op, OpBuilder &builder) const;

  /// Expand values for tail operations.
  template <typename OpTy>
  FailureOr<Operation *> expandForTail(OpTy &tensorOutOp, Value &collapsedVal,
                                       OpBuilder &builder);

  /// Adjusts tensor output operations (source).
  template <typename OpTy>
  void adjustTensorOutOpSource(OpTy tensorOutOp, OpBuilder &builder);

  /// Adjusts tensor output operations (destination).
  template <typename OpTy>
  void adjustTensorOutOpDest(OpTy tensorOutOp, OpBuilder &builder);

  /// Adjusts tensor output operations (source alternative).
  template <typename OpTy>
  void adjustTensorOutOpSrc(OpTy tensorOutOp, OpBuilder &builder);

  /// Adjusts tensor output operations (source alternative).
  template <typename OpTy>
  void adjustMemrefOutOpSrc(OpTy tensorOutOp, OpBuilder &builder);

  void eraseOp(Operation *op);
  void replaceOpUsage(Operation *oldOp, Operation *newOp);
  SetVector<Operation *> flattenerWorkList;

  template <typename OpTy>
  void adjustMemrefAccessOp(
      OpTy op, Value memref, OperandRange &indices, OpBuilder &builder,
      llvm::function_ref<void(SmallVector<Value> &)> prepareOperands);
};

} // namespace detail
} // namespace hfusion
} // namespace mlir
#endif
