//===- DimensionAnalyzer.h ------------------------------------------------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_DIMENSION_ANALYZER_H
#define BISHENGIR_DIALECT_HIVM_DIMENSION_ANALYZER_H

#include "bishengir/Dialect/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace hivm {
namespace detail {

class DimensionAnalyzer : public ::mlir::detail::DimensionAnalyzerBase {
public:
  enum class TilingDimensionKind {
    Parallel,
    RankReduced,
    Reduce,
    InvalidColumnSplit,
  };
  explicit DimensionAnalyzer(Operation *op, int64_t tilingSize = 2);
  LogicalResult initialize() override;

  //===--------------------------------------------------------------------===//
  // Dimension Analyzer API.
  //===--------------------------------------------------------------------===//

  bool isParallelDim(Dimension dim);
  bool isReduceDim(Dimension dim);

  /// @description: Analyze the tiling dimension for the current operation.
  ///
  /// @param isVectorOp boolean value that check whether input operation is
  /// vectorOp or cubeOp.
  /// For all storeOp, get parallel and common dimension if exists.
  /// @return bool True if tiled dimension can be broadcast two different
  /// axis case; otherwise, -1
  bool computeTilingDim(bool isVectorOp = true);

  /// @description: Identifies the parallel dimension based on the given parent
  /// index related to tiling.
  ///
  /// This function iterates through each dimension of the input value and
  /// checks if the parent index of the current dimension matches the provided
  /// tiling dimension parent index. If a match is found, it returns the
  /// corresponding dimension index; otherwise, it returns -1.
  ///
  /// @param v The input value whose dimensions are being analyzed.
  /// @return int64_t The dimension index if a match is found; otherwise, -1.
  int64_t getTilingDim(Value v);

  /// @description: Retrieves the globally unique identifier (ancestor ID) of
  /// the final selected tiling axis for a given value.
  ///
  /// @param v The input value whose global tiling axis ID is being queried.
  /// @return int64_t The globally unique ID of the tiling axis.
  int64_t getGlobalTilingAxisId(Value v);

protected:
  //===--------------------------------------------------------------------===//
  // Functions for initialization
  //===--------------------------------------------------------------------===//

  void initializeStructures() override;
  void processBFS() override;
  void processPreOrderWalk() override;
  void combineInferable() override;
  int64_t allocateArguments(int rank, ArrayRef<int64_t> dimensionRef) override;

  /// Merges dimension analysis information between input and output values,
  /// establishing relationships based on mutated dimensions and shape
  /// constraints. For example The operation VBrc [A, 1, C] -> [A, B, C], the
  /// mutatedDims is [1].
  ///
  /// @param inputs Array of input values to analyze
  /// @param outputs Array of output values to merge with inputs
  /// @param mutatedDims Dimensions that undergo mutation during the operation
  /// @param mergeMutation Whether to merge mutation information using
  ///                      joinCollapser. For mutated dimensions, or only mark
  ///                      them as having mutations by default, the value is
  ///                      true as if we want to consider them as the same
  ///                      "collapse entity"
  /// @see VGatherOp
  void mergeValues(ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                   ArrayRef<int64_t> mutatedDims = {},
                   bool mergeMutation = true);

  /// Analyzes a HIVM structured operation to determine which dimensions are
  /// mutated during execution. Mutated dimensions are those that are not
  /// parallel loop dimensions.
  ///
  /// @param hivmOp The HIVM structured operation to analyze
  /// @return A vector containing the indices of dimensions that are mutated
  ///         SmallVector<int64_t>
  SmallVector<int64_t> getMutatedDims(HIVMStructuredOp hivmOp) const;

  //===--------------------------------------------------------------------===//
  // Processors for operations
  //===--------------------------------------------------------------------===//

  bool processOperation(Operation *op, Value current) override;

  void processVBrcOp(hivm::VBrcOp op);
  void processVReduceOp(hivm::VReduceOp op);
  void processVTransposeOp(hivm::VTransposeOp op);
  void processVGatherOp(hivm::VGatherOp op);
  void processVGatherMaskOp(hivm::VGatherMaskOp op);
  void processVConcatOp(hivm::VConcatOp op);
  void processVInterleaveOp(hivm::VInterleaveOp op);
  void processVDeinterleaveOp(hivm::VDeinterleaveOp op);
  void processVPadOp(hivm::VPadOp op);
  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, hivm::VCumsumOp> ||
                                        std::is_same_v<T, hivm::VCumprodOp> ||
                                        std::is_same_v<T, hivm::VCummaxOp> ||
                                        std::is_same_v<T, hivm::VCumminOp>>>
  void processVCumOp(T op);
  void processYieldOp(scf::YieldOp op);
  void processForOp(scf::ForOp op);
  void processConditionOp(scf::ConditionOp op);
  void processExpandShapeOpLeftmostNonUnit(tensor::ExpandShapeOp op);
  void processCollapseShapeOpLeftmostNonUnit(tensor::CollapseShapeOp op);
  template <typename T, typename = std::enable_if_t<
                            std::is_same_v<T, tensor::ExpandShapeOp> ||
                            std::is_same_v<T, tensor::CollapseShapeOp>>>
  void processReshapeOp(T op);
  void processScopeOp(scope::ScopeOp op);
  void processTilingDimMapping(tensor::ExpandShapeOp expandShapeOp,
                               DictionaryAttr tilingDimMapping);
  void processMmadL1Op(hivm::MmadL1Op op, bool isTransposeA = false,
                       bool isTransposeB = false);

  void startTransaction(Operation *op);
  bool finalizeTransaction();

  //===--------------------------------------------------------------------===//
  // Helper function
  //===--------------------------------------------------------------------===//

  /// mark each index of dimension as its type, and store it inside
  /// tilingDimKindMap or transposedDimMap, the key of this map is the dimension
  /// index based on structuralDsu_. If map doesn't exist, it means its a
  /// parallel
  void markDimensions();

  void markTransposedDim(hivm::VTransposeOp op);

  /// transfer marked information through the dimensions merged by
  /// structuralDsu_
  void transferDimMark();

  template <typename IntegerRange>
  void transferDimMarkImpl(Value input, Value output,
                           const IntegerRange &mutated);

  void transferDimMarkImpl(annotation::MarkOp op);
  void transferDimMarkImpl(tensor::ExpandShapeOp op);
  void transferDimMarkImpl(tensor::CollapseShapeOp op);
  void transferDimMarkImpl(tensor::ExtractSliceOp op);
  void transferDimMarkImpl(tensor::InsertSliceOp op);
  void transferDimMarkImpl(hivm::VBrcOp op);
  void transferDimMarkImpl(hivm::VTransposeOp op);

  int64_t getHigherDimCounts(ArrayRef<Dimension> candidate,
                             SmallVectorImpl<int64_t> *candidateDims = nullptr);

  bool isValidTilingSize(int64_t dim) const;
  /// Tells us if we can still treat axis \p i as a tiling candidate for every
  /// \c StoreOpTy, even when the *view* on that axis has unknown size or
  /// size 1. This will try to recover the size of the parent buffer.
  ///
  /// Example: axis \p i is 0. The store operands are \c memref.subview results;
  /// axis 0 of each view may be `?` (or a length-1 row), but the parent buffers
  /// are still \c 16x16, so the helper can recover size 16 for both sides.
  /// \code
  /// %subSrc = memref.subview %ub[0, 0] [1, 16] [1, 1]
  ///     : memref<16x16xf16, #hivm.address_space<ub>>
  ///     to memref<?x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  /// %subDst = memref.subview %gm[0, 0] [1, 16] [1, 1]
  ///     : memref<16x16xf16, #hivm.address_space<gm>>
  ///     to memref<?x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
  /// hivm.store
  ///     ins(%subSrc : memref<?x16xf16, strided<[16, 1]>,
  ///     #hivm.address_space<ub>>) outs(%subDst : memref<?x16xf16, strided<[16,
  ///     1]>, #hivm.address_space<gm>>)
  /// \endcode
  template <typename StoreOpTy>
  bool checkTileableMaskedStore(StoreOpTy storeOp, size_t i) const;

  template <typename StoreOpTy>
  void computeTilingDimImpl(
      DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
          &parallelDimMaps,
      DenseMap<int64_t, int> &numStoreOps);

  void joinShape(int a, int b) override;
  void joinCollapser(int a, int b) override;

  template <typename StoreOpTy>
  std::optional<size_t> inferForcedTilingDim(StoreOpTy op);

protected:
  /// Chosen tiling axis index per SSA \c Value; \c -1 when not chosen.
  DenseMap<Value, int64_t> tilingDim_;

  /// How each solver dimension behaves for tiling (\c Parallel, \c RankReduced,
  /// \c Reduce). Keys follow \c markDimensionKind
  DenseMap<int64_t, TilingDimensionKind> tilingDimKindMapForCollapser;
  DenseMap<int64_t, TilingDimensionKind> tilingDimKindMapForShape;

  /// Collapsed \c parentIndex values (\c structuralDsu_) picked as tiling
  /// parents after the store-walk heuristics in \c computeTilingDim.
  llvm::SmallDenseSet<int> selectedTilingParIdx;

  /// For transposed tensors: maps a \c equivalentDsu_ index to the
  /// pre-transpose axis index used when matching tiling dims.
  DenseMap<int64_t, int64_t> transposedDimMap;

  /// \c parentIndex values where one store had two parallel axes mapping to the
  /// same collapsed parent (possible broadcast / ambiguous tiling).
  llvm::SmallDenseSet<int64_t> broadcastAxisCaseCandidate;

  int64_t tilingSize;

  /// \c processOperation transaction information
  Operation *processingOperation = nullptr;
  SmallVector<std::pair<int, int>> equivalentUpdates;
  SmallVector<std::pair<int, int>> structuralUpdates;
  SmallVector<std::pair<int, DimensionIndex>> invalidUpdates;

  /// Per structural parent index, records parent indices that are mutually
  /// exclusive and therefore must never be merged in the same transaction.
  /// The vector is indexed by dimension/parent id and each set stores the
  /// incompatible peer ids.
  ///
  /// Initialization example (see \c allocateArguments):
  /// - For one value with \c rank=3 at indices [s, s+1, s+2], each index marks
  ///   the other two as exclusive:
  ///   - \c exclusiveDimIdx[s]     = {s+1, s+2}
  ///   - \c exclusiveDimIdx[s+1]   = {s, s+2}
  ///   - \c exclusiveDimIdx[s+2]   = {s, s+1}
  /// This forms a dense rank x rank exclusivity graph without self-edges
  /// (\c i != j), i.e., \c rank*(rank-1) directed links.
  SmallVector<DenseSet<int64_t>> exclusiveDimIdx;

  bool isRegbased;

private:
  /// Updates value-group relationships for one operand use of \p user when
  /// processing \p current. \p operandNumber is the logical operand slot used
  /// by region-aware ops (e.g. scf) to map \p use to analyzer rules.
  void handleValueGroupForUse(Operation *user, Value current, OpOperand *use,
                              unsigned operandNumber);
};

} // namespace detail
} // namespace hivm
} // namespace mlir
#endif