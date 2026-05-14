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
  };
  explicit DimensionAnalyzer(Operation *op, int64_t tilingSize = 2);
  LogicalResult initialize() override;

  //===--------------------------------------------------------------------===//
  // Dimension Analyzer API.
  //===--------------------------------------------------------------------===//

  bool isParallelDim(Dimension dim);

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

protected:
  //===--------------------------------------------------------------------===//
  // Functions for initialization
  //===--------------------------------------------------------------------===//

  void initializeStructures() override;
  void processBFS() override;
  void combineInferable() override;

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
                                        std::is_same_v<T, hivm::VCumprodOp>>>
  void processVCumOp(T op);
  void processYieldOp(scf::YieldOp op);
  void processForOp(scf::ForOp op);
  void processConditionOp(scf::ConditionOp op);
  void processExpandShapeOpLeftmostNonUnit(tensor::ExpandShapeOp op);
  template <typename T, typename = std::enable_if_t<
                            std::is_same_v<T, tensor::ExpandShapeOp> ||
                            std::is_same_v<T, tensor::CollapseShapeOp>>>
  void processReshapeOp(T op);
  void processScopeOp(scope::ScopeOp op);
  void processTilingDimMapping(tensor::ExpandShapeOp expandShapeOp,
                               DictionaryAttr tilingDimMapping);

  //===--------------------------------------------------------------------===//
  // Helper function
  //===--------------------------------------------------------------------===//

  /// mark each index of dimension as its type, and store it inside
  /// tilingDimKindMap, the key of this map is the dimension index based on
  /// solverCollapserElem_. If map doesn't exist, it means its a parallel
  void markDimensionKind();

  /// Marks transposed dimensions in the operation graph and records the mapping
  /// of transposed dimensions to their original indices.
  void markTransposedDims();

  void markTransposedDimImpl(hivm::VTransposeOp op);
  void markTransposedDimImpl(annotation::MarkOp op);
  void markTransposedDimImpl(tensor::ExpandShapeOp op);

  template <typename StoreOpTy>
  void computeTilingDimImpl(
      DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
          &parallelDimMaps,
      DenseMap<int64_t, int> &numStoreOps);

protected:
  /// Chosen tiling axis index per SSA \c Value; \c -1 when not chosen.
  DenseMap<Value, int64_t> tilingDim_;

  /// How each solver dimension behaves for tiling (\c Parallel, \c RankReduced,
  /// \c Reduce). Keys follow \c markDimensionKind (via \c solverCollapserElem_).
  DenseMap<int64_t, TilingDimensionKind> tilingDimKindMap;

  /// Collapsed \c parentIndex values (\c solverCollapserElem_) picked as tiling
  /// parents after the store-walk heuristics in \c computeTilingDim.
  llvm::SmallDenseSet<int> selectedTilingParIdx;

  /// Union-find over \c argumentsRefPointer_ ids: merges SSA values that should
  /// be tiled together
  std::unique_ptr<mlir::detail::SimpleUnionFind> solverGroup_;

  /// For transposed tensors: maps a \c solverShapeElem_ index to the pre-transpose
  /// axis index used when matching tiling dims.
  DenseMap<int64_t, int64_t> transposedDimMap;

  /// \c parentIndex values where one store had two parallel axes mapping to the
  /// same collapsed parent (possible broadcast / ambiguous tiling).
  llvm::SmallDenseSet<int64_t> broadcastAxisCaseCandidate;

  int64_t tilingSize;
};

} // namespace detail
} // namespace hivm
} // namespace mlir
#endif