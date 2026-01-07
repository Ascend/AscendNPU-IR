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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

namespace mlir {
namespace hivm {
namespace detail {

class DimensionAnalyzer : public ::mlir::detail::DimensionAnalyzerBase {
public:
  enum class TilingDimensionKind {
    Parallel,
    RankReduced,
    Reduce,
    NotAligned,
  };
  explicit DimensionAnalyzer(Operation *op);
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
  void computeTilingDim(bool isVectorOp = true);

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
  template <typename T, typename = std::enable_if_t<
                            std::is_same_v<T, tensor::ExpandShapeOp> ||
                            std::is_same_v<T, tensor::CollapseShapeOp>>>
  void processReshapeOp(T op);

  //===--------------------------------------------------------------------===//
  // Helper function
  //===--------------------------------------------------------------------===//

  /// mark each index of dimension as its type, and store it inside
  /// tilingDimKindMap, the key of this map is the dimension index based on
  /// solverCollapserElem_. If map doesn't exist, it means its a parallel
  void markDimensionKind();

  template <typename StoreOpTy>
  void computeTilingDimImpl(
      DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>> &parallelDimMaps,
      DenseMap<int64_t, int> &numStoreOps);

protected:
  DenseMap<Value, int64_t> tilingDim_;
  DenseMap<int64_t, TilingDimensionKind> tilingDimKindMap;
  llvm::SmallDenseSet<int> selectedTilingParIdx;
  std::unique_ptr<mlir::detail::SimpleUnionFind> solverGroup_;
};

} // namespace detail
} // namespace hivm
} // namespace mlir
#endif