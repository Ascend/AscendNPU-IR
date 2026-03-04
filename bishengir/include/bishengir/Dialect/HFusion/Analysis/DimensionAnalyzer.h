//===- DimensionAnalyzer.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/UnionFind.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_DIMENSION_ANALYZER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_DIMENSION_ANALYZER_H

namespace mlir {
namespace hfusion {
namespace detail {

using ::mlir::detail::DimensionPosition;
using ::mlir::detail::kMaxDimPos;
using ::mlir::detail::kUndefinedShaped;

/// AnchorElement struct
struct AnchorElement {
  /// index of solverCollapserElem_ element of corresponding index
  int64_t axisIndex;
  /// maximum static shape of related solverShapeElem_ element
  int64_t maxStaticShape = 1;
  /// all dynamic shape of related solverShapeElem_ element
  SmallVector<int64_t> dynamicShapeIndices;
};

/// The DimensionAnalyzer class implements the logic to analyze dimensions
/// relationship and can flatten operations in MLIR.
/// It processes operations and flattens their iteration spaces when possible.
class DimensionAnalyzer : public ::mlir::detail::DimensionAnalyzerBase {
public:
  using AnchorDimension = Dimension;
  /// Constructor that takes the operation to flatten.
  explicit DimensionAnalyzer(
      Operation *op, mlir::detail::DimensionAnalyzerOptions options = {})
      : DimensionAnalyzerBase(op, options) {};
  ~DimensionAnalyzer() override = default;

  /// @description: Find and set the anchor value based on current arguments.
  /// In this implementation, we choose the argument with the highest
  /// tensor-rank as the anchor.
  void computeAnchor();

  //===--------------------------------------------------------------------===//
  // Dimension Analyzer API.
  //===--------------------------------------------------------------------===//

  /// @description: get the current value's axis relationship with the anchor
  /// @input v: tensor-typed value
  /// @return a BitVector of size equal to the anchor's rank.
  ///         Exactly v.getType().getRank() bits will be set to 1, indicating
  ///         that these axes of the anchor are "common" with v. For example,
  ///         if:
  ///           - anchor is <A1, A2, A3, R>
  ///           - v is a tensor of rank 1 (e.g. <R>)
  ///         then this function will return a BitVector [false, false, false,
  ///         true], implying that v corresponds to the last axis of the anchor.
  BitVector getCommonAxis(Value v);
  SmallVector<int64_t> getMaxRankDimShape();

  /// @description: Get the dimension size of the anchor
  ///
  /// This function iterates through the anchor and extracts candidates of
  /// the maximum rank dimensions.
  ///
  /// @return A vector of pairs containing the shape values and indices.
  SmallVector<SmallVector<AnchorDimension>> getAnchorShape();

  /// @description: Find the interchange mapping of this certain value v
  /// in regards of anchor
  SmallVector<int64_t> getInterchange(Value v);

  /// @description: Normalizes an interchange vector by replacing -1 values with
  /// available dimension indices.
  ///
  /// This function takes an interchange vector that may contain -1 values
  /// (indicating unspecified dimensions) and replaces them with the smallest
  /// available dimension indices to create a complete, normalized interchange
  /// mapping. This case can happen when an observed value is different from the
  /// anchor, although with the latest change this can will not happen (?)
  ///
  /// The normalization process:
  /// - Identifies all explicitly specified dimensions (non-negative values)
  /// - Finds available dimension indices that are not already used
  /// - Replaces -1 values with available indices in ascending order
  ///
  SmallVector<int64_t> getNormalizedInterchange(Value v);

  /// @description: Get the rank of the anchor.
  size_t getAnchorRank() const;

  /// @description: Check whether \c other is equivalent to the anchor's
  /// \c anchorDimIdx
  ///
  /// @param anchorDimIdx Dimension in the anchor.
  /// @param other A pair of value and dim idx.
  /// @param isStrict if set to false, two dims are considered equal if they're
  ///                 structurally equivalent (e.g., before and after concat).
  bool isDimensionEqualToAnchor(int64_t anchorDimIdx, Dimension other,
                                bool isStrict = false);

protected:
  /// Consists of (parent index of solverCollapserElem_, number of previous
  /// occurrence in one value) pairs to ensure two same indices are
  /// distinguished.
  /// For example: %arg0 = <AxBx30x20xf32>
  ///
  /// solverShapeElem : [S1, S1, S2 = 30, S3 = 20]
  ///
  /// %transpose = transpose %arg0 : <BxAx30x20xf32>
  /// solverShapeElem : [S1, S1, S2 = 30, S3 = 20]
  ///
  /// (A and B is joined in the solverShapeElem)
  /// %binary = %arg0 + %transpose : <(A or B)x(A or B)x30x20xf32>
  /// solverShapeElem : [S1, S1, S2 = 30, S3 = 20]
  ///
  /// IndexSet = {(S1, 0), (S1, 1), (S2, 0), (S3, 0)}
  using IndexSet = SetVector<std::pair<int64_t, int64_t>>;

  //===--------------------------------------------------------------------===//
  // Processors for operations
  //===--------------------------------------------------------------------===//

  /// Processes an individual operation during BFS.
  bool processOperation(Operation *op, Value current) override;

  /// Processes broadcast operations.
  void processBroadcastOp(linalg::BroadcastOp op);

  /// Processes reduce-like operations.
  template <class T, typename = std::enable_if_t<
                         std::is_same_v<T, linalg::ReduceOp> ||
                         std::is_same_v<T, hfusion::ReduceWithIndexOp>>>
  void processReduceLikeOp(T reduceOp);

  /// Processes transpose operations.
  void processTransposeOp(linalg::TransposeOp op);

  // Processes cumulative operations.
  template <class T>
  void processCumOp(T cumOp);

  /// Processes gather operations.
  void processGatherOp(hfusion::GatherOp op);

  // Process interleave and deinterleave op
  void processInterleaveOp(hfusion::InterleaveOp op);
  void processDeinterleaveOp(hfusion::DeinterleaveOp op);
  void processForOp(scf::ForOp op);
  void processYieldOp(scf::YieldOp op);

  void processToTensorOp(bufferization::ToTensorOp op);
  void processToMemrefOp(bufferization::ToMemrefOp op);
  void processSubviewOp(memref::SubViewOp op);

  void processFlipOp(hfusion::FlipOp op);
  void processSortOp(hfusion::SortOp op);
  void processEmbeddingGatherOp(hfusion::EmbeddingGatherOp op);
  void processIndirectLoadOp(hfusion::IndirectLoadOp op);
  void processIndirectStoreOp(hfusion::IndirectStoreOp op);

  //===--------------------------------------------------------------------===//
  // Helper function
  //===--------------------------------------------------------------------===//

  /// @description: Compute the index set from the given array reference.
  ///
  /// This function iterates over the given array reference and computes
  /// the index set based on the solver Collapser element. The index set
  /// is a set of pairs, where each pair contains an index and its corresponding
  /// count.
  ///
  /// @param indexSet The set to store the computed index set.
  /// @param vRef The array reference to compute the index set from.
  void computeIndexSet(IndexSet &indexSet, ArrayRef<int64_t> vRef);

  /// @description: Compute the anchor element for a given anchor candidate.
  ///
  /// @param indexAncherElemMap Map of parent index of solverCollapserElem_ to
  /// anchor element.
  /// @param anchorCandidate Anchor candidate value.
  ///
  /// This function computes the anchor element for a given anchor candidate. It
  /// uses a count to handle cases where two different axes can be in the same
  /// collapser. It also uses a DenseMap to keep track of the count of each
  /// axis. The function updates the anchorIndexSet and indexAncherElemMap
  /// accordingly.
  void
  computeAnchorElement(DenseMap<int64_t, AnchorElement> &indexAncherElemMap,
                       Value anchorCandidate);

  /// @description: Retrieves all tensor values that have been analyzed and
  /// originate from candidate anchor operations.
  ///
  /// This function walks through all operations and identifies candidate anchor
  /// operations (those matching important patterns or implementing
  /// DestinationStyleOpInterface). It then collects result tensors from these
  /// anchors that have already been dimension-analyzed.
  ///
  /// @return SmallVector<Value> Collection of analyzed tensor values from
  /// anchor operations
  SmallVector<Value> getAnchorCandidate();

#ifndef NDEBUG
  void debugPrintAnchor();
#endif

protected:
  IndexSet anchorIndexSet_;
  SmallVector<AnchorElement> anchor_;
};

} // namespace detail
} // namespace hfusion
} // namespace mlir
#endif
