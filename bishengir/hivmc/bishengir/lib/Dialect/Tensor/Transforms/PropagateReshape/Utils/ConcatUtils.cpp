//===- ConcatUtils.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//

#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#define DEBUG_TYPE "tensor-propagate-expand-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir {
namespace tensor {
namespace reshape_utils {

/// Computes the expansion of a concatenation operation.
///
/// This class analyzes a tensor concatenation operation and computes how it
/// should be expanded based on the provided reassociation indices. It handles
/// both the concatenation dimension and other dimensions, ensuring that the
/// expansion is valid and computable.
class ConcatExpansionComputer {
public:
  /// Construct a ConcatExpansionComputer with the necessary context.
  ///
  /// @param rewriter The pattern rewriter used for creating new operations
  /// @param concatOp The concatenation operation to be expanded
  /// @param reassociation The reassociation indices defining how dimensions are
  /// grouped
  /// @param mixedExpandedDimSizeOpr Mixed representation of expanded dimensions
  ConcatExpansionComputer(PatternRewriter &rewriter, tensor::ConcatOp &concatOp,
                          ArrayRef<ReassociationIndices> reassociation,
                          ArrayRef<OpFoldResult> mixedExpandedDimSizeOpr)
      : rewriter(rewriter), concatOp(concatOp), reassociation(reassociation),
        mixedExpandedDimSizeOpr(mixedExpandedDimSizeOpr),
        staticExpandedDimSizeOpr(
            decomposeMixedValues(llvm::to_vector(mixedExpandedDimSizeOpr))
                .first) {
    // Collect shapes at concat dimension
    auto dim = concatOp.getDim();
    for (auto op : concatOp->getOperands()) {
      auto shapeAtDim = utils::getShape(op.getType())[dim];
      srcShapesAtDim.push_back(shapeAtDim);
    }
  }

  /// Compute the expansion of the concatenation operation.
  ///
  /// Example transformations:
  /// - [6] + [16] -> [22] expanded to [2, 3] + [2, 8] -> [2, 11]
  /// - [A] + [B] -> [?] cannot be expanded if divisibility cannot be determined
  ///
  /// @param[out] newConcatDim The new concatenation dimension after expansion
  /// @param[out] newExpandOutputShape The computed output shape for the
  /// expansion
  /// @param[out] operandsNewDimSize New dimensions for each operand after
  /// expansion
  /// @return success() if the expansion is valid and computed successfully,
  ///         failure with appropriate message otherwise
  LogicalResult compute(uint64_t &newConcatDim,
                        SmallVector<OpFoldResult> &newExpandOutputShape,
                        SmallVector<OpFoldResult> &operandsNewDimSize) {
    auto dim = concatOp.getDim();
    uint32_t accumulator = 0;
    rewriter.setInsertionPointAfter(concatOp);

    // Process each reassociation group
    for (size_t i = 0; i < reassociation.size(); i++) {
      if (dim == i) {
        // Handle expansion of concatenated dimension
        LDBG("Checking for dim");
        // Find candidate dimensions
        SmallVector<int64_t> candidateDimension =
            findCandidateDimensions(reassociation[i]);
        if (candidateDimension.empty()) {
          return rewriter.notifyMatchFailure(concatOp,
                                             "No candidate dimension");
        }
        // Process the leftmost candidate dimension
        processSelectedCandidateDimension(
            candidateDimension.front(), reassociation[i], accumulator,
            newExpandOutputShape, operandsNewDimSize, newConcatDim);
      } else {
        // Handle normal reassociation (non-concat dimensions)
        LDBG("Appending for normal reassoc");
        for (auto idx : reassociation[i]) {
          newExpandOutputShape.push_back(mixedExpandedDimSizeOpr[idx]);
        }
        accumulator += reassociation[i].size();
      }
    }

    return success();
  }

private:
  /// Find candidate dimensions that can be used for expansion in a concat
  /// operation.
  ///
  /// This function analyzes the reassociation indices and source shapes to
  /// determine which dimensions can be safely expanded. It handles special
  /// cases like single non-unit dimensions and checks divisibility constraints.
  ///
  /// @param reassocIndices The reassociation indices for the current dimension
  /// group
  /// @return A vector of candidate dimension indices that can be used for
  /// expansion. Empty if no valid candidates exist.
  SmallVector<int64_t>
  findCandidateDimensions(const ReassociationIndices &reassocIndices) {
    // This is expanding something like
    // %concat = [6] + [16] -> [22]
    // %expanded_concat = tensor.expand_shape %concat :[6] -> [2, 3]
    // We need to count the totalMultiplication, in this case, it's 6

    // %concat = [6] + [16] -> [22]
    // %expanded_concat = tensor.expand_shape %concat :[6] -> [3, 2]
    // [8, 2] + [3, 2] = [11, 2]
    // We need to count the totalMultiplication, in this case, it's 6, concat
    // dimension is 0

    // %concat = [A] + [B] -> [?]
    // %expanded_concat = tensor.expand_shape %concat :[?] -> [2, ?]
    // %expanded_concat = tensor.expand_shape %concat :[?] -> [?, 2] we can't
    // tell if A or B is divisible by 2... so its no, also cant tell if it's
    // divisible by ? for the concat dimension

    // Special case if there's exactly
    // one dynamic dimension and all else are units
    SmallVector<int64_t> candidateDimension;
    int64_t unitCount = 0;
    int64_t dynamicCount = 0;
    int64_t totalMultiplication = 1;

    for (auto idx : reassocIndices) {
      if (staticExpandedDimSizeOpr[idx] == 1)
        unitCount++;
      if (ShapedType::isDynamic(staticExpandedDimSizeOpr[idx])) {
        dynamicCount++;
      } else {
        totalMultiplication *= staticExpandedDimSizeOpr[idx];
      }
    }

    int64_t rank = static_cast<int64_t>(reassocIndices.size());
    // Special case: only one non-unit dimension
    if (unitCount == rank - 1) {
      for (auto idx : reassocIndices) {
        if (staticExpandedDimSizeOpr[idx] != 1) {
          candidateDimension.push_back(idx);
        }
      }
      return candidateDimension;
    }

    // Cannot handle dynamic dimensions
    if (dynamicCount != 0) {
      return candidateDimension; // empty
    }

    // Check each dimension for divisibility
    for (auto idx : reassocIndices) {
      LDBG("Checking reassociation i, j " << idx);
      int64_t currentMultiplier =
          totalMultiplication / staticExpandedDimSizeOpr[idx];
      bool canBeNewDimension = true;

      for (int64_t shapeAtDim : srcShapesAtDim) {
        if (currentMultiplier == 1)
          continue;
        if (ShapedType::isDynamic(shapeAtDim) ||
            (shapeAtDim % currentMultiplier != 0)) {
          canBeNewDimension = false;
          break;
        }
      }

      if (canBeNewDimension) {
        candidateDimension.push_back(idx);
      }
    }

    return candidateDimension;
  }

  /// Process the selected candidate dimension for expansion.
  ///
  /// This function computes the new shapes for the expanded concat operation
  /// based on the selected candidate dimension. It handles the calculation of
  /// multipliers, shape contributions, and dynamic dimension detection.
  ///
  /// @param candidateDim The selected candidate dimension index
  /// @param reassocIndices The reassociation indices for the current dimension
  /// group
  /// @param accumulator Current position in the accumulated dimension count
  /// @param[out] newExpandOutputShape The computed output shape for the
  /// expansion
  /// @param[out] operandsNewDimSize New dimensions for each operand after
  /// expansion
  /// @param[out] newConcatDim The new concatenation dimension index after
  /// expansion
  void processSelectedCandidateDimension(
      int64_t candidateDim, const ReassociationIndices &reassocIndices,
      uint32_t &accumulator, SmallVector<OpFoldResult> &newExpandOutputShape,
      SmallVector<OpFoldResult> &operandsNewDimSize, uint64_t &newConcatDim) {
    int64_t totalMultiplication = 1;
    for (auto idx : reassocIndices) {
      if (!ShapedType::isDynamic(staticExpandedDimSizeOpr[idx])) {
        totalMultiplication *= staticExpandedDimSizeOpr[idx];
      }
    }

    for (auto idx : reassocIndices) {
      if (idx == candidateDim) {
        int64_t totalledNewDimension = 0;
        newConcatDim = accumulator;
        int64_t currentMultiplier = 1;
        bool hasDynamic = false;

        if (!ShapedType::isDynamic(staticExpandedDimSizeOpr[idx])) {
          currentMultiplier =
              totalMultiplication / staticExpandedDimSizeOpr[idx];
        }

        for (int64_t shapeAtDim : srcShapesAtDim) {
          auto currentShapeContribution = shapeAtDim / currentMultiplier;
          if (!ShapedType::isDynamic(currentShapeContribution)) {
            totalledNewDimension += currentShapeContribution;
          } else {
            hasDynamic = true;
          }
          operandsNewDimSize.push_back(
              rewriter.getI64IntegerAttr(currentShapeContribution));
        }

        newExpandOutputShape.push_back(rewriter.getI64IntegerAttr(
            hasDynamic ? ShapedType::kDynamic : totalledNewDimension));
      } else {
        newExpandOutputShape.push_back(mixedExpandedDimSizeOpr[idx]);
        accumulator++;
      }
    }
  }

  // Member variables
  PatternRewriter &rewriter;
  tensor::ConcatOp &concatOp;
  ArrayRef<ReassociationIndices> reassociation;
  ArrayRef<OpFoldResult> mixedExpandedDimSizeOpr;
  SmallVector<int64_t> staticExpandedDimSizeOpr;
  SmallVector<int64_t> srcShapesAtDim;
};

/// Public API function that creates a ConcatExpansionComputer and runs the
/// computation.
///
/// This function serves as the main entry point for computing the expansion of
/// a concatenation operation. It creates an instance of ConcatExpansionComputer
/// and delegates the actual computation to it.
///
/// @param rewriter The pattern rewriter used for creating new operations
/// @param concatOp The concatenation operation to be expanded
/// @param reassociation The reassociation indices defining how dimensions are
/// grouped
/// @param mixedExpandedDimSizeOpr Mixed representation of expanded dimensions
/// @param[out] newConcatDim The new concatenation dimension after expansion
/// @param[out] newExpandOutputShape The computed output shape for the expansion
/// @param[out] operandsNewDimSize New dimensions for each operand after
/// expansion
/// @return success() if the expansion is valid and computed successfully,
///         failure with appropriate message otherwise
LogicalResult
computeExpandConcat(PatternRewriter &rewriter, tensor::ConcatOp &concatOp,
                    ArrayRef<ReassociationIndices> reassociation,
                    ArrayRef<OpFoldResult> mixedExpandedDimSizeOpr,
                    uint64_t &newConcatDim,
                    SmallVector<OpFoldResult> &newExpandOutputShape,
                    SmallVector<OpFoldResult> &operandsNewDimSize) {
  ConcatExpansionComputer computer(rewriter, concatOp, reassociation,
                                   mixedExpandedDimSizeOpr);
  return computer.compute(newConcatDim, newExpandOutputShape,
                          operandsNewDimSize);
}

} // namespace reshape_utils
} // namespace tensor
} // namespace mlir
