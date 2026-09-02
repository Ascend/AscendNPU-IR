//===- FlattenInterface.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_H

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hivm {
/// A map that describes how to group adjacent dimensions of a tensor.
/// For example, `[[0, 1], [2]]` collapses the first two dimensions of a rank-3
/// tensor into a single dimension, resulting in a rank-2 tensor.
using ReassociationMap = SmallVector<ReassociationIndices>;

/// A pair representing an operand's type and its kind (true for DPS Input).
using KindTypePair = std::pair<bool, Type>;

// Forward declarations
class HIVMStructuredOp;

/// Enumerates the different kinds of DPS operands.
enum class DpsKind { kDpsInput, kDpsInit, kDpsAll };

class FlattenOptions {
public:
  // Default constructor
  FlattenOptions() = default;
  // Destructor
  ~FlattenOptions() = default;
  bool checkMarkStride = false;
  bool checkInputConsistency = false;
  bool strictBarrierWithUnit = false;
};

/// Encapsulates the result of a shape-flattening analysis on an operation.
/// It contains the necessary information to transform the operation into its
/// flattened form, such as reassociation maps and new operand types.

/// Similar to a wrapper or a result of an operation before any collapse
/// reassociation applied to it
class FlattenResult {
public:
  explicit FlattenResult(Operation *op) : op(op){};
  Operation *op;
  /// A vector of reassociation maps. Most ops have one map applied uniformly,
  /// like (VBrc, VTranspose, etc) Permutation-like ops may have different maps
  /// for inputs and inits, for example like elementwise Transposable OTF.
  SmallVector<ReassociationMap> reassociation;

  // This maps to the operand's original value before any collapse applied to it
  SmallVector<Value> operandOriginalVal;

  /// The types of the operands after being flattened. Each element corresponds
  /// to an operand in the original op's operands.
  SmallVector<KindTypePair> operandTypes;

  SmallVector<int64_t> originalTargetDims;

  /// The updated indices for any dimension-based attributes (e.g.,
  /// `reduce_dims`) after the shape has been collapsed.
  SmallVector<int64_t> adjustedTargetDims;

  /// The updated barrier indices for any dimension-based attributes (e.g.,
  /// `reduce_dims`) after the shape has been collapsed. Usually this value is
  /// the same as adjustedDims
  SmallVector<int64_t> barrierDims;

  /// Returns the rank of the operation's shape after flattening.
  int getRankAfterFlatten() const;

  /// Returns true if the reassociation is an identity transformation, meaning
  /// no dimensions are actually collapsed.
  bool isIdentityCollapse() const;

  /// Returns the reassociation map for the input operands.
  /// Assumes the first map is for inputs.
  ReassociationMap getInputReassociation() const {
    return reassociation.front();
  }

  /// Returns the flattened types for a specific category of operands.
  /// @param kind The category of operands to retrieve (input, init, or all).
  SmallVector<Type> getOperandTypes(DpsKind kind) const;

  /// Returns the reassociation map for the init/output operands.
  /// Assumes the last map is for inits.
  ReassociationMap getInitReassociation() const { return reassociation.back(); }

  /// Returns true if all operands (inputs and inits) use the same
  /// reassociation map.
  bool uniformReassociation() const { return reassociation.size() == 1; }

  void fillWithIdentity();

  void adjustBarrierAndTargetDims(ArrayRef<int64_t> mapping);
  /// return the strided type after adding annotation mark and then collapsed
  std::optional<Type> getOperandTypeAfterFlattened(Value val);
};

namespace detail {

/// Collapses the shape of a type if it is a MemRefType, otherwise returns the
/// original type.
/// @param type The type to potentially collapse.
/// @param reassociation The reassociation map to apply.
/// @return The collapsed type or the original type.
Type collapseTypeIfMemRef(Type type,
                          ArrayRef<ReassociationIndices> reassociation);

/// Collapses the shapes of a list of types.
/// @param types The list of types to collapse.
/// @param reassociation The reassociation map to apply.
/// @return A list of collapsed types.
SmallVector<Type> collapseTypes(ArrayRef<Type> types,
                                ArrayRef<ReassociationIndices> reassociation);

/// Computes a `FlattenResult` for an operation by applying a single, uniform
/// reassociation map to all of its operands.
/// @param op The structured operation to flatten.
/// @param reassociation The uniform reassociation map to apply.
/// @return The result of the flattening analysis.
FlattenResult
collapseOperandsUniformly(HIVMStructuredOp op,
                          ArrayRef<ReassociationIndices> reassociation);

/// Computes a new `FlattenResult` by applying a further reassociation to an
/// existing `FlattenResult`.
/// @param payload The existing flatten result to build upon.
/// @param reassociation The new reassociation map to apply.
/// @return The result of the composed flattening analysis.
FlattenResult
collapseOperandsUniformly(FlattenResult &payload,
                          ArrayRef<ReassociationIndices> reassociation);

/// Computes a `FlattenResult` for an operation that has a uniform
/// reassociation map (i.e., inputs and inits are treated the same).
/// @param op The structured operation to flatten.
/// @param barrierDims The dimensions with special semantic meaning that may act
/// as barriers to collapsing.
/// @return The result of the flattening analysis.
FlattenResult collapseUniformReassociationPipeline(
    HIVMStructuredOp op, FlattenOptions &options, ArrayRef<int64_t> barrierDims,
    std::optional<ArrayRef<int64_t>> adjustedDims = std::nullopt);

/// The default implementation for the `FlattenInterface::getFlattened` method.
/// Dispatches to the correct specialized flattening logic based on op traits.
FailureOr<FlattenResult> getFlattenedImpl(Operation *op,
                                          FlattenOptions &options);

/// Get identity reassociation of HIVMStructuredOp.
FlattenResult getIdentityFlattenResult(HIVMStructuredOp hivmOp);

/// Computes the `FlattenResult` for a generic elementwise operation, handling
/// potential inline broadcast or transpose attributes.
FlattenResult getFlattenedElementwise(HIVMStructuredOp op,
                                      FlattenOptions &options);

/// Computes the `FlattenResult` for an elementwise operation that has inline
/// broadcasting.
FlattenResult getFlattenedBroadcastableOTF(HIVMStructuredOp op,
                                           FlattenOptions &options);

/// Computes the `FlattenResult` for an elementwise operation that has inline
/// transposing.
FlattenResult getFlattenedTransposableOTF(HIVMStructuredOp op,
                                          FlattenOptions &options);

/// Computes the `FlattenResult` for any operation that is known to have a
/// uniform reassociation map (e.g., rank-preserving ops).
FlattenResult getFlattenedUniformReassociation(HIVMStructuredOp op,
                                               FlattenOptions &options);

/// Computes a `FlattenResult` by collapsing only the unit dimensions of an
/// operation's shape.
/// @param op The structured operation to flatten.
/// @return A `FlattenResult` reflecting the collapsed unit dimensions and
/// adjusted target dimensions.
FlattenResult getFlattenedUnit(FlattenResult &payload, FlattenOptions &options);

/// Computes a `FlattenResult` by collapsing unit dimensions independently for
/// each operand. This is necessary for permutation-like ops where input/init
/// layouts differ.
/// @param op The structured operation to flatten.
/// @param permutationArray The permutation array of transposableOTF
/// @return A `FlattenResult` reflecting the collapsed unit dimensions.
FlattenResult
getFlattenedUnitTransposableOTF(HIVMStructuredOp op,
                                const FlattenOptions &options,
                                ArrayRef<int64_t> permutationArray);

/// Computes the limited axes for a generic elementwise operation.
SmallVector<int64_t> computeElementwiseLimitation(HIVMStructuredOp op);

/// Adjusts the target dimension attributes of a generic elementwise operation
/// after its shape has been flattened.
void adjustElementwiseTargetDimensions(OpBuilder &builder, HIVMStructuredOp op,
                                       const FlattenResult &result);

/// Composes two sequential `FlattenResult` instances into a single equivalent
/// `FlattenResult`. This is used to merge flattening steps in a pipeline.
/// @param producer The result of the first flattening operation.
/// @param consumer The result of the second flattening operation.
/// @return A single `FlattenResult` representing the combined transformation.
FlattenResult composeFlattenResults(FlattenResult producer,
                                    FlattenResult consumer,
                                    MLIRContext *context);

FlattenResult collapseUniformReassociation(FlattenResult &payload,
                                           const FlattenOptions &options);

FlattenResult computeAnnotationMarkedOp(FlattenResult payload);

/// Computes a consistency mask indicating which dimensions have the same size
/// across all shaped types in the input collection.
///
/// This function analyzes a collection of shaped types (specifically
/// MemRefType) and determines which dimensions are consistent (have the same
/// size) across all types. The consistency mask is computed by comparing each
/// dimension of all shaped types against a pivot shape (the first valid
/// MemRefType encountered).
///
/// @param shapedTypes Collection of types to analyze for shape consistency
/// @return BitVector where bit i is true if dimension i has consistent size
///         across all shaped types, false otherwise. Returns empty BitVector
///         if no valid MemRefType is found.
///
/// @note Only MemRefType instances are considered; other types are ignored
/// @note Types with different ranks than the pivot are skipped
/// @note The pivot shape is established by the first valid MemRefType
/// encountered
BitVector getInputConsistencyMask(ArrayRef<Type> shapedTypes);

} // namespace detail

} // namespace mlir::hivm

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_FLATTENINTERFACE_H
