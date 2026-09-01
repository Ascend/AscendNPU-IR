//===- HIVMInterfaces.h - HIVM dialect interface definitions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H

#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/LibraryFunctionOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hivm {
/// Forward declarations.
class HIVMStructuredOp;
class UnitFlagEnabledInterface;
enum class TCoreType : uint32_t;
enum class IteratorType : uint32_t;
enum class AddressSpace : uint32_t;

namespace detail {

std::optional<TCoreType> queryCoreTypeHelper(Operation *op);

/// Return positions in `iteratorTypes` that match `iteratorTypeName`.
inline void findPositionsOfType(ArrayRef<IteratorType> iteratorTypes,
                                IteratorType iteratorTypeName,
                                SmallVectorImpl<int64_t> &res) {
  for (const auto &en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == iteratorTypeName)
      res.push_back(en.index());
  }
}

/// Implementation to get HIVM Structured Op's memory effects.
void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    HIVMStructuredOp hivmOp);

// Implementation to get operands with or without extra buffer.
SmallVector<OpOperand *> getHIVMOperandsImpl(Operation *op,
                                             bool includeExtraBuffer = false);

// Implementation to get operand types with or without extra buffer.
SmallVector<Type> getHIVMOperandTypesImpl(Operation *op,
                                          bool includeExtraBuffer = false);

/// Determines which axes are contiguous in memory across multiple shaped types.
///
/// This function analyzes an array of shaped types (specifically MemRefTypes) and
/// returns a BitVector indicating which axes maintain contiguous memory layout
/// across all provided types. index i is true if axis (i - 1) is contiguous with axis i
///
/// \param shapedTypes Array of types to analyze (all presumed to have same rank)
/// \return BitVector where bit[i] is true if axis i is contiguous across all types
///
/// An axis is considered contiguous if elements along that dimension are packed
/// tightly in memory such that: stride[axis] * shape[axis] == stride[axis-1]
///
/// Notes:
/// - Axis 0 is always marked as contiguous (outermost dimension)
/// - Dynamic dimensions are conservatively marked as non-contiguous
/// - Identity layouts are skipped as they're naturally contiguous
/// - Non-MemRef types are ignored
BitVector getContiguousAxesImpl(ArrayRef<Type> shapedTypes);
BitVector getContiguousAxesImpl(Operation *op);

// Return mask of unit axes of an operation
BitVector getUnitAxesMaskImpl(MemRefType type);
BitVector getUnitAxesMaskImpl(ArrayRef<Type> types);
BitVector getUnitAxesMaskImpl(Operation *op);

// Return mask of permuted axes of an operation
BitVector getPermutedAxesMaskImpl(Operation *op);

// Implementation to get input operands with or without extra buffer.
SmallVector<OpOperand *>
getHIVMInputOperandsImpl(Operation *op, bool includeExtraBuffer = false);

/// Get Operands with target memref space
SmallVector<Value> getTargetSpaceOperandsImpl(Operation *op,
                                              AddressSpace hivmSpace,
                                              bool includeExtraBuffer);

// check if the operand is vector only at a specific index
bool isVectorOnlyOperandImpl(Operation *op, size_t idx);

/// Verify that `op` conforms to the invariants of StructuredOpInterface
LogicalResult verifyStructuredOpInterface(Operation *op);

LogicalResult verifyUnitFlagEnabledInterface(UnitFlagEnabledInterface op);

Value getUnitFlagModeLibValueImpl(UnitFlagEnabledInterface op,
                                  PatternRewriter &rewriter);

ArrayAttr getIndexingMapsImpl(HIVMStructuredOp op);

ArrayAttr getIndexingMapsElementwiseImpl(HIVMStructuredOp op);

} // namespace detail
} // namespace hivm
} // namespace mlir

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
