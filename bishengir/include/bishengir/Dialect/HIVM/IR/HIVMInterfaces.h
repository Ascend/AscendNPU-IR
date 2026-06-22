//===- HIVMInterfaces.h - HIVM dialect interface definitions ----*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H

#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/LibraryFunctionOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.h"
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
enum class AlignKind : uint32_t;

/// Deduce Alignment information for DPS Op's init operand.
///
/// If operand has memref semantic, we try to deduce the information from the
/// memref type. Otherwise, we look for annotations on the tied result value. If
/// there is conflicting annotations, a warning is produced.
AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand);
AlignKind deduceAlignmentForMemRefType(MemRefType vecType);
bool hasHWUnsupportedScalarOperandImpl(HIVMStructuredOp op);

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

// Return mask of continuous axes of an operation
BitVector getContiguousAxesImpl(Operation *op);
BitVector getContiguousAxesImpl(ArrayRef<Type> shapedTypes);

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
