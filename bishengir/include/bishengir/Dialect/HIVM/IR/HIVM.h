//===- HIVM.h - Hybrid Intelligence Virtual Machine Dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVM_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVM_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

//===----------------------------------------------------------------------===//
// HIVM Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMDialect.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMEnums.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Types
//===----------------------------------------------------------------------===//

// generated type declarations
#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMTypes.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMAttrs.h.inc"

//===----------------------------------------------------------------------===//
// HIVM Trait and Interface
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"

//===----------------------------------------------------------------------===//
// HIVM Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMIntrinOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.h.inc"

namespace mlir {
class TypeConverter;

namespace hivm {
//===----------------------------------------------------------------------===//
// Printing/parsing for EventID
//===----------------------------------------------------------------------===//

ParseResult
parseEventID(OpAsmParser &parser, EventAttr &eventIDAttr,
             std::optional<OpAsmParser::UnresolvedOperand> &eventIDValue);

void printEventID(OpAsmPrinter &printer, Operation *op, EventAttr eventIDAttr,
                  Value eventIDValue);

//===----------------------------------------------------------------------===//
// Printing/parsing for FlagID
//===----------------------------------------------------------------------===//

ParseResult
parseFlagID(OpAsmParser &parser, IntegerAttr &flagIDAttr,
            std::optional<OpAsmParser::UnresolvedOperand> &flagIDValue);

void printFlagID(OpAsmPrinter &printer, Operation *op, IntegerAttr flagIDAttr,
                 Value flagIDValue);

//===----------------------------------------------------------------------===//
// Printing/parsing for SyncID
//===----------------------------------------------------------------------===//

ParseResult
parseSyncID(OpAsmParser &parser, IntegerAttr &syncIDAttr,
            std::optional<OpAsmParser::UnresolvedOperand> &syncIDValue);

void printSyncID(OpAsmPrinter &printer, Operation *op, IntegerAttr syncIDAttr,
                 Value syncIDValue);

namespace detail {

//===----------------------------------------------------------------------===//
// Printing/parsing for Structured Op
//===----------------------------------------------------------------------===//

/// Printer and Parser for HIVM Ops that follows Destination Style Op Interface
/// \note Only applicable for ops that only have input and init operands.
ParseResult parseHIVMStructuredDPSOp(OpAsmParser &parser,
                                     OperationState &result);
void printHIVMStructuredDPSOp(OpAsmPrinter &p, Operation *op, ValueRange inputs,
                              ValueRange outputs);

/// Return the elementType as string for library call name.
std::string getTypeName(Location loc, Type type,
                        hivm::TypeFn casting = hivm::TypeFn::cast_signed);
                        
} // namespace detail

/// Populates rules for lowering HIVM AddressSpaceAttribute to integer
/// values.
void populateHIVMAddressSpaceAttributeConversions(TypeConverter &typeConverter);

/// Get HIVM Address Space Attr from input type.
AddressSpaceAttr getHIVMAddressSpaceAttr(Type type);

/// Get HIVM Address Space from input type.
AddressSpace getHIVMAddressSpace(Type type);

/// Judge whether input type has HIVM Address Space.
std::optional<AddressSpace> getOptionalHIVMAddressSpace(Type type);

/// Infer TCoreType based on pipes
std::optional<TCoreType> inferCoreTypeBasedOnPipes(ArrayRef<hivm::PIPE> pipes);

/// Check whether the inferred core type based on pipe matches the given core
/// type. If strict is false, CUBE_OR_VECTOR is considered matching any core
/// type.
bool checkPipeInferredCoreType(hivm::PIPE pipe, hivm::TCoreType coreType,
                               bool strict = false);

/// Infer core type for GlobalMixMatmulOps
template <typename GlobalMixMatmulTy>
std::optional<TCoreType>
inferCoreTypeForGlobalMixMatmulOps(GlobalMixMatmulTy *mixMatmulOp);

constexpr llvm::StringLiteral kMultibufferUnrollAttrName =
    "multibuffer_unroll_factor";
constexpr llvm::StringLiteral kPipelinedLoopCoreTypeAttrName =
    "hivm.loop_core_type";
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVM_H
