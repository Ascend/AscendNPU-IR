//===- HIVM.h - Hybrid Intelligence Virtual Machine Dialect -----*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVM_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVM_H

#include "bishengir/Interfaces/AggregatedOpInterface.h"
#include "bishengir/Interfaces/CopyOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <string>

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

constexpr llvm::StringLiteral kCVUnrolledLoopName =
    "cv_unrolled_loop";
constexpr llvm::StringLiteral kMultibufferUnrollAttrName =
    "multibuffer_unroll_factor";
constexpr llvm::StringLiteral kPipelinedLoopCoreTypeAttrName =
    "hivm.loop_core_type";
constexpr llvm::StringLiteral kPreLoadAttrName =
    "preload_num";

/// Suffixes appended by SplitMixKernel when cloning a MIX function into AIC/AIV
/// copies (e.g. `kernel` -> `kernel_mix_aic` / `kernel_mix_aiv`).
constexpr llvm::StringLiteral kMixFuncAicSuffix = "_mix_aic";
constexpr llvm::StringLiteral kMixFuncAivSuffix = "_mix_aiv";

/// Tags attached by SplitMixedIfConditionals on uniform-core `scf.if`s so
/// SplitMixKernel can filter whole ifs as CUBE or VECTOR.
constexpr llvm::StringLiteral kCubeOnlyAttrName = "hivm.cube_only";
constexpr llvm::StringLiteral kVecOnlyAttrName = "hivm.vec_only";

inline bool hasMixFuncSuffix(llvm::StringRef name) {
  return name.ends_with(kMixFuncAicSuffix) || name.ends_with(kMixFuncAivSuffix);
}

/// Strip a trailing `_mix_aic` / `_mix_aiv` suffix if present.
/// Returns an owning string so callers are not tied to the lifetime of `name`
/// (a bare `StringRef` from `drop_back` would dangle if `name` is temporary).
inline std::string stripMixFuncSuffix(llvm::StringRef name) {
  if (name.ends_with(kMixFuncAicSuffix))
    return name.drop_back(kMixFuncAicSuffix.size()).str();
  if (name.ends_with(kMixFuncAivSuffix))
    return name.drop_back(kMixFuncAivSuffix.size()).str();
  return name.str();
}

/// Attribute placed on a memref.alloca holding the multi-buffer iteration
/// counter for a particular scf.while loop. Its IntegerAttr value matches the
/// kMultiBufferLoopIdAttr placed on the owning scf.while op so multiple
/// passes (GraphSyncSolver, EnableMultiBuffer) can locate and reuse the same
/// counter without sharing pass-level state.
constexpr llvm::StringLiteral kMultiBufferCounterAttr =
    "hivm.multi_buffer_counter_for";

/// Attribute placed on an scf.while op once a counter alloca has been
/// associated with it. The IntegerAttr value is unique within the parent
/// FunctionOpInterface.
constexpr llvm::StringLiteral kMultiBufferLoopIdAttr =
    "hivm.multi_buffer_loop_id";

constexpr llvm::StringLiteral kMixFuncAicSuffix = "_mix_aic";
constexpr llvm::StringLiteral kMixFuncAivSuffix = "_mix_aiv";
constexpr llvm::StringLiteral kFuncBackupSuffix = "_backup";
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVM_H
