//===- BMU.h - BMU dialect -------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BMU_IR_BMU_H
#define BISHENGIR_DIALECT_BMU_IR_BMU_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// ODS-generated declarations.
#include "bishengir/Dialect/BMU/IR/BMUDialect.h.inc"
#include "bishengir/Dialect/BMU/IR/BMUEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUAttrs.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/BMU/IR/BMUIntrinOps.h.inc"

namespace mlir::bmu {

/// Buffer metadata: maps buffer enum to hardware properties.
struct BMUBufferInfo {
  uint32_t bufField;     // 3-bit buffer field encoding
  uint32_t addressSpace; // LLVM pointer address space
  StringRef allocPrefix; // intrinsic name prefix for alloc
  StringRef sprSuffix;   // SPR name suffix
};

/// Get buffer info for a given buffer enum value. Returns nullptr if
/// the buffer is not supported.
const BMUBufferInfo *getBMUBufferInfo(Buffer buffer);

/// Get the pipe encoding as i64 integer.
int64_t getPipeEncoding(Pipe pipe);

} // namespace mlir::bmu

#endif // BISHENGIR_DIALECT_BMU_IR_BMU_H
