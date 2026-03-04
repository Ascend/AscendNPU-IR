//===- HACC.h - Heterogeneous Async Computing Call dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACC_H
#define BISHENGIR_DIALECT_HACC_IR_HACC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

//===----------------------------------------------------------------------===//
// HACC Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACCEnums.h.inc"

#include "bishengir/Dialect/HACC/IR/HACCBaseDialect.h.inc"

// generated type declarations
#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/HACC/IR/HACCTypes.h.inc"

//===----------------------------------------------------------------------===//
// HACC Interfaces
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACCInterfaces.h"

//===----------------------------------------------------------------------===//
// HACC Target Specifications
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.h.inc"

//===----------------------------------------------------------------------===//
// HACC Attributes
//===----------------------------------------------------------------------===//

// Attributes are dependent on Interface
#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HACC/IR/HACCAttrs.h.inc"

namespace mlir {
namespace hacc {

namespace func_ext {
void registerHACCDialectExtension(DialectRegistry &registry);
} // namespace func_ext

namespace llvm_ext {
void registerHACCDialectExtension(DialectRegistry &registry);
} // namespace llvm_ext

} // namespace hacc
} // namespace mlir

#endif // BISHENGIR_DIALECT_HACC_IR_HACC_H
