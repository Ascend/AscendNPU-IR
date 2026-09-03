//===- HACC.h - Heterogeneous Async Computing Call dialect ------*- C++ -*-===//
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
