//===- MathExt.h - MathExt dialect -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_MATH_IR_MATHEXT_H
#define BISHENGIR_DIALECT_MATH_IR_MATHEXT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/Dialect/Math/IR/Math.h"

//===----------------------------------------------------------------------===//
// Math Ext Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MathExt/IR/MathExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Math Ext Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/MathExt/IR/MathExtOps.h.inc"

#endif // BISHENGIR_DIALECT_MATH_IR_MATHEXT_H
