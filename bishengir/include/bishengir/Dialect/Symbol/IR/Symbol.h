//===- Symbol.h - Symbol Dialect ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SYMBOL_IR_SYMBOL_H
#define BISHENGIR_DIALECT_SYMBOL_IR_SYMBOL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Symbol Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/SymbolDialect.h.inc"

//===----------------------------------------------------------------------===//
// Symbol Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/Symbol/IR/SymbolOps.h.inc"

#endif // BISHENGIR_DIALECT_SYMBOL_IR_SYMBOL_H