//===- SymbolDialect.cpp - Implementation of Symbol dialect and types -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::symbol;

#define GET_OP_CLASSES
#include "bishengir/Dialect/Symbol/IR/SymbolOps.cpp.inc"

void mlir::symbol::SymbolDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/Symbol/IR/SymbolOps.cpp.inc"
      >();
}

#include "bishengir/Dialect/Symbol/IR/SymbolDialect.cpp.inc"
