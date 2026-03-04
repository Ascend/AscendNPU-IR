//===----TestDialect.h - MLIR Dialect for Testing BishengIR  ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for
// testing things that do not have a respective counterpart in the
// main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TESTDIALECT_H
#define TEST_TESTDIALECT_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TestOps.h.inc"

namespace bishengir_test {
void registerTestDialect(::mlir::DialectRegistry &registry);
} // namespace bishengir_test

#endif // TEST_TESTDIALECT_H
