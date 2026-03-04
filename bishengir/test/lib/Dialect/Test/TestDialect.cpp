//===- TestDialect.cpp - MLIR Dialect for Testing BishengIR  ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "InitTestDialect.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/TypeUtilities.h"

// Include this before the using namespace lines below to
// test that we don't have namespace dependencies.
#include "TestOpsDialect.cpp.inc"

namespace bishengir_test {

void TestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
}

void registerTestDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

} // namespace bishengir_test

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"