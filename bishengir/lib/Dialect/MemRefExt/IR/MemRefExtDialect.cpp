//===- MemRefExtDialect.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include <optional>

using namespace mlir;
using namespace bishengir::memref_ext;

#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOpsDialect.cpp.inc"

void bishengir::memref_ext::MemRefExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"
      >();
  declarePromisedInterface<ConvertToLLVMPatternInterface, MemRefExtDialect>();
}