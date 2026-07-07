//===- HIVMRegbaseIntrins.cpp - Implementation of hivm regbase dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace hivm_regbaseintrins;

#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrinsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseAttrs.cpp.inc"

void hivm_regbaseintrins::HIVMRegbaseIntrinsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseAttrs.cpp.inc"
      >();
}
