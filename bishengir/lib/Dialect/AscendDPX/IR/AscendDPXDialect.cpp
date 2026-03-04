//===- AscendDPXDialect.cpp - AscendDPX dialect implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace mlir::ascend_dpx;

#include "bishengir/Dialect/AscendDPX/IR/AscendDPXEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.cpp.inc"

#include "bishengir/Dialect/AscendDPX/IR/AscendDPXDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AscendDPXDialect
//===----------------------------------------------------------------------===//

void ascend_dpx::AscendDPXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "bishengir/Dialect/AscendDPX/IR/AscendDPXOps.cpp.inc"
