//===- AVE.cpp - AVE ops implementation -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hivmave;

#include "bishengir/Dialect/HIVMAVE/IR/AVEEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
// Suppress warnings when these static functions unused.
LLVM_ATTRIBUTE_UNUSED static OptionalParseResult
generatedTypeParser(AsmParser &parser, StringRef *mnemonic, Type &value);
LLVM_ATTRIBUTE_UNUSED static LogicalResult
generatedTypePrinter(Type def, AsmPrinter &printer);
#include "bishengir/Dialect/HIVMAVE/IR/AVETypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVMAVE/IR/AVEAttrs.cpp.inc"

#include "bishengir/Dialect/HIVMAVE/IR/AVEDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AVEDialect
//===----------------------------------------------------------------------===//

void hivmave::AVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVMAVE/IR/AVEOps.cpp.inc"
      >();
  // uncomment when adding types
  addTypes<
#define GET_TYPEDEF_LIST
#include "bishengir/Dialect/HIVMAVE/IR/AVETypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HIVMAVE/IR/AVEAttrs.cpp.inc"
      >();
}