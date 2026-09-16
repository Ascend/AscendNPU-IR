//===- AnnotationDialect.cpp - MLIR dialect for Annotation implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::annotation;

#include "bishengir/Dialect/Annotation/IR/AnnotationEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/Annotation/IR/AnnotationAttrs.cpp.inc"

#include "bishengir/Dialect/Annotation/IR/AnnotationOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/Annotation/IR/AnnotationOps.cpp.inc"

void mlir::annotation::AnnotationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/Annotation/IR/AnnotationOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/Annotation/IR/AnnotationAttrs.cpp.inc"
      >();
}
