//===- HFusionDialect.cpp - Implementation of HFusion dialect and types ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionAttrs.cpp.inc"

void mlir::hfusion::HFusionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HFusion/IR/HFusionOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HFusion/IR/HFusionAttrs.cpp.inc"
      >();

  declarePromisedInterfaces<bufferization::BufferizableOpInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
}

#include "bishengir/Dialect/HFusion/IR/HFusionEnums.cpp.inc"

#include "bishengir/Dialect/HFusion/IR/HFusionOpsDialect.cpp.inc"
