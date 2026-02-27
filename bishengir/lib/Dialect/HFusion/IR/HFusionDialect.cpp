//===- HFusionDialect.cpp - Implementation of HFusion dialect and types ---===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Config/bishengir-config.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
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
