//===- HFusion.h - Hybrid Fusion dialect ------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H

#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#endif
#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// HFusion Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusionOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// HFusion Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusionEnums.h.inc"

//===----------------------------------------------------------------------===//
// HFusion Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionAttrs.h.inc"

//===----------------------------------------------------------------------===//
// HFusion Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.h.inc"

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H
