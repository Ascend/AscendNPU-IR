//===- HFusion.h - Hybrid Fusion dialect ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H

#include "bishengir/Dialect/HMAP/IR/HMAP.h"
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

namespace mlir {
namespace hfusion {
std::string generateLibraryCallName(Operation *op);
} // namespace hfusion
} // namespace mlir

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
