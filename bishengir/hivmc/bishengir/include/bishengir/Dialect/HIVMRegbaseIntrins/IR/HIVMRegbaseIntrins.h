//===- HIVMRegbaseIntrins.h - Regbase dialect definitions --*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVMREGBASEINTRINS_IR_HIVMREGBASEINTRINS_H
#define BISHENGIR_DIALECT_HIVMREGBASEINTRINS_IR_HIVMREGBASEINTRINS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// generated dialect declaration
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrinsDialect.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseAttrs.h.inc"

namespace mlir {
namespace hivm_regbaseintrins {
constexpr char kDavinciCallingConvAttrName[] = "hivm_regbaseintrins.cconv";
constexpr char kDavinciTargetAttrName[] = "hivm_regbaseintrins.target";
constexpr char kDavinciKernelAttrName[] = "hivm_regbaseintrins.kernel";
} // namespace hivm_regbaseintrins
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVMREGBASEINTRINS_IR_HIVMREGBASEINTRINS_H
