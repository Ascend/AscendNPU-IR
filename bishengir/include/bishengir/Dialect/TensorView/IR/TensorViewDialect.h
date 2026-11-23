//===- TensorViewDialect.h - MLIR TensorView dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TensorView (`tv`) dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSORVIEW_DIALECT_H
#define BISHENGIR_DIALECT_TENSORVIEW_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewInterfaces.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewTypes.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/TensorView/IR/TensorViewOps.h.inc"

#endif // BISHENGIR_DIALECT_TENSORVIEW_DIALECT_H
