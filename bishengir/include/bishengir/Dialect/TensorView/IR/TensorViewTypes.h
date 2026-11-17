//===- TensorViewTypes.h - TensorView types ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSORVIEW_TYPES_H
#define BISHENGIR_DIALECT_TENSORVIEW_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/TensorView/IR/TensorViewTypes.h.inc"

#endif // BISHENGIR_DIALECT_TENSORVIEW_TYPES_H
