//===- TensorViewInterfaces.h - TensorView op interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSORVIEW_INTERFACES_H
#define BISHENGIR_DIALECT_TENSORVIEW_INTERFACES_H

#include "mlir/IR/OpDefinition.h"

// Prerequisites referenced by the interface default method bodies.
#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewTypes.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewInterfaces.h.inc"

#endif // BISHENGIR_DIALECT_TENSORVIEW_INTERFACES_H
