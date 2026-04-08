//===- TritonExtAttrs.h - TritonExt attribute declarations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TritonExt dialect attributes, including the
// FractalSharedEncodingAttr for Ascend Cube fractal shared memory layouts.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITONEXT_IR_TRITONEXTATTRS_H
#define BISHENGIR_DIALECT_TRITONEXT_IR_TRITONEXTATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

// Dialect declaration must come before attribute definitions
#include "bishengir/Dialect/TritonExt/IR/Dialect.h.inc"

#include "bishengir/Dialect/TritonExt/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/TritonExt/IR/AttrDefs.h.inc"

#endif // BISHENGIR_DIALECT_TRITONEXT_IR_TRITONEXTATTRS_H
