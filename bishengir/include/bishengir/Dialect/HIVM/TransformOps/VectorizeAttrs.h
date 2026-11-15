//===- VectorizeAttrs.h - Vectorizer-owned metadata -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_HIVM_VECTORIZE_ATTRS_H
#define BISHENGIR_HIVM_VECTORIZE_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeDialect.h.inc"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeAttrs.h.inc"

#endif
