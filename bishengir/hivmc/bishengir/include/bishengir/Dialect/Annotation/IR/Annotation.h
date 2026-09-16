//===- Annotation.h - Annotation dialect -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATION_H
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Annotation Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/AnnotationOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Annotation Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/Annotation/IR/AnnotationOps.h.inc"

#include "bishengir/Dialect/Annotation/IR/AnnotationEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/Annotation/IR/AnnotationAttrs.h.inc"

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATION_H
