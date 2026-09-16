//===- Passes.h - Pass Entrypoints --------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANNOTATION_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_ANNOTATION_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace annotation {

#define GEN_PASS_DECL
#include "bishengir/Dialect/Annotation/Transforms/Passes.h.inc"

/// Creates a pass that lowering the Annotation dialect.
std::unique_ptr<Pass> createAnnotationLoweringPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Annotation/Transforms/Passes.h.inc"

} // namespace annotation
} // namespace mlir

#endif // BISHENGIR_DIALECT_ANNOTATION_TRANSFORMS_PASSES_H
