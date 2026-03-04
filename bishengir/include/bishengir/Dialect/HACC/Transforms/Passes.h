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

#ifndef BISHENGIR_DIALECT_HACC_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_HACC_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hacc {

#define GEN_PASS_DECL
#include "bishengir/Dialect/HACC/Transforms/Passes.h.inc"

/// Create a pass to rename function name.
std::unique_ptr<Pass> createRenameFuncPass();

/// Create a pass to append target spec. information to the top-level module.
std::unique_ptr<Pass>
createAppendDeviceSpecPass(const AppendTargetDeviceSpecOptions &options = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HACC/Transforms/Passes.h.inc"

} // namespace hacc
} // namespace mlir

#endif // BISHENGIR_DIALECT_HACC_TRANSFORMS_PASSES_H
