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

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_PASSES_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_PASSES_H

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Analysis/VFFusion/Passes.h.inc"
namespace analysis {

/// Creates a pass to fuse vf function.
std::unique_ptr<Pass> createVFFusionPass(const VFFusionOptions &option = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Analysis/VFFusion/Passes.h.inc"

} // namespace analysis
} // namespace mlir

#endif // BISHENGIR_DIALECT_ANALYSIS_VFFUSION_PASSES_H
