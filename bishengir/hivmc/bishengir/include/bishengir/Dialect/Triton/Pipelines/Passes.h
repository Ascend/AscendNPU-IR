//===- Passes.h - Triton pipeline entry points ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all Triton pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITON_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_TRITON_PIPELINES_PASSES_H

#include "bishengir/Tools/hivmc/Config.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace bishengir {
namespace triton {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the pipeline to lower Triton SIMT to LLVM.
void buildLowerSIMTToLLVMPipeline(
    mlir::OpPassManager &pm,
    const bishengir::HIVMCMainConfig &config);

} // namespace triton
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRITON_PIPELINES_PASSES_H
