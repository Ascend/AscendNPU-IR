//===- PassPipeline.h - BiShengIR regbase pass pipeline -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RegBase (A5) pass pipeline declarations. Migrated from AscendNPU-IR-Dev.
// These live in the regbase subdirectory to coexist with the membase (A3)
// pipeline under bishengir-compile/.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_PASSPIPELINE_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_PASSPIPELINE_H

#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Tools/bishengir-compile/Config.h"

#include "mlir/Pass/PassOptions.h"

namespace bishengir {
namespace regbase {

/// Build the pipelines of BiShengHIR from config.
void buildBiShengHIRPipeline(mlir::OpPassManager &pm,
                             const BiShengIRCompileMainConfig &config);

/// Build the pipelines of lowering BiShengHIR to LLVM from config.
void buildBiShengHIRAVEToLLVMPipeline(mlir::OpPassManager &pm,
                                      const BiShengIRCompileMainConfig &config);
void buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                              const BiShengIRCompileMainConfig &config);
void buildBiShengHIRFinishPipeline(mlir::OpPassManager &pm,
                                   const BiShengIRCompileMainConfig &config);

void buildBiShengTTIRPipeline(mlir::OpPassManager &pm,
                              const BiShengIRCompileMainConfig &config);

void buildFinalHIVMPipelines(mlir::OpPassManager &pm,
                             const BiShengIRCompileMainConfig &config);

/// Register a pass that compiles module into binary.
void registerBiShengIRCompilePass();

/// Main entry point for the regbase pipeline.
/// This orchestrates the full A5 compile flow including retry/fallback logic.
mlir::LogicalResult runRegBasePipeline(mlir::ModuleOp mod,
                                       BiShengIRCompileMainConfig config);

} // namespace regbase
} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_PASSPIPELINE_H
