//===- PassPipeline.h - BiShengIR HIVMC pass pipeline------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_PASSPIPELINE_H
#define BISHENGIR_TOOLS_HIVMC_PASSPIPELINE_H

#include "bishengir/Tools/hivmc/Config.h"
#include "mlir/Pass/PassOptions.h"

namespace bishengir {

/// Build the pipelines of lowering HIVM To LLVM from config.
void buildBiShengHIRAVEToLLVMPipeline(mlir::OpPassManager &pm,
                                      const HIVMCMainConfig &config);
void buildLowerToLLVMPipeline(OpPassManager &pm, const HIVMCMainConfig &config);

/// Build SIMT SIMD pipeline
void buildSIMTPipeline(OpPassManager &pm, const HIVMCMainConfig &config);

void buildFinalMixVFCompilePipeline(OpPassManager &pm, const HIVMCMainConfig &config);

/// Register a pass that compiles module into binary.
void registerHIVMCCompilePass();

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_PASSPIPELINE_H
