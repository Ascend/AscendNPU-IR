//===- PassPipeline.h - BiShengIR pass pipeline------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_PASSPIPELINE_H
#define BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_PASSPIPELINE_H

#include "bishengir/Tools/bishengir-compile/Config.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace hivm {
struct HIVMPipelineOptions;
}
} // namespace mlir

namespace bishengir {

// Helper function to set up HIVMPipelineOptions
void setupHIVMPipelineOptions(hivm::HIVMPipelineOptions &hivmPipelineOptions,
                              const BiShengIRCompileMainConfig &config);

/// Build the pipelines of BiShengHIR from config.
void buildBiShengHIRHIVMPipeline(mlir::OpPassManager &pm,
                                 const BiShengIRCompileMainConfig &config);

/// Register a pass that compiles module into binary.
void registerBiShengIRHIVMCompilePass();

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_PASSPIPELINE_H
