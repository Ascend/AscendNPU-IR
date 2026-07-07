//===- PassPipeline.h - BiShengIR pass pipeline------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_PASSPIPELINE_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_PASSPIPELINE_H

#include "bishengir/Dialect/HIVM/Pipelines/Passes.h"
#include "bishengir/Tools/bishengir-compile/Config.h"

#include "mlir/Pass/PassOptions.h"


namespace bishengir {

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

/// Build the SIMT pipeline.
void buildSIMTPipeline(mlir::OpPassManager &pm,
                       const BiShengIRCompileMainConfig &config);


/// Register a pass that compiles module into binary.
void registerBiShengIRCompilePass();

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_PASSPIPELINE_H
