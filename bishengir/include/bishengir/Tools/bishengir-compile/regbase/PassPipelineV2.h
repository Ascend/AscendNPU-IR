//===- PassPipelineV2.h - v2 driver pipeline ---------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Driver glue for --use-pipeline-version=2. The v2 pass order itself lives in
// bishengir/Dialect/HIVM/Pipelines/v2; this file only composes the HFusion,
// convert-to-hivm and v2 stages for the two driver steps.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_PASSPIPELINEV2_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_PASSPIPELINEV2_H

#include "bishengir/Tools/bishengir-compile/Config.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace bishengir {
namespace regbase {

/// v2 front path: HFusion prep, convert-to-hivm, v2 tensor phase and
/// bufferization.
void buildV2HIRPipeline(mlir::OpPassManager &pm,
                        const BiShengIRCompileMainConfig &config);

/// v2 final path: post-bufferization, cross-core solve, split and tail.
void buildV2FinalHIVMPipelines(mlir::OpPassManager &pm,
                               const BiShengIRCompileMainConfig &config);

} // namespace regbase
} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_PASSPIPELINEV2_H
