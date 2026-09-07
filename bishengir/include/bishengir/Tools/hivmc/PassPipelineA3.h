//===- PassPipeline.h - BiShengIR HIVMC pass pipeline------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_PASSPIPELINE_A3_H
#define BISHENGIR_TOOLS_HIVMC_PASSPIPELINE_A3_H

#include "bishengir/Tools/hivmc/Config.h"
#include "mlir/Pass/PassOptions.h"

namespace bishengir {

void buildBiShengHIRHIVMToLLVMPipeline(mlir::OpPassManager &pm,
                                       const HIVMCMainConfig &config);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_PASSPIPELINE_H
