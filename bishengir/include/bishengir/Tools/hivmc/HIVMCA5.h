//===- HIVMCA5.h - HIVMC Tool Support ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_HIVMC_A5_H
#define BISHENGIR_TOOLS_HIVMC_HIVMC_A5_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/hivmc/Utility.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <map>
#include <string>

namespace bishengir {


mlir::LogicalResult runHIVMCCompileA5(mlir::ModuleOp module,
                                   HIVMCMainConfig config);

mlir::LogicalResult runBiShengLIRCompileA5(
    mlir::ModuleOp module, HIVMCMainConfig config,
    const std::map<SubCoreTarget, std::string> &bitcodePaths);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_HIVMC_H
