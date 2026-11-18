//===- HIVMCA5.h - HIVMC Tool Support ------------------------------*-
// C++-*-===//
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

#include "llvm/Support/Error.h"

#include <map>
#include <string>

namespace bishengir {

/// True when \p err reports a device kernel whose outlined vector function
/// is taken to have overflowed the VF stack. The one distinction callers
/// need: a smaller vector function may well fix it, so the compile driver
/// can retry with the VF stack limit, whereas retrying a failure in any
/// other stage or tool is pointless. Every stage reports its own details on
/// stderr, so the error itself carries nothing further.
///
/// "Taken to have" because the signal is that the bisheng invocation failed,
/// not the overflow itself; see the comment where it is raised.
///
/// Leaves \p err unchecked, so the caller still owns consuming it.
bool isVFStackOverflowError(const llvm::Error &err);

llvm::Error runHIVMCCompileA5(mlir::ModuleOp module, HIVMCMainConfig config);

llvm::Error runBiShengLIRCompileA5(
    mlir::ModuleOp module, HIVMCMainConfig config,
    const std::map<SubCoreTarget, std::string> &bitcodePaths);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_HIVMC_H
