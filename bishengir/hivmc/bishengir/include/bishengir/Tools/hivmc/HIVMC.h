//===- HIVMC.h - HIVMC Tool Support ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_HIVMC_HIVMC_H
#define BISHENGIR_TOOLS_HIVMC_HIVMC_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/hivmc/Utility.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <map>
#include <string>

/// Overview of the relationships between the binary tools and the effects of
/// some key compile options.
///
/// +bishengir-compile--------------+
/// |                               |
/// |     MLIR+-----------+--+      |
/// |      |  |  HFusion  +--+------+-- enable-hfusion-compile
/// |      |  +-----+-----+  |      |
/// |  bishengir-hivm-compile----+  |
/// |  |   |  +-----v-----+  |   |  |
/// |  |   |  |   HIVM    +--+---+--+----- enable-hivm-compile
/// |  |   |  +-----+-----+  |   |  |
/// |  | hivmc-----------------+ |  |
/// |  | | | +------v------+ | | |  |
/// |  | | | |    HIVM     | | | |  |
/// |  | | | |Minimal OpSet| | | |  |
/// |  | | +-+------+------+-+ | |  |
/// |  | |          +----------+-+--+------ convert-hir-to-lir
/// |  | |   +------v------+   | |  |
/// |  | |   |   LLVM IR   |   | |  |
/// |  | |   +------+------+   | |  |
/// |  | |          +----------+-+--+-----  enable-lir-compile
/// |  | |   +------v------+   | |  |
/// |  | |   |   binary    |   | |  |
/// |  | |   +-------------+   | |  |
/// |  | +---------------------+ |  |
/// |  +-------------------------+  |
/// +-------------------------------+
///
/// NEXT: After open-sourcing HIVM, the `bishengir-hivm-compile` tool
/// will be merged with `bishengir-compile`.

namespace bishengir {

std::string registerAndParseCLIOptions(int argc, char **argv);

/// Main entry point to run BiShengIR pipeline to compile module into
/// binary.
mlir::LogicalResult runHIVMCCompile(mlir::ModuleOp module,
                                   HIVMCMainConfig config);

void registerHIVMCCompilePass();

} // namespace bishengir

#endif // BISHENGIR_TOOLS_HIVMC_HIVMC_H
