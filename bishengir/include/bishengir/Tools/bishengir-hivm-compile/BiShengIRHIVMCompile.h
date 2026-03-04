//===- BiShengIRHIVMCompile.h - BiShengIR HIVM Compile Tool Support  C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_BISHENGIRHIVMCOMPILE_H
#define BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_BISHENGIRHIVMCOMPILE_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/bishengir-compile/Config.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace bishengir {

/// Register and parse command line options.
/// \return the input file path
std::string registerAndParseCLIOptions(int argc, char **argv);

/// Main entry point to run BiShengIR pipeline to compile module into binary.
mlir::LogicalResult runBiShengIRHIVMPipeline(mlir::ModuleOp hirCompileModule,
                                             BiShengIRCompileMainConfig config);
} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_BISHENGIRHIVMCOMPILE_H
