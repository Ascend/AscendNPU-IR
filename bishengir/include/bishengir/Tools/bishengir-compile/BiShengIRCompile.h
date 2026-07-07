//===- BiShengIRCompile.h - BiShengIR Compile Tool Support -------*- C++-*-===//
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

#ifndef BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H
#define BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Tools/bishengir-compile/Config.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace bishengir {

using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

/// Main entry point to run BiShengIR pipeline to compile module into binary.
llvm::FailureOr<OwningModuleRef>
runBiShengIRPipeline(mlir::ModuleOp mod, BiShengIRCompileMainConfig config);

/// Register and parse command line options.
/// \return the input file path
std::string registerAndParseCLIOptions(int argc, char **argv);

/// Main entry point to run BiShengIR pipeline to compile module into binary.
mlir::LogicalResult runBiShengIRPipeline(mlir::ModuleOp module,
                                         BiShengIRCompileMainConfig config);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIRCOMPILE_BISHENGIRCOMPILE_H
