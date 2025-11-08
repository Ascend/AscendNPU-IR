//===- BiShengIRHIVMCompile.h - BiShengIR HIVM Compile Tool Support  C++-*-===//
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

#ifndef BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_BISHENGIRHIVMCOMPILE_H
#define BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_BISHENGIRHIVMCOMPILE_H

#include "bishengir/Tools/bishengir-hivm-compile/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"

namespace bishengir {

using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

/// Build the pipelines of BiShengHIR from config.
void buildBiShengHIRHIVMPipeline(mlir::OpPassManager &pm,
                                 const BiShengIRHIVMCompileMainConfig &config);

/// Main entry point to run BiShengIR pipeline to compile module into binary.
llvm::FailureOr<OwningModuleRef>
runBiShengIRHIVMPipeline(mlir::ModuleOp module,
                         BiShengIRHIVMCompileMainConfig config);

} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_HIVM_COMPILE_BISHENGIRHIVMCOMPILE_H
