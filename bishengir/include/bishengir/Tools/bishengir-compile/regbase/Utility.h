//===- Utility.h - RegBase compile utility ----------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_UTILITY_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_UTILITY_H

#include "bishengir/Tools/bishengir-compile/Config.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace bishengir {
namespace regbase {

llvm::LogicalResult inferLayoutOptimization(
    mlir::ModuleOp &module, BiShengIRCompileMainConfig &config);

llvm::LogicalResult inferMixedCV(mlir::ModuleOp &module,
                                 BiShengIRCompileMainConfig &config);

std::vector<std::string>
filterRegBaseForwardedHIVMCOptions(llvm::ArrayRef<std::string> originalCLArgs);

std::vector<std::string>
filterSharedHIVMCOptions(llvm::ArrayRef<std::string> options);

bool hasCLIArg(llvm::ArrayRef<std::string> args,
                        llvm::StringRef argName);

} // namespace regbase
} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_UTILITY_H
