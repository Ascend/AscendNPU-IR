//===- HACCPipelines.cpp - HACC pipelines ---------------------------------===//
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

#include "bishengir/Conversion/HACCToLLVM/HACCToLLVM.h"
#include "bishengir/Dialect/HACC/Pipelines/Passes.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/LLVMIR/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hacc {

void buildLowerHACCToLLVMPipeline(OpPassManager &pm,
                                  std::string tmpDeviceBinName) {
  pm.addPass(LLVM::createParameterPackingPass());
  ConvertHACCToLLVMOptions hacc2llvmOptions;
  hacc2llvmOptions.tempDeviceLLVMFilePath = tmpDeviceBinName;
  pm.addPass(createConvertHACCToLLVMPass(hacc2llvmOptions));
}

} // namespace hacc
} // namespace mlir
