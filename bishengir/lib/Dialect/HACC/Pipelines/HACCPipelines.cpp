//===- HACCPipelines.cpp - HACC pipelines ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
