//===- AdapterSanitizer.cpp - Mssanitizer enabling --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some funcs used to enable mssanitizer
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/DIBuilder.h"

#define DEBUG_TYPE "enable-sanitizer"

namespace bishengir {
using namespace mlir;
// Enabled when we need sanitizer
// The name of the arg need to be .arg_address_sanitizer_gm_ptr
LogicalResult
setSanitizerAddrArgName(mlir::ModuleOp module,
                        const std::unique_ptr<llvm::Module> &llvmModule) {
  // First: Find the kernel funcs with sanitizer input arg
  //        get the index of arg in the LLVM func OP
  DenseMap<StringRef, int> argsToRenameIdxMap;
  module.walk([&](Operation *op) {
    if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      if (!hacc::utils::isDevice(funcOp)) {
        return WalkResult::advance();
      }
      StringRef kernelName = funcOp.getSymName();
      auto funcParamSize = funcOp.getNumArguments();
      for (size_t i = 0; i < funcParamSize; i++) {
        if (hacc::utils::isKernelArg(funcOp, i,
                                     hacc::KernelArgType::kSanitizerAddr)) {
          argsToRenameIdxMap.insert({kernelName, i});
          return WalkResult::advance();
        }
      }
    }
    return WalkResult::advance();
  });
  if (!llvmModule) {
    return failure();
  }
  // second: Modify their name in the LLVM IR
  //         Use hashmap to get the idx of the arg
  for (auto &F : *llvmModule) {
    if (argsToRenameIdxMap.find(F.getName()) != argsToRenameIdxMap.end()) {
      auto arg = F.getArg(argsToRenameIdxMap[F.getName()]);
      arg->setName(".arg_address_sanitizer_gm_ptr");
    }
  }

  return success();
}
} // namespace bishengir