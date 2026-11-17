//===- BMUToLLVMIRTranslation.cpp - Translate BMU to LLVM IR --------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

#include "bishengir/Target/LLVMIR/Dialect/BMU/BMUToLLVMIRTranslation.h"
#include "bishengir/Dialect/BMU/IR/BMU.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IntrinsicsHIVM.h"

using namespace mlir;
using namespace mlir::bmu;

namespace {

/// Implementation of the dialect interface that converts BMU operations
/// to LLVM IR. The bmu.intr.* wrappers become llvm.hivm intrinsics that only
/// exist in A6 builds (BSPUB_DAVINCI_BISHENGIR_A6).
class BMUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#if !defined(__LLVM_MAJOR_VERSION_20_COMPATIBLE__) &&                          \
    !defined(__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
#include "bishengir/Dialect/BMU/IR/BMUConversions.inc"
#endif

    return failure();
  }
};

} // namespace

void mlir::registerBMUDialectTranslation(DialectRegistry &registry) {
  registry.insert<BMUDialect>();
  registry.addExtension(+[](MLIRContext *ctx, BMUDialect *dialect) {
    dialect->addInterfaces<BMUDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerBMUDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerBMUDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
