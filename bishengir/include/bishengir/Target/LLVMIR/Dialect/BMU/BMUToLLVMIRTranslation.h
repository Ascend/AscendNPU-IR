//===- BMUToLLVMIRTranslation.h - BMU to LLVM IR --------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_BMU_BMUTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_BMU_BMUTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the BMU dialect and the translation from it to LLVM IR.
void registerBMUDialectTranslation(DialectRegistry &registry);

/// Register the BMU dialect and the translation from it in the registry
/// associated with the given context.
void registerBMUDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_BMU_BMUTOLLVMIRTRANSLATION_H
