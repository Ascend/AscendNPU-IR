//===- InitAllDialects.h - BiShengIR Dialects Registration ------*- C++ -*-===//
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
//
// This file defines a helper to trigger the registration of all bishengir
// related dialects to the system.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_INITALLDIALECTS_H
#define BISHENGIR_INITALLDIALECTS_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "bishengir/Dialect/Annotation/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/Bufferization/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/DecomposeOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/HIVMTilingInterfaceImpl.h"
#include "bishengir/Dialect/Tensor/Transforms/TilingInterfaceImpl.h"
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#endif

namespace bishengir {

/// Add all the hivm-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::annotation::AnnotationDialect,
                  mlir::hacc::HACCDialect,
                  mlir::hfusion::HFusionDialect,
                  mlir::hivm::HIVMDialect,
                  mlir::mathExt::MathExtDialect,
                  mlir::scope::ScopeDialect,
                  mlir::symbol::SymbolDialect,
                  bishengir::memref_ext::MemRefExtDialect>();
  // clang-format on

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
  // clang-format off
  registry.insert<mlir::torch::Torch::TorchDialect,
                  mlir::torch::TorchConversion::TorchConversionDialect,
                  mlir::torch::TMTensor::TMTensorDialect>();
  // clang-format on
#endif

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  // Register all external models.
  mlir::hivm::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::annotation::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::hfusion::registerTilingInterfaceExternalModels(registry);
  mlir::hfusion::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::hfusion::registerDecomposeInterfaceExternalModels(registry);
  mlir::hivm::registerTilingInterfaceExternalModels(registry);
  bishengir::tensor::registerTilingInterfaceExternalModels(registry);
  bishengir::bufferization::registerTilingInterfaceExternalModels(registry);
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
}

/// Append all the bishengir-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  bishengir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace bishengir

#endif // BISHENGIR_INITALLDIALECTS_H
