//===- InitAllDialects.h - BiShengIR Dialects Registration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "bishengir/Dialect/Annotation/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/Arith/Transforms/MeshShardingInterfaceImpl.h"
#include "bishengir/Dialect/Bufferization/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/DecomposeOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/MeshShardingInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/BufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/HIVMTilingInterfaceImpl.h"
#include "bishengir/Dialect/HMAP/IR/HMAP.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Symbol/IR/Symbol.h"
#include "bishengir/Dialect/Tensor/Transforms/MeshShardingInterfaceImpl.h"
#include "bishengir/Dialect/Tensor/Transforms/TilingInterfaceImpl.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Conversion/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#endif

#if BISHENGIR_ENABLE_TRITON_COMPILE
#include "RegisterTritonDialects.h"
#endif

namespace bishengir {

/// Add all the hivm-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::annotation::AnnotationDialect,
                  mlir::ascend_dpx::AscendDPXDialect,
                  mlir::hacc::HACCDialect,
                  mlir::hfusion::HFusionDialect,
                  mlir::hivm::HIVMDialect,
                  mlir::hmap::HMAPDialect,
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

#if BISHENGIR_ENABLE_TRITON_COMPILE
  registerTritonDialects(registry);
  registerConvertTritonAscendGPUToLLVMPass();
  registerConvertProtonAscendGPUToLLVMPass();
  triton::registerGetTritonMetadataPass();
#endif

  // Register all external models.
  mlir::hivm::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::annotation::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::hfusion::registerTilingInterfaceExternalModels(registry);
  mlir::hfusion::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::hfusion::registerShardingInterfaceExternalModels(registry);
  mlir::hfusion::registerDecomposeInterfaceExternalModels(registry);
  mlir::hivm::registerTilingInterfaceExternalModels(registry);
  mlir::arith::registerShardingInterfaceExternalModels(registry);
  bishengir::tensor::registerTilingInterfaceExternalModels(registry);
  bishengir::tensor::registerMeshShardingInterfaceExternalModels(registry);
  bishengir::bufferization::registerTilingInterfaceExternalModels(registry);
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
