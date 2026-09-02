//===- MeshShardingInterfaceImpl.h - Impl. of MeshShardingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_TENSOR_MESHSHARDINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_TENSOR_MESHSHARDINGINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace bishengir {
namespace tensor {
void registerMeshShardingInterfaceExternalModels(
    mlir::DialectRegistry &registry);
} // namespace tensor
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TENSOR_MESHSHARDINGINTERFACEIMPL_H
