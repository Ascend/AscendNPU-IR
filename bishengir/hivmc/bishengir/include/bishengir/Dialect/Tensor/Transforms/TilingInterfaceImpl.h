//===- TilingInterfaceImpl.h - Implementation of TilingInterface ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_TILINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_TILINGINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
}

namespace bishengir {
namespace tensor {
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);
} // namespace tensor
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_TILINGINTERFACEIMPL_H
