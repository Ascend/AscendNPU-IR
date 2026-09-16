//===- TilingInterfaceImpl.h - Implementation of TilingInterface ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_TILINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_TILINGINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace hfusion {
void registerTilingInterfaceExternalModels(DialectRegistry &registry);
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_TILINGINTERFACEIMPL_H
