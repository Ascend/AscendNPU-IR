//===- MeshShardingInterfaceImpl.h - Impl. of MeshShardingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_MESHSHARDINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_MESHSHARDINGINTERFACEIMPL_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
class DialectRegistry;

namespace hfusion {

void registerShardingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace hfusion
} // namespace mlir
#endif
