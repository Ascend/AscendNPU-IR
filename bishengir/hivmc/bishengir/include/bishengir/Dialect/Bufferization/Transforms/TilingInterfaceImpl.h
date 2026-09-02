//===- TilingInterfaceImpl.h - Implementation of TilingInterface ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Tiling interface for Bufferization Dialect Ops with
// ExternalModel.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_TILINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_TILINGINTERFACEIMPL_H

#include "mlir/IR/Dialect.h"

namespace bishengir {
namespace bufferization {

/// Registers external models for Tiling interface for bufferization ops.
/// Currently, it registers:
///
/// * TilingInterface for `bufferization.to_tensor`.
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace bufferization
} // namespace bishengir

#endif // BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_TILINGINTERFACEIMPL_H
