//===- ValueBoundsOpInterfaceImpl.h - Arith value bounds models -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ARITH_TRANSFORMS_VALUEBOUNDSOPINTERFACEIMPL_H
#define BISHENGIR_DIALECT_ARITH_TRANSFORMS_VALUEBOUNDSOPINTERFACEIMPL_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace arith {

void registerBiShengIRValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry);

} // namespace arith
} // namespace mlir

#endif // BISHENGIR_DIALECT_ARITH_TRANSFORMS_VALUEBOUNDSOPINTERFACEIMPL_H
