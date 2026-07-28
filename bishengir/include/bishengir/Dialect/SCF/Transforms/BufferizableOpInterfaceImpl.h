//===- BufferizableOpInterfaceImpl.h - BufferizableOpInterface Impl -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_SCF_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define BISHENGIR_DIALECT_SCF_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace scf_ext {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace scf_ext
} // namespace mlir

#endif // BISHENGIR_DIALECT_SCF_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H