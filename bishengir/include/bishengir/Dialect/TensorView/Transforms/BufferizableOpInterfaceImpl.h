//===- BufferizableOpInterfaceImpl.h ----------------------------*- C++ -*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BISHENGIR_DIALECT_TENSORVIEW_BUFFERIZABLEOPINTERFACEIMPL_H
#define BISHENGIR_DIALECT_TENSORVIEW_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
namespace tv {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace tv
} // namespace mlir

#endif
