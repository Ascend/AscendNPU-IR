//===- FuncBufferizableOpInterfaceImpl.h - BufferizableOpInterface Impl. --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BUFFERIZATION_FUNCBUFFERIZABLEOPINTERFACEIMPL_H
#define BISHENGIR_DIALECT_BUFFERIZATION_FUNCBUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace bishengir {
namespace bufferization {
void registerFuncBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
} // namespace bufferization
} // namespace bishengir

#endif // BISHENGIR_DIALECT_BUFFERIZATION_FUNCBUFFERIZABLEOPINTERFACEIMPL_H
