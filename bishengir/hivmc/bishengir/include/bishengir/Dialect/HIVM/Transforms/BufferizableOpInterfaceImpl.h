//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_BUFFERIZABLEOPINTERFACEIMPL_H
#define BISHENGIR_DIALECT_HIVM_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace hivm {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_BUFFERIZABLEOPINTERFACEIMPL_H
