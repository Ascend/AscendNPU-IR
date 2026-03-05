//===- HIVMDiallctExtension.h - HIVM Dialect Extension -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDIALECTEXTENSION_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDIALECTEXTENSION_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace bishengir {
namespace hivm {
namespace detail {
void registerLibraryFunctionOpInterfaceExtension(
    mlir::DialectRegistry &registry);
} // namespace detail

void registerHIVMDialectExtension(mlir::DialectRegistry &registry);

} // namespace hivm
} // namespace bishengir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDIALECTEXTENSION_H