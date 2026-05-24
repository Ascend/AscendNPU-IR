//===- OpLayoutInterface.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace hivm {
/// Forward declaration.
class DataLayoutAttr;
} // namespace hivm
} // namespace mlir

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H
