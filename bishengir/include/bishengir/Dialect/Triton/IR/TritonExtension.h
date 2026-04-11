//===- TritonExtension.h - BishengIR Triton dialect extension ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers bishengir-specific ops (e.g. tt.call_scalar) into the upstream
// Triton dialect via a DialectRegistry extension.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITON_IR_TRITONEXTENSION_H
#define BISHENGIR_DIALECT_TRITON_IR_TRITONEXTENSION_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/Triton/IR/TritonOps.h.inc"

namespace bishengir {

/// Register the bishengir Triton dialect extension into \p registry.
/// When the Triton dialect is loaded, this adds tt.call_scalar (and any other
/// bishengir-specific ops) to the dialect.
void registerTritonDialectExtension(mlir::DialectRegistry &registry);

} // namespace bishengir

#endif // BISHENGIR_DIALECT_TRITON_IR_TRITONEXTENSION_H
