//===- LiftScalarProducer.h - Scalar producer lifting ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_LIFTSCALARPRODUCER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_LIFTSCALARPRODUCER_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
namespace func {
class FuncOp;
}

namespace hfusion {

LogicalResult liftScalarProducers(func::FuncOp func);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_LIFTSCALARPRODUCER_H
