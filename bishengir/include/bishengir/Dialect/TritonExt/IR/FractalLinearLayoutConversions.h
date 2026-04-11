//===- FractalLinearLayoutConversions.h - Fractal layout conv. ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the fractalSharedToLinearLayout function, which converts
// a FractalSharedEncodingAttr to a LinearLayout for shared memory mapping.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITONEXT_IR_FRACTALLINEARLAYOUTCONVERSIONS_H
#define BISHENGIR_DIALECT_TRITONEXT_IR_FRACTALLINEARLAYOUTCONVERSIONS_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::triton {
class LinearLayout;
} // namespace mlir::triton

namespace bishengir::triton_ext {

class FractalSharedEncodingAttr;

/// Compute the linear layout for a fractal shared memory encoding.
/// This maps a flat "offset" (element index in shared memory) to
/// multi-dimensional logical coordinates (dim0, dim1, ...).
mlir::triton::LinearLayout
fractalSharedToLinearLayout(llvm::ArrayRef<int64_t> shape,
                            FractalSharedEncodingAttr fractal);

} // namespace bishengir::triton_ext

#endif // BISHENGIR_DIALECT_TRITONEXT_IR_FRACTALLINEARLAYOUTCONVERSIONS_H
