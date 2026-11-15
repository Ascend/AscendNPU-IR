//===- HIVMVectorizeBuffers.h - Late buffer data movement --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMVECTORIZEBUFFERS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_HIVMVECTORIZEBUFFERS_H

#include "mlir/Pass/Pass.h"

namespace mlir::hivm {
std::unique_ptr<Pass> createHIVMVectorizeBuffersPass();
void registerHIVMVectorizeBuffersPass();
} // namespace mlir::hivm

#endif
