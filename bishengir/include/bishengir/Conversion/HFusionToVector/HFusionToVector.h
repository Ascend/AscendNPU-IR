
//===- HFusionToVector.h - HFusion To Vector ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HFusion dialect to the vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HFUSIONTOVECTOR_HFUSIONTOVECTOR_H
#define BISHENGIR_CONVERSION_HFUSIONTOVECTOR_HFUSIONTOVECTOR_H

#include <memory>

namespace mlir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTHFUSIONTOVECTOR
#include "bishengir/Conversion/Passes.h.inc"


/// Creates a pass to convert the HFusion dialect to the AVE dialect.
std::unique_ptr<Pass> createHFusionToVectorConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOVECTOR_HFUSIONTOVECTOR_H
