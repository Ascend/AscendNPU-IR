//===- HFusionToHIVM.h - HFusion to HIVM Conversion Patterns ----*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert HFusion dialect to HIVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVM_H
#define BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVM_H

#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
class RewritePatternSet;

void populateReductionPatternsAndLegality(RewritePatternSet &patterns,
                                          ConversionTarget &target,
                                          bool _isRegBasedArch);

void populateMatmulPatternsAndLegality(
    RewritePatternSet &patterns, ConversionTarget &target,
    const ConvertHFusionToHIVMOptions &options, bool _isRegBasedArch);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVM_H
