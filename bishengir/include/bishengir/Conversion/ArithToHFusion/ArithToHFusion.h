//===- ArithToHFusion.h - Arith to HFusion conversion -----------*- C++ -*-===//
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

#ifndef BISHENGIR_CONVERSION_ARITHTOHFUSION_ARITHTOHFUSION_H
#define BISHENGIR_CONVERSION_ARITHTOHFUSION_ARITHTOHFUSION_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

namespace hfusion {
// Collect a set of patterns to convert Arith dialect ops to Linalg dialect ops.
void populateArithToLinalgConversionPatterns(RewritePatternSet &patterns);
// Collect a set of patterns to convert Arith dialect ops to HFusion dialect
// ops.
void populateArithToHFusionConversionPatterns(RewritePatternSet &patterns);
} // namespace hfusion

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
std::unique_ptr<Pass> createArithToHFusionConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ARITHTOHFUSION_ARITHTOHFUSION_H
