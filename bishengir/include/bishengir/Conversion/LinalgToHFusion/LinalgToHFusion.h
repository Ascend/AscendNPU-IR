//===- LinalgToHFusion.h - Linalg to HFusion conversion ---------*- C++ -*-===//
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

#ifndef BISHENGIR_CONVERSION_LINALGTOHFUSION_LINALGTOHFUSION_H
#define BISHENGIR_CONVERSION_LINALGTOHFUSION_LINALGTOHFUSION_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTLINALGTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

namespace hfusion {
void populateLinalgToHFusionConversionPatterns_membase(RewritePatternSet &patterns);
void populateLinalgToHFusionConversionPatterns_regbase(RewritePatternSet &patterns);
} // namespace hfusion

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
std::unique_ptr<Pass> createLinalgToHFusionConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_LINALGTOHFUSION_LINALGTOHFUSION_H