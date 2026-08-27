//===- ArithToAffine.h - Arith to Affine conversion -------------*- C++ -*-===//
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

#ifndef BISHENGIR_CONVERSION_ARITHTOAFFINE_ARITHTOAFFINE_H
#define BISHENGIR_CONVERSION_ARITHTOAFFINE_ARITHTOAFFINE_H

#include <memory>

namespace mlir {
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOAFFINE
#include "bishengir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToAffineConversionPatterns(RewritePatternSet &patterns);
} // namespace arith

/// Creates a pass to convert the Arith dialect to the Affine dialect.
std::unique_ptr<Pass> createArithToAffineConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_ARITHTOAFFINE_ARITHTOAFFINE_H
