//===-------- NormalizePatterns.h - Header for Normalize Patterns -*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// This file declares the populate functions for HIVM Normalize patterns.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_NORMALIZEPATTERNS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_NORMALIZEPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::hivm {

void populateNormalizeArithmeticPatterns(RewritePatternSet &patterns);
void populateNormalizeCmpVnePatterns(RewritePatternSet &patterns);

} // namespace mlir::hivm
#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_NORMALIZEPATTERNS_H