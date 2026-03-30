//===-------- NormalizeTraitsBase.h ----------------------------------------===//
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

#ifndef BISHENGIR_LIB_DIALECT_HFUSION_TRANSFORMS_NORMALIZETRAITSBASE_H
#define BISHENGIR_LIB_DIALECT_HFUSION_TRANSFORMS_NORMALIZETRAITSBASE_H

#include "bishengir/Transforms/Normalize/Utils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::hfusion {

class NormalizeTraitsBase {
public:
  static Value createUnaryOp(PatternRewriter &rewriter, Location loc,
                             Value input, Value dst, UnaryKind kind);
};

} // namespace mlir::hfusion

#endif // BISHENGIR_LIB_DIALECT_HFUSION_TRANSFORMS_NORMALIZETRAITSBASE_H
