//===- MathTemplateHelpers.h ------------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_MATHTEMPLATEHELPERS_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_MATHTEMPLATEHELPERS_H

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

namespace mlir {

/// Replaces each NaN element in `src` with `paddingInfValue` so the
/// subsequent maxf/minf op treats NaN as a guaranteed loser.
template <typename Traits>
Value maskNanWithInf(PatternRewriter &rewriter, Location loc, Value src,
                     double paddingInfValue) {
  Type elementType = getElementTypeOrSelf(src.getType());
  Value constInfinity =
      utils::createConstantOp<double>(rewriter, loc, elementType,
                                      paddingInfValue);
  Value isNanResult = Traits::createIsNanOp(rewriter, loc, src);
  Value selectDst = utils::createEmptyOp(rewriter, loc, src);
  return Traits::createSelectOp(rewriter, loc, isNanResult, constInfinity, src,
                                selectDst);
}

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_UTILS_MATHTEMPLATEHELPERS_H
