//===- CustomOpCanonicalizations.cpp - Custom op canonicalizers -----------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
template <typename CustomOpT>
struct CustomOpCanonicalizer : public OpRewritePattern<CustomOpT> {
  using OpRewritePattern<CustomOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomOpT customOp,
                                PatternRewriter &rewriter) const final {
    if (!customOp.isBuiltin())
      return failure();

    const auto &builtinInfo = CustomOpT::kBuiltins.at(customOp.getName());
    const auto &coreType = customOp.getCoreType();
    if (!coreType || *coreType != builtinInfo.coreType) {
      customOp.setCoreType(builtinInfo.coreType);
      return success();
    }

    if constexpr (std::is_same_v<CustomOpT, CustomOp>) {
      if (customOp.getPipe() != builtinInfo.pipe) {
        customOp.setPipe(builtinInfo.pipe);
        return success();
      }
    } else if constexpr (std::is_same_v<CustomOpT, CustomMacroOp>) {
      if (customOp.getInPipe() != builtinInfo.inPipe) {
        customOp.setInPipe(builtinInfo.inPipe);
        return success();
      }
      if (customOp.getOutPipe() != builtinInfo.outPipe) {
        customOp.setOutPipe(builtinInfo.outPipe);
        return success();
      }
    }

    const auto &vfMode = customOp.getVFMode();
    if (!vfMode || *vfMode != builtinInfo.vfMode) {
      customOp.setVFMode(builtinInfo.vfMode);
      return success();
    }

    return failure();
  }
};
} // namespace

void CustomOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                           ::mlir::MLIRContext *context) {
  results.add<CustomOpCanonicalizer<CustomOp>>(context);
}

void CustomMacroOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add<CustomOpCanonicalizer<CustomMacroOp>>(context);
}
