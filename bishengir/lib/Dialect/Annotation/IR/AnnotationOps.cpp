//===- AnnotationOps.cpp - Implementation of Annotation Dialect Ops -------===//
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::annotation;

namespace {
static constexpr llvm::StringLiteral kBufferSizeInByteAttr =
    "buffer_size_in_byte";
} // namespace

//===----------------------------------------------------------------------===//
// MarkOp
//===----------------------------------------------------------------------===//

void MarkOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value src) {
  build(odsBuilder, odsState, src, /*values=*/ValueRange{}, /*keys=*/nullptr);
}

/// Fold buffer size annotation to mark the root alloc.
LogicalResult foldBufferSizeAnnotationToAlloc(MarkOp markOp) {
  if (!markOp.isAnnotatedByStaticAttr(kBufferSizeInByteAttr))
    return failure();

  // find the root alloc and move upwards
  auto markedVal = markOp.getSrc();
  if (utils::isAllocLikeOp(markedVal))
    return failure();

  auto maybeAllocOp = utils::tracebackMemRefToAlloc(markedVal);
  if (!maybeAllocOp.has_value())
    return failure();

  markOp.getSrcMutable().assign((maybeAllocOp.value()).getMemref());
  return success();
}

LogicalResult MarkOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return foldBufferSizeAnnotationToAlloc(*this);
}

struct FoldUselessBufferSizeMarkOp : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                PatternRewriter &rewriter) const override {
    if (!markOp.isAnnotatedByStaticAttr(kBufferSizeInByteAttr))
      return failure();

    if (!llvm::hasSingleElement(markOp->getAttrs()))
      return failure();

    auto srcVal = markOp.getSrc();
    // If the alloc is a static one, we can ignore the buffer size.
    if (isa<MemRefType>(srcVal.getType())) {
      auto maybeAlloc = utils::tracebackMemRefToAlloc(srcVal);
      if (maybeAlloc.has_value() && (*maybeAlloc).getType().hasStaticShape()) {
        rewriter.eraseOp(markOp);
        return success();
      }
    }

    auto users = srcVal.getUses();
    if (!llvm::hasSingleElement(users))
      return failure();

    // if the value marked by annotation only have one user...

    // and that the source op is a tensor/memref cast,
    // propagate the annotation mark to its source
    auto *srcDefiningOp = srcVal.getDefiningOp();
    if (isa_and_present<tensor::CastOp, memref::CastOp, tensor::CollapseShapeOp,
                        tensor::ExpandShapeOp, memref::CollapseShapeOp,
                        memref::ExpandShapeOp>(srcDefiningOp)) {
      rewriter.modifyOpInPlace(markOp, [&]() {
        markOp.setOperand(0, srcDefiningOp->getOperand(0));
      });
      return success();
    }

    // otherwise, directly remote it
    rewriter.eraseOp(markOp);
    return success();
  }
};

void MarkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldUselessBufferSizeMarkOp>(context);
}

bool MarkOp::isAnnotatedBy(StringRef key) {
  return isAnnotatedByStaticAttr(key) || isAnnotatedByDynamicAttr(key);
}

bool MarkOp::isAnnotatedByStaticAttr(StringRef key) {
  return (*this)->hasAttr(key);
}

bool MarkOp::isAnnotatedByDynamicAttr(StringRef key) {
  if (!getKeys())
    return false;

  return llvm::any_of(getKeysAttr().getValue(), [&](Attribute attr) {
    return cast<StringAttr>(attr).getValue() == key;
  });
}

OpFoldResult MarkOp::getMixedAttrValue(StringRef key) {
  if (isAnnotatedByStaticAttr(key))
    return OpFoldResult{getStaticAttrValue(key)};

  return OpFoldResult{getDynamicAttrValue(key)};
}

Attribute MarkOp::getStaticAttrValue(StringRef key) {
  return (*this)->getAttr(key);
}

Value MarkOp::getDynamicAttrValue(StringRef key) {
  for (auto [storedKey, value] :
       llvm::zip_equal(getKeysAttr().getValue(), getValues())) {
    if (cast<StringAttr>(storedKey).getValue() == key)
      return value;
  }
  return Value();
}
