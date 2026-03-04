//===- AnnotationOps.cpp - Implementation of Annotation Dialect Ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  build(odsBuilder, odsState, src,
        odsBuilder.getStrArrayAttr(
            llvm::ArrayRef<StringRef>{stringifyEffectMode(EffectMode::Write)}),
        /*values=*/ValueRange{},
        /*keys=*/nullptr);
}

void MarkOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value src,
                   ValueRange values, ArrayAttr keys) {
  build(odsBuilder, odsState, src,
        odsBuilder.getStrArrayAttr(
            llvm::ArrayRef<StringRef>{stringifyEffectMode(EffectMode::Write)}),
        values, keys);
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

    if (markOp.getAttrNum() != 1)
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

void MarkOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto effect : getEffectsAttr()) {
    auto stringEffect = mlir::cast<StringAttr>(effect).getValue();
    if (stringEffect == stringifyEffectMode(EffectMode::Write))
      effects.emplace_back(MemoryEffects::Write::get(), &getSrcMutable(),
                           SideEffects::DefaultResource::get());
    if (stringEffect == stringifyEffectMode(EffectMode::Read))
      effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                           SideEffects::DefaultResource::get());
  }
}

bool MarkOp::isAttrEmpty() { return (*this).getAttrNum() == 0; }

static bool isIgnoredStringAttr(NamedAttribute attr, StringAttr stringAttr) {
  return attr.getName() == stringAttr;
}

template <typename Container>
static Container filterNonIgnoredAttr(const Container &container,
                                      StringAttr stringAttr) {
  auto filteredRange =
      llvm::make_filter_range(container, [&stringAttr](NamedAttribute attr) {
        return !isIgnoredStringAttr(attr, stringAttr);
      });
  return Container(filteredRange.begin(), filteredRange.end());
}

int64_t MarkOp::getAttrNum() {
  // if the annotation only has the default attribute, it can be ignored.
  return filterNonIgnoredAttr(
             DenseSet<NamedAttribute>((*this)->getAttrs().begin(),
                                      (*this)->getAttrs().end()),
             (*this).getEffectsAttrName())
      .size();
}