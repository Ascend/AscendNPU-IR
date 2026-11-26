//===- MemRefImpl.cpp -----------------------------------------------------===//
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

#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "mlir/Dialect/Utils/ExpandShapeUtils.h"
#endif
#include "mlir/IR/TypeUtilities.h"

namespace {

using namespace mlir;

/// Return the canonical type of the result of a ReinterpretCastOp.
struct ReinterpretCastReturnTypeCanonicalizer {
  MemRefType operator()(memref::ReinterpretCastOp op,
                        ArrayRef<OpFoldResult> mixedOffsets,
                        ArrayRef<OpFoldResult> mixedSizes,
                        ArrayRef<OpFoldResult> mixedStrides) {
    SmallVector<int64_t> staticOffsets;
    SmallVector<int64_t> staticSizes;
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicOffsets;
    SmallVector<Value> dynamicSizes;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedOffsets, dynamicOffsets, staticOffsets);
    dispatchIndexOpFoldResults(mixedSizes, dynamicSizes, staticSizes);
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto sourceMemRefType = cast<BaseMemRefType>(op.getSource().getType());
    auto resType =
        MemRefType::get(staticSizes, sourceMemRefType.getElementType(),
                        StridedLayoutAttr::get(sourceMemRefType.getContext(),
                                               staticOffsets[0], staticStrides),
                        sourceMemRefType.getMemorySpace());
    return resType;
  }
};

/// A canonicalizer wrapper to replace ReinterpretCastOps.
struct ReinterpretCastCanonicalizer {
  void operator()(PatternRewriter &rewriter, memref::ReinterpretCastOp op,
                  memref::ReinterpretCastOp newOp) {
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), newOp);
  }
};

} // namespace

namespace mlir {
namespace memref {

Value createMemRefAllocOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout) {
  auto shapedType = cast<ShapedType>(source.getType());
  ArrayRef<int64_t> staticShapes = shapedType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      Operation *dynDimOp = builder.create<memref::DimOp>(loc, source, i);
      dynamicSizes.push_back(dynDimOp->getResults()[0]);
    }
  }
  MemRefType origType = cast<MemRefType>(shapedType);
  MemRefType newMemTy = MemRefType::get(
      staticShapes, targetElemType,
      layout.has_value() ? layout.value() : origType.getLayout(),
      origType.getMemorySpace());
  return builder.create<memref::AllocOp>(loc, newMemTy, dynamicSizes);
}

Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source) {
  auto elementType = mlir::getElementTypeOrSelf(source);
  auto emptyOp =
      createMemRefAllocOpWithTargetElemType(builder, loc, source, elementType);
  return emptyOp;
}

void getExtendedCanonicalizationPatterns(mlir::RewritePatternSet &results) {
  auto *context = results.getContext();
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  results.add<OpWithOffsetSizesAndStridesConstantArgumentFolder<
      ReinterpretCastOp, ReinterpretCastReturnTypeCanonicalizer,
      ReinterpretCastCanonicalizer>>(context,
                                     "ReinterpretCastConstantArgumentFolder");
  results.add<FoldConstantDimOfOutputShape<ExpandShapeOp, CastOp>>(context);
#endif
}

/// This is a common class used for patterns of the form
/// "someop(memref.memory_space_cast) -> someop".  It folds the source of any
/// memref.cast into the root operation directly.
LogicalResult foldMemRefSpaceCast(Operation *op, Value inner) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto memrefCast = operand.get().getDefiningOp<MemorySpaceCastOp>();
    if (memrefCast && operand.get() != inner &&
        !llvm::isa<UnrankedMemRefType>(memrefCast.getOperand().getType())) {
      operand.set(memrefCast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

} // namespace memref
} // namespace mlir
