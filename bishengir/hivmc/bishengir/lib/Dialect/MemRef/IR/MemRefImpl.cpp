//===- MemRefImpl.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace memref {

Value createMemRefAllocOpWithTargetElemType(OpBuilder &builder, Location loc,
                                            Value source, Type targetElemType) {
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
  MemRefType newMemTy = MemRefType::get(staticShapes, targetElemType, nullptr,
                                        origType.getMemorySpace());

  return builder.create<memref::AllocOp>(loc, newMemTy, dynamicSizes);
}

Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source) {
  auto elementType = mlir::getElementTypeOrSelf(source);
  auto emptyOp =
      createMemRefAllocOpWithTargetElemType(builder, loc, source, elementType);
  return emptyOp;
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
