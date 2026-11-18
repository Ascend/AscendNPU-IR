//===- Utils.cpp - helpers for OptimizeTransposePlacement ----------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/OptimizeTransposePlacement/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir {
namespace hivm {
namespace otp {

bool isRegBaseScope(Operation *op) {
  for (Operation *parent = op; parent; parent = parent->getParentOp()) {
    auto module = dyn_cast<ModuleOp>(parent);
    if (!module)
      continue;
    if (auto target = hacc::utils::getTargetDevice(module))
      return hacc::utils::isRegBasedArch(*target);
  }
  return false;
}

bool isIgnorableUser(Operation *user) {
  return isa<annotation::MarkOp, DebugOp>(user);
}

bool hasSingleRealUse(Value value) {
  Operation *found = nullptr;
  for (Operation *user : value.getUsers()) {
    if (isIgnorableUser(user))
      continue;
    if (found)
      return false;
    found = user;
  }
  return found != nullptr;
}

bool isRank2Static(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  return ranked && ranked.getRank() == 2 && ranked.hasStaticShape();
}

FailureOr<RankedTensorType> swap2D(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  // FUTURE: dynamic dims. Only static 2D shapes are swapped; a dynamic axis
  // needs tensor.empty size operands and a matching extract_slice rewrite.
  if (!ranked || ranked.getRank() != 2 || !ranked.hasStaticShape())
    return failure();
  return RankedTensorType::get({ranked.getDimSize(1), ranked.getDimSize(0)},
                               ranked.getElementType());
}

bool isTransparentChainOp(Operation *op) {
  if (!op || !isLayoutAgnosticOp(op) || op->getNumResults() != 1)
    return false;
  if (auto transpose = op->getAttrOfType<DenseI64ArrayAttr>("transpose"))
    if (!transpose.empty())
      return false;
  if (auto broadcast = op->getAttrOfType<DenseI64ArrayAttr>("broadcast"))
    if (!broadcast.empty())
      return false;
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps)
    return false;
  // A memref temp_buffer is a DPS input (it sits after the inits), so the
  // operand-count check below does not see it. Swapping that memref is not
  // this rewrite, and mapping it to a missing sibling crashes the clone.
  for (Value operand : op->getOperands()) {
    Type ty = operand.getType();
    if (isa<MemRefType>(ty))
      return false;
    if (isa<ShapedType>(ty) && !isa<RankedTensorType>(ty))
      return false;
  }
  if (static_cast<int64_t>(op->getNumOperands()) !=
      dps.getNumDpsInputs() + dps.getNumDpsInits())
    return false;
  return true;
}

bool isSwapTranspose(VTransposeOp transpose) {
  ArrayRef<int64_t> perm = transpose.getPermutation();
  if (perm.size() != 2 || perm[0] != 1 || perm[1] != 0)
    return false;
  if (transpose.getResult().size() != 1)
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(transpose.getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(transpose.getResult()[0].getType());
  if (!srcTy || !dstTy || !isRank2Static(srcTy) || !isRank2Static(dstTy))
    return false;
  // The verifier does not require the result shape to be the permuted source.
  // Rewriting a [1, 0] whose result is not that swap produces illegal IR.
  return srcTy.getDimSize(0) == dstTy.getDimSize(1) &&
         srcTy.getDimSize(1) == dstTy.getDimSize(0);
}

bool isFreelyTransposableSource(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (isa<tensor::EmptyOp>(def))
    return true;
  if (auto vbrc = dyn_cast<VBrcOp>(def))
    return isScalarLike(vbrc.getSrc().getType());
  // FUTURE: dense tensor constants. A splat can be rebuilt with the swapped
  // type; a non-splat needs its elements permuted. Scalar constants never
  // appear here (they are not shaped siblings).
  return false;
}

FailureOr<Value> getOnlyShapedInput(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps)
    return failure();
  Value shaped;
  for (Value input : dps.getDpsInputs()) {
    if (!isa<ShapedType>(input.getType()))
      continue;
    if (shaped)
      return failure();
    shaped = input;
  }
  if (!shaped)
    return failure();
  return shaped;
}

Value createSwappedEmpty(IRRewriter &rewriter, Location loc, Value oldInit,
                         RankedTensorType newType) {
  auto empty = rewriter.create<tensor::EmptyOp>(loc, newType.getShape(),
                                                newType.getElementType());
  if (Operation *def = oldInit.getDefiningOp()) {
    if (Attribute attr = def->getAttr(AddressSpaceAttr::name))
      empty->setAttr(AddressSpaceAttr::name, attr);
    if (Attribute attr = def->getAttr(kInsertedTensorAttr::name))
      empty->setAttr(kInsertedTensorAttr::name, attr);
  }
  return empty.getResult();
}

void overlayBufferAttrs(Value dstEmpty, Value attrSrc) {
  Operation *dst = dstEmpty.getDefiningOp();
  Operation *src = attrSrc.getDefiningOp();
  if (!dst || !src)
    return;
  if (Attribute attr = src->getAttr(AddressSpaceAttr::name))
    dst->setAttr(AddressSpaceAttr::name, attr);
  if (Attribute attr = src->getAttr(kInsertedTensorAttr::name))
    dst->setAttr(kInsertedTensorAttr::name, attr);
}

Value cloneWithSwappedLayout(IRRewriter &rewriter, Operation *op,
                             Value chainSrc, Value newChainSrc,
                             const DenseMap<Value, Value> &siblings) {
  RankedTensorType newResType = *swap2D(op->getResult(0).getType());
  IRMapping mapping;
  mapping.map(chainSrc, newChainSrc);
  auto dps = cast<DestinationStyleOpInterface>(op);
  for (Value input : dps.getDpsInputs()) {
    if (input == chainSrc || !isa<ShapedType>(input.getType()))
      continue;
    auto found = siblings.find(input);
    // applyPlan validates plans before rewriting, so this never fires.
    assert(found != siblings.end() && found->second &&
           "OptimizeTransposePlacement: shaped input has no sibling "
           "replacement");
    mapping.map(input, found->second);
  }
  for (Value init : dps.getDpsInits()) {
    if (mapping.contains(init))
      continue;
    RankedTensorType initType = *swap2D(init.getType());
    mapping.map(init,
                createSwappedEmpty(rewriter, op->getLoc(), init, initType));
  }
  Operation *cloned = rewriter.clone(*op, mapping);
  cloned->getResult(0).setType(newResType);
  return cloned->getResult(0);
}

void eraseIfDead(IRRewriter &rewriter, Operation *op, Value replacement) {
  if (!op)
    return;
  for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
    if (!isIgnorableUser(user))
      continue;
    if (!replacement) {
      rewriter.eraseOp(user);
      continue;
    }
    // The replacement is built at the transpose, which is after a mark that
    // sat on the old chain op. Clone the mark there so it dominates.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(replacement);
    Operation *cloned = rewriter.clone(*user);
    for (Value result : op->getResults())
      cloned->replaceUsesOfWith(result, replacement);
    rewriter.eraseOp(user);
  }
  if (op->use_empty())
    rewriter.eraseOp(op);
}

void sweepDeadOps(Operation *root, const SmallPtrSetImpl<Operation *> &keep) {
  // Consumers come before their producers in reverse program order, so a
  // chain of dead ops collapses in one pass. Erasing a parent after its
  // children were processed never leaves a dangling entry behind.
  SmallVector<Operation *> ops;
  root->walk([&](Operation *op) { ops.push_back(op); });
  for (Operation *op : llvm::reverse(ops)) {
    if (op == root || isIgnorableUser(op) || keep.contains(op))
      continue;
    if (isOpTriviallyDead(op))
      op->erase();
  }
}

} // namespace otp
} // namespace hivm
} // namespace mlir
