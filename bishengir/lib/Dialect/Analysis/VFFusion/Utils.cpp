//===- Utils.cpp - Implementation of VFFusion Utils -----------------------===//
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

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "vf-fusion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::analysis {

// Checks whether operation is part of tensor or memref reshaping operations.
bool isReshapeOp(Operation *const op) {
  return isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             memref::CollapseShapeOp, memref::ExpandShapeOp, tensor::ReshapeOp>(
      op);
}

bool isInitialOp(Operation *const op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (utils::isAllocLikeOp(op))
    return true;
  if (reshape_utils::isInitOp(op))
    return true;
  return false;
}

bool isSafeToExcludeOps(Operation *const op) {
  return isInitialOp(op) && op->getNumOperands() == 0;
}

bool isCubeFunc(func::FuncOp funcOp) {
  auto fusionKind = mlir::hfusion::tryGetFusionKind(funcOp);
  if (fusionKind.has_value() &&
      (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
       fusionKind.value() == mlir::hfusion::FusionKind::SingleCube)) {
    return true;
  }
  std::optional<mlir::hivm::TFuncCoreType> funcCoreType =
      mlir::hivm::queryFuncCoreType(funcOp);
  return (funcCoreType.has_value() &&
          funcCoreType.value() != hivm::TFuncCoreType::AIV);
}

bool isVsstbPatternTransposeOp(Operation *op) {
  auto transpose = dyn_cast<linalg::TransposeOp>(op);
  if (!transpose) {
    return false;
  }

  auto inputType = dyn_cast<ShapedType>(transpose.getInput().getType());
  if (!inputType || !inputType.hasStaticShape()) {
    return false;
  }

  auto elemType = inputType.getElementType();
  if (!(elemType.isBF16() || elemType.isF16() || elemType.isF32() ||
        elemType.isFloat8E4M3FN() || elemType.isFloat8E5M2())) {
    return false;
  }

  auto perm = transpose.getPermutation();
  int64_t rank = static_cast<int64_t>(perm.size());
  // Rule 0: Should be 3-dim transpose
  if (rank != 3)
    return false;

  // Rule 1: Should not be inner axis transpose
  if (perm[rank - 1] != rank - 1)
    return false;

  // Rule 2: Last axis should fit in exactly 32 bytes
  ArrayRef<int64_t> shape = inputType.getShape();
  // Calculate element width in bytes
  uint64_t elemByteWidth =
      llvm::divideCeil(inputType.getElementType().getIntOrFloatBitWidth(),
                       utils::INTR_BITS_PER_BYTE);
  int64_t lastDim = shape[rank - 1];
  return lastDim * static_cast<int64_t>(elemByteWidth) == 32;
}

static bool hasSyncBlockOpBetween(Operation *op, Operation *user) {
  if (op->getBlock() != user->getBlock())
    return true;

  // SyncBlockWaitOp/SyncBlockSetOp are ordering boundaries inserted before
  // binary conversion. Check the concrete producer-user edge instead of a
  // use-list boundary, since users may appear in a different block order.
  for (auto *curOp = op->getNextNode(); curOp; curOp = curOp->getNextNode()) {
    if (curOp == user)
      return false;
    if (isa<hivm::SyncBlockWaitOp, hivm::SyncBlockSetOp>(curOp))
      return true;
  }
  return true;
}

bool userCanFuseIntoVsstbPatternTransposeOp(Operation *op) {
  if (op->getUsers().empty())
    return false;

  for (Operation *user : op->getUsers()) {
    if (isVsstbPatternTransposeOp(user) && !hasSyncBlockOpBetween(op, user))
      return true;
  }

  for (Operation *user : op->getUsers()) {
    if (!isa<linalg::GenericOp>(user))
      continue;
    if (!hasSyncBlockOpBetween(op, user) &&
        userCanFuseIntoVsstbPatternTransposeOp(user))
      return true;
  }
  return false;
}

bool isExpandShapeOpCanFuseIntoVsstbPatternTranspose(Operation *op) {
  auto expandShape = dyn_cast<tensor::ExpandShapeOp>(op);
  if (!expandShape) {
    return false;
  }
  // FIXME: expand_shape with one-dim src will cause error when tile after
  // fusing into vsstb pattern transpose, see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/1100
  auto srcType = dyn_cast<TensorType>(expandShape.getSrc().getType());
  auto resType = dyn_cast<TensorType>(expandShape.getResult().getType());
  if (srcType.getShape().size() != 2 || resType.getShape().size() != 3) {
    return false;
  }

  return userCanFuseIntoVsstbPatternTransposeOp(op);
}


} // namespace mlir::analysis
