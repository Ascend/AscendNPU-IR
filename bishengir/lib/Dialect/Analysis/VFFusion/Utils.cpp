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
          funcCoreType.value() != hivm::TFuncCoreType::AIV) ||
         funcOp->hasAttr(hivm::TPartOfMixAttr::name);
}
} // namespace mlir::analysis