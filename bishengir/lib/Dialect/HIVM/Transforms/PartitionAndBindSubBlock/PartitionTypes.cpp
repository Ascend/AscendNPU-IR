//-----------------------------PartitionTypes.cpp-----------------------------//
//
// Shared cube/shared-op classifier and the two {sub_block} scope-discovery predicates.
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

#include "bishengir/Dialect/HIVM/Transforms/PartitionAndBindSubBlock/PartitionTypes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hivm {
namespace partition_and_bind {

bool isCubeOrSharedOp(Operation *op) {
  if (!op)
    return false;

  if (isa<hivm::MatmulOp, hivm::MmadL1Op, hivm::BatchMmadL1Op, hivm::FixpipeOp,
          hivm::MixMatmulOp>(op))
    return true;

  // Anything explicitly tagged to run on the cube.
  if (FailureOr<TCoreType> ct = hivm::getCoreType(op);
      succeeded(ct) && *ct == TCoreType::CUBE)
    return true;
  return false;
}

Core getSubBlockCoreOf(Operation *op) {
  auto scopeOp = dyn_cast_or_null<scope::ScopeOp>(op);
  if (!scopeOp)
    return Core::Bottom;
  auto attr = scopeOp->getAttrOfType<IntegerAttr>(kSubBlockAttrName);
  if (!attr)
    return Core::Bottom;
  return coreFromIndex(attr.getInt());
}


static bool tracesToSubBlockIdx(Value v) {
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    v = cast.getIn();
  return v.getDefiningOp<hivm::GetSubBlockIdxOp>() != nullptr;
}

static bool definesConstant(Value v) {
  return v.getDefiningOp<arith::ConstantOp>() != nullptr;
}

bool isOperandParallelSubBlockGuard(Operation *op) {
  auto ifOp = dyn_cast_or_null<scf::IfOp>(op);
  if (!ifOp)
    return false;
  auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
    return false;
  return (tracesToSubBlockIdx(cmp.getLhs()) && definesConstant(cmp.getRhs())) ||
         (tracesToSubBlockIdx(cmp.getRhs()) && definesConstant(cmp.getLhs()));
}

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir
