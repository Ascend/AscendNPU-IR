//===- TransformOpForSIMT.cpp - Transform Op For SIMT Pass ----------------===//
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
//
// This file implements a pass to transform operations for SIMT execution.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "transform-op-for-simt"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_TRANSFORMOPFORSIMT
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"

using namespace impl;

namespace mlir {
namespace scope {

class TransformOpForSIMTPass
    : public TransformOpForSIMTBase<TransformOpForSIMTPass> {
public:
  explicit TransformOpForSIMTPass() : TransformOpForSIMTBase() {}
  void runOnOperation() final;
};

// Transform operations in SIMT scope for SIMT execution.
// 1. Convert tensor.extract:
//    If tensor.extract's input is a multi-elem (non-scalar) tensor, adopt the following pattern:
//      Before:
//          scope {vf_mode="simt"} {
//            %0 = tensor.extract %1, %2 : tensor<n x i32>, index  // n > 1
//          }
//      After:
//          %buf = memref.alloc(): memref<n x i32>
//          scope {vf_mode="simt"} {
//            hivm.hir.local_store ins(%buf, %1)
//            %subview = memref.subview %buf[%2] [1] [1]
//                : memref<nxi32> -> memref<1xi32>
//            %0 = memref.load %subview[%c0] : memref<1xi32>
//          }
//
//    Else (scalar tensor with shape [1]), extract tensor.extract and its defining ops outside simt_scope:
//      Before:
//          scope {vf_mode="simt"} {
//            %0 = tensor.extract %1, %2 : tensor<1 x i32>, index
//          }
//      After:
//          %0 = tensor.extract %1, %2 : tensor<1 x i32>, index
//          scope {vf_mode="simt"} {
//            ...
//          }

// Helper: check if a tensor is a scalar tensor (shape = [1])
static bool isScalarTensor(RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  return shape.size() == 1 && shape[0] == 1;
}

// Move scalar tensor.extract and its backward slice outside the scope
static void moveScalarExtractOutsideScope(tensor::ExtractOp extractOp,
                                          scope::ScopeOp scopeOp) {
  SetVector<Operation *> toHoist;

  auto isInsideScope = [&](Operation *op) -> bool {
    return scopeOp->isAncestor(op);
  };

  toHoist.insert(extractOp);

  // Backward slice: collect all defining ops of operands inside the scope.
  SmallVector<Value> worklist(extractOp->getOperands().begin(),
                              extractOp->getOperands().end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto *defOp = v.getDefiningOp();
    if (!defOp || !isInsideScope(defOp))
      continue;
    if (toHoist.insert(defOp)) {
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
  }

  // Move ops before the scope, maintaining original block order.
  for (Operation &op : llvm::make_early_inc_range(scopeOp.getRegion().front())) {
    if (toHoist.count(&op))
      op.moveBefore(scopeOp);
  }
}

// Convert multi-elem tensor.extract using local_store + subview + load pattern
static void convertMultiElemExtract(tensor::ExtractOp extractOp,
                                    scope::ScopeOp scopeOp,
                                    llvm::MapVector<Value, Value> &tensorToBuffer,
                                    OpBuilder &builder) {
  Block &scopeBlock = scopeOp.getRegion().front();
  Value tensor = extractOp.getTensor();
  auto tensorType = cast<RankedTensorType>(tensor.getType());

  if (!tensorToBuffer.count(tensor)) {
    builder.setInsertionPoint(scopeOp);
    auto memrefType = MemRefType::get(tensorType.getShape(),
                                      tensorType.getElementType());
    Value buffer = builder.create<memref::AllocOp>(scopeOp.getLoc(),
                                                   memrefType);
    tensorToBuffer[tensor] = buffer;

    if (auto defOp = tensor.getDefiningOp())
      builder.setInsertionPointAfter(defOp);
    else
      builder.setInsertionPointToStart(&scopeBlock);
    builder.create<hivm::LocalStoreOp>(scopeOp.getLoc(), buffer, tensor);
  }

  Value buffer = tensorToBuffer[tensor];
  builder.setInsertionPoint(extractOp);
  Location loc = extractOp.getLoc();

  auto indices = extractOp.getIndices();
  SmallVector<OpFoldResult> offsets(indices.begin(), indices.end());
  SmallVector<OpFoldResult> sizes(indices.size(), builder.getIndexAttr(1));
  SmallVector<OpFoldResult> strides(indices.size(), builder.getIndexAttr(1));
  Value subview = builder.create<memref::SubViewOp>(
      loc, buffer, offsets, sizes, strides);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> loadIndices(indices.size(), c0);
  Value scalar = builder.create<memref::LoadOp>(loc, subview,
                                                ValueRange(loadIndices));

  extractOp.getResult().replaceAllUsesWith(scalar);
  extractOp.erase();
}

// 2. Move tensor.from_elements and related ops outside simt_scope:
//    Before:
//        scope {vf_mode="simt"} {
//          %97 = memref.load %reinterpret_cast_2[%c0]
//          %98 = arith.cmpi slt, %97, %c0_i32
//          %from_elem = tensor.from_elements %98 : tensor<1xi1>
//          %empty = tensor.empty() : tensor<1xf16>
//          %vcast = hivm.hir.vcast ins(%from_elem) outs(%empty) -> tensor<1xf16>
//          hivm.hir.local_store ins(%buf, %vcast)
//          ...
//        }
//    After:
//        %97 = memref.load %reinterpret_cast_2[%c0]
//        %98 = arith.cmpi slt, %97, %c0_i32
//        %from_elem = tensor.from_elements %98 : tensor<1xi1>
//        %empty = tensor.empty() : tensor<1xf16>
//        %vcast = hivm.hir.vcast ins(%from_elem) outs(%empty) -> tensor<1xf16>
//        scope {vf_mode="simt"} {
//          hivm.hir.local_store ins(%buf, %vcast)
//          ...
//        }

static void moveFromElementsOutsideScope(scope::ScopeOp scopeOp) {
  // Find all tensor.from_elements inside the scope.
  SmallVector<tensor::FromElementsOp> fromElementsOps;
  scopeOp.walk([&](tensor::FromElementsOp op) {
    fromElementsOps.push_back(op);
  });

  if (fromElementsOps.empty())
    return;

  SetVector<Operation *> toHoist;

  auto isInsideScope = [&](Operation *op) -> bool {
    return scopeOp->isAncestor(op);
  };

  for (auto fromElem : fromElementsOps) {
    toHoist.insert(fromElem);

    // Backward slice: collect all defining ops of operands inside the scope.
    SmallVector<Value> worklist(fromElem.getOperands().begin(),
                                fromElem.getOperands().end());
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      auto *defOp = v.getDefiningOp();
      if (!defOp || !isInsideScope(defOp))
        continue;
      if (toHoist.insert(defOp)) {
        for (Value operand : defOp->getOperands())
          worklist.push_back(operand);
      }
    }
  }

  // Move ops before the scope, maintaining original block order.
  for (Operation &op : llvm::make_early_inc_range(scopeOp.getRegion().front())) {
    if (toHoist.count(&op))
      op.moveBefore(scopeOp);
  }
}

void TransformOpForSIMTPass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([&](scope::ScopeOp scopeOp) {
    if (!hivm::util::isSIMTVF(scopeOp))
      return;

    // --- Transformation 1: Convert tensor.extract ---
    {
      SmallVector<tensor::ExtractOp> extractOps;
      scopeOp.walk([&](tensor::ExtractOp extractOp) {
        extractOps.push_back(extractOp);
      });

      if (!extractOps.empty()) {
        OpBuilder builder(module.getContext());
        llvm::MapVector<Value, Value> tensorToBuffer;

        for (auto extractOp : extractOps) {
          auto tensorType = cast<RankedTensorType>(extractOp.getTensor().getType());

          if (isScalarTensor(tensorType)) {
            // Scalar tensor (shape=[1]): move extract and backward slice outside scope
            moveScalarExtractOutsideScope(extractOp, scopeOp);
          }
          else {
            // Multi-elem tensor: use local_store + subview + load pattern
            convertMultiElemExtract(extractOp, scopeOp, tensorToBuffer, builder);
          }
        }
      }
    }

    // --- Transformation 2: Move tensor.from_elements outside scope ---
    moveFromElementsOutsideScope(scopeOp);
  });
}

std::unique_ptr<Pass> createTransformOpForSIMTPass() {
  return std::make_unique<TransformOpForSIMTPass>();
}

} // namespace scope
} // namespace mlir
