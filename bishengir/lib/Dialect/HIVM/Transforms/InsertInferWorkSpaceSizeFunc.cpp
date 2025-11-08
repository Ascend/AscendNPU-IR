//===----------------- InsertInferWorkSpaceSizeFunc.cpp -------------------===//
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
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#define DEBUG_TYPE "hivm-insert-infer-workspace-size-func"

namespace mlir {
#define GEN_PASS_DEF_INSERTINFERWORKSPACESIZEFUNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
SmallVector<Operation *> collectAllocWorkspaceOp(func::FuncOp funcOp) {
  SmallVector<Operation *> candidate;
  funcOp.walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
    candidate.push_back(op);
  });

  return candidate;
}

FailureOr<int64_t>
calculateWorkspaceByte(ArrayRef<Operation *> allocWorkspaceOps) {
  constexpr int64_t byteWidth = 8;
  int64_t endLength = 0;
  for (Operation *op : allocWorkspaceOps) {
    auto allocWorkspaceOp =
        dyn_cast_or_null<bishengir::memref_ext::AllocWorkspaceOp>(op);
    if (!allocWorkspaceOp)
      return op->emitOpError("illegal op when calculate workspace size");

    assert(!allocWorkspaceOp.getOffset().empty() &&
           allocWorkspaceOp.getOffset().size() <= 2 &&
           "Offset could only be either single or double when infer size");
    std::optional<SmallVector<int64_t>> offsets = getConstantIntValues(
        SmallVector<OpFoldResult>{allocWorkspaceOp.getOffset()});
    if (!offsets.has_value())
      return op->emitOpError("just support `AllocWorkspaceOp` with "
                             "static offset");

    MemRefType curType = allocWorkspaceOp.getType();
    if (!curType.hasStaticShape())
      return op->emitOpError("just support `AllocWorkspaceOp` with static "
                             "shape result");

    int64_t curLength =
        static_cast<int64_t>(curType.getElementTypeBitWidth() / byteWidth) *
        curType.getNumElements();

    assert((*offsets).back() >= (*offsets).front());
    endLength = std::max(endLength, curLength + (*offsets).back());
  }

  return endLength;
}

func::FuncOp insertInferWorkspaceSizeFuncImpl(func::FuncOp funcOp,
                                              int64_t workspaceByte,
                                              StringRef funcName) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(funcOp);

  FunctionType funcType = FunctionType::get(
      funcOp.getContext(),
      /*input*/ TypeRange{}, /*result*/ TypeRange{builder.getIndexType()});
  auto func =
      builder.create<func::FuncOp>(funcOp.getLoc(),
                                   /*name*/ funcName, /*type*/ funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto byteVal =
      builder.create<arith::ConstantIndexOp>(funcOp.getLoc(), workspaceByte);
  builder.create<func::ReturnOp>(funcOp.getLoc(),
                                 ValueRange{byteVal.getResult()});
  return func;
}

// ToDo: This function is just enable in triton compilation, and there may be
// conflicts with `HoistTensorEmptyPass`
void insertInferWorkspaceSizeFunc(func::FuncOp funcOp, int64_t workspaceByte) {
  std::string callbackFuncName = hacc::constructHostFunctionName(
      funcOp.getSymName().str(),
      hacc::HostFuncType::kInferWorkspaceShapeFunction);
  func::FuncOp callbackFunc =
      insertInferWorkspaceSizeFuncImpl(funcOp, workspaceByte, callbackFuncName);

  hacc::utils::setHost(callbackFunc);
  hacc::utils::setHostFuncType(
      callbackFunc, hacc::HostFuncType::kInferWorkspaceShapeFunction);
}

} // anonymous namespace

class InsertInferWorkSpaceSizeFuncPass
    : public impl::InsertInferWorkSpaceSizeFuncBase<
          InsertInferWorkSpaceSizeFuncPass> {
public:
  using InsertInferWorkSpaceSizeFuncBase<
      InsertInferWorkSpaceSizeFuncPass>::InsertInferWorkSpaceSizeFuncBase;
  void runOnOperation() override;
};

void InsertInferWorkSpaceSizeFuncPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  SmallVector<Operation *> allocWorkspaceOps = collectAllocWorkspaceOp(funcOp);
  if (allocWorkspaceOps.empty())
    return;

  // 1. After plan-workspace, here calculate total workspace size
  auto workspaceByte = calculateWorkspaceByte(allocWorkspaceOps);
  if (failed(workspaceByte))
    signalPassFailure();

  // 2. Insert host callback func to return workspace size
  insertInferWorkspaceSizeFunc(funcOp, *workspaceByte);
}

std::unique_ptr<Pass> mlir::hivm::createInsertInferWorkSpaceSizeFuncPass() {
  return std::make_unique<InsertInferWorkSpaceSizeFuncPass>();
}
