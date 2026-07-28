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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
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
#include "llvm/Support/LogicalResult.h"

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

/// Calculate total workspace bytes from AllocWorkspaceOps.
/// Mem-based (A3): offset size may be 1, 2, or 4 (multi-buffer).
/// Reg-based (A5): offset size may be 1 or 2 only.
FailureOr<int64_t>
calculateWorkspaceByte(ArrayRef<Operation *> allocWorkspaceOps,
                       bool isMemBasedArch) {
  constexpr int64_t byteWidth = 8;
  int64_t endLength = 0;
  for (Operation *op : allocWorkspaceOps) {
    auto allocWorkspaceOp =
        dyn_cast_or_null<bishengir::memref_ext::AllocWorkspaceOp>(op);
    if (!allocWorkspaceOp)
      return op->emitOpError("illegal op when calculate workspace size");

    size_t offsetSize = allocWorkspaceOp.getOffset().size();
    bool validOffsetSize =
        !allocWorkspaceOp.getOffset().empty() &&
        (offsetSize <= 2 || (isMemBasedArch && offsetSize == 4));
    assert(validOffsetSize &&
           "Offset could only be either single or double when infer size"
           " (or size 4 on mem-based arch)");
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

/// Reg-based (A5): strip `_mix_aic` / `_mix_aiv` for split mix kernels so AIC
/// and AIV share one host callback name.
static std::string getWorkspaceShapeFuncName(func::FuncOp funcOp) {
  auto funcName = funcOp.getSymName();
  if (funcOp->hasAttr(hivm::TPartOfMixAttr::name)) {
    auto funcCoreType = funcOp
                            ->getAttrOfType<hivm::TFuncCoreTypeAttr>(
                                hivm::TFuncCoreTypeAttr::name)
                            .getFuncCoreType();
    if (funcCoreType == hivm::TFuncCoreType::AIC) {
      auto consumeResult = funcName.consume_back("_mix_aic");
      assert(consumeResult && "Incorrect suffix of AIC kernel");
    } else if (funcCoreType == hivm::TFuncCoreType::AIV) {
      auto consumeResult = funcName.consume_back("_mix_aiv");
      assert(consumeResult && "Incorrect suffix of AIV kernel");
    } else {
      llvm::report_fatal_error("FuncCoreType must be either AIC or AIV for Splitted kernel");
    }
  }
  return hacc::constructHostFunctionName(
      funcName.str(), hacc::HostFuncType::kInferWorkspaceShapeFunction);
}

// ToDo: This function is just enable in triton compilation, and there may be
// conflicts with `HoistTensorEmptyPass`
void insertInferWorkspaceSizeFunc(func::FuncOp funcOp, int64_t workspaceByte,
                                  bool isRegBasedArch) {
  std::string callbackFuncName =
      isRegBasedArch
          ? getWorkspaceShapeFuncName(funcOp)
          : hacc::constructHostFunctionName(
                funcOp.getSymName().str(),
                hacc::HostFuncType::kInferWorkspaceShapeFunction);
  func::FuncOp callbackFunc =
      insertInferWorkspaceSizeFuncImpl(funcOp, workspaceByte, callbackFuncName);

  hacc::utils::setHost(callbackFunc);
  hacc::utils::setHostFuncType(
      callbackFunc, hacc::HostFuncType::kInferWorkspaceShapeFunction);
}

/// Reg-based (A5): accumulate sub-workspace into the shared host callback,
/// shift sub-workspace offsets by the existing workspace size, and rebind
/// AllocWorkspaceOps from sub_workspace to workspace.
static LogicalResult modifyInferWorkspaceSizeFunc(func::FuncOp funcOp,
                                                  BlockArgument subWorkspaceArg,
                                                  int64_t workspaceByte) {
  IRRewriter rewriter(funcOp.getContext());
  auto callbackFuncName =
      rewriter.getStringAttr(getWorkspaceShapeFuncName(funcOp));
  auto callbackFunc = llvm::dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(funcOp, callbackFuncName));
  if (!callbackFunc) {
    callbackFunc =
        insertInferWorkspaceSizeFuncImpl(funcOp, 0, callbackFuncName);
    hacc::utils::setHost(callbackFunc);
    hacc::utils::setHostFuncType(
        callbackFunc, hacc::HostFuncType::kInferWorkspaceShapeFunction);
  }

  auto returnOp =
      cast<func::ReturnOp>(callbackFunc.getBody().front().getTerminator());
  auto origWorkspaceOp =
      returnOp.getOperands()[0].getDefiningOp<arith::ConstantIndexOp>();
  if (!origWorkspaceOp)
    return callbackFunc->emitOpError()
           << "WorkspaceShapeFunction does not have static shape";
  auto origWorkspaceByte = origWorkspaceOp.value();
  rewriter.setInsertionPoint(origWorkspaceOp);
  rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
      origWorkspaceOp, origWorkspaceByte + workspaceByte);
  auto workspaceArg =
      hacc::utils::getBlockArgument(funcOp, hacc::KernelArgType::kWorkspace)
          .value();

  // Add offset to all Sub-Workspace
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  origWorkspaceOp = rewriter.create<arith::ConstantIndexOp>(
      workspaceArg.getLoc(), origWorkspaceByte);
  for (auto *op : subWorkspaceArg.getUsers()) {
    auto allocWorkspaceOp =
        dyn_cast<bishengir::memref_ext::AllocWorkspaceOp>(op);
    if (!allocWorkspaceOp)
      return op->emitOpError()
             << "Workspace argument is not used by AllocWorkspaceOp";
    rewriter.setInsertionPoint(op);
    for (auto &offset : allocWorkspaceOp.getOffsetMutable()) {
      auto offsetVal = offset.get();
      offsetVal = rewriter.createOrFold<arith::AddIOp>(
          offsetVal.getLoc(), origWorkspaceOp, offsetVal);
      rewriter.modifyOpInPlace(op, [&]() { offset.set(offsetVal); });
    }
  }
  rewriter.replaceAllUsesWith(subWorkspaceArg, workspaceArg);
  return success();
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
  ModuleOp moduleOp = getOperation();
  const bool isMemBased = hacc::utils::isMemBasedArch(moduleOp);
  const bool isRegBased = hacc::utils::isRegBasedArch(moduleOp);

  WalkResult walkResult = moduleOp->walk([&](func::FuncOp funcOp) {
    SmallVector<Operation *> allocWorkspaceOps =
        collectAllocWorkspaceOp(funcOp);
    if (allocWorkspaceOps.empty())
      return WalkResult::advance();

    // 1. After plan-workspace, here calculate total workspace size
    auto workspaceByte =
        calculateWorkspaceByte(allocWorkspaceOps, isMemBased);
    if (failed(workspaceByte))
      return WalkResult::interrupt();

    if (isRegBased) {
      // A5 / reg-based: support mix-kernel naming and sub-workspace merge.
      auto subWorkspaceArg = hacc::utils::getBlockArgument(
          funcOp, hacc::KernelArgType::kSubWorkspace);
      if (subWorkspaceArg) {
        if (failed(modifyInferWorkspaceSizeFunc(funcOp, *subWorkspaceArg,
                                                *workspaceByte)))
          return WalkResult::interrupt();
      } else {
        insertInferWorkspaceSizeFunc(funcOp, *workspaceByte,
                                     /*isRegBasedArch=*/true);
      }
    } else {
      // A3 / mem-based: insert a host callback that returns workspace size.
      insertInferWorkspaceSizeFunc(funcOp, *workspaceByte,
                                   /*isRegBasedArch=*/false);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hivm::createInsertInferWorkSpaceSizeFuncPass() {
  return std::make_unique<InsertInferWorkSpaceSizeFuncPass>();
}
