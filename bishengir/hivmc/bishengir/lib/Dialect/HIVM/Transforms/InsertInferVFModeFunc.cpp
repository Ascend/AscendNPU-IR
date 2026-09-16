//===-------------------- InsertInferVFModeFunc.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "hivm-insert-infer-vf-mode-func"

namespace mlir {
#define GEN_PASS_DEF_INSERTINFERVFMODEFUNC
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

func::FuncOp insertInferVFModeFuncImpl(func::FuncOp funcOp, const VFMode vfMode,
                                       StringRef funcName) {
  const auto &ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(funcOp);

  FunctionType funcType = FunctionType::get(
      ctx,
      /*input*/ TypeRange{}, /*result*/ TypeRange{builder.getIndexType()});

  const auto &loc = funcOp.getLoc();
  auto func =
      builder.create<func::FuncOp>(loc,
                                   /*name*/ funcName, /*type*/ funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto byteVal =
      builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(vfMode));
  builder.create<func::ReturnOp>(loc, ValueRange{byteVal.getResult()});
  return func;
}

void insertInferVFModeFunc(func::FuncOp funcOp, const VFMode vfMode) {
  std::string callbackFuncName = hacc::constructHostFunctionName(
      funcOp.getSymName().str(), hacc::HostFuncType::kInferVFModeFunction);
  func::FuncOp callbackFunc =
      insertInferVFModeFuncImpl(funcOp, vfMode, callbackFuncName);

  hacc::utils::setHost(callbackFunc);
  hacc::utils::setHostFuncType(callbackFunc,
                               hacc::HostFuncType::kInferVFModeFunction);
}

} // anonymous namespace

class InsertInferVFModeFuncPass
    : public impl::InsertInferVFModeFuncBase<InsertInferVFModeFuncPass> {
public:
  using InsertInferVFModeFuncBase<
      InsertInferVFModeFuncPass>::InsertInferVFModeFuncBase;
  void runOnOperation() override;
};

void InsertInferVFModeFuncPass::runOnOperation() {
  Operation *op = getOperation();
  if (!hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()))
    return;

  if (!hacc::utils::isDeviceEntry(op))
    return;

  // 1. Query VF mode attribute, if not exist, try to infer.
  if (!op->hasAttr(VFModeAttr::name)) {
    PassManager pm(op->getContext());
    pm.addPass(createInferVFModePass());
    if (failed(pm.run(op)))
      return signalPassFailure();
  }

  if (const auto vfModeAttr = op->getAttrOfType<VFModeAttr>(VFModeAttr::name)) {
    // 2. Insert host callback func to return VF mode
    insertInferVFModeFunc(cast<func::FuncOp>(op), vfModeAttr.getValue());
  } else {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertInferVFModeFuncPass() {
  return std::make_unique<InsertInferVFModeFuncPass>();
}
