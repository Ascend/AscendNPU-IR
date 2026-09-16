//===- AddFFTSAddr.cpp ---- Add FFTS Base Addr Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_ADDFFTSADDR
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

/// This pass add ffts base address to func param and annotation.
struct AddFFTSAddrPass : public impl::AddFFTSAddrBase<AddFFTSAddrPass> {
  explicit AddFFTSAddrPass(const AddFFTSAddrOptions &options)
      : AddFFTSAddrBase(options) {}

public:
  void runOnOperation() override;

  void updateUsedFuncCalls(func::FuncOp funcOp, size_t idx) {
    auto usesPtr = funcOp.getSymbolUses(getOperation());
    if (usesPtr == std::nullopt) {
      return;
    }
    SymbolTable::UseRange maybeUses = *usesPtr;
    for (SymbolTable::SymbolUse use : maybeUses) {
      func::CallOp call = cast<func::CallOp>(use.getUser());
      auto callSite = call->getParentOfType<func::FuncOp>();
      addFFTSBaseAddressToFuncParam(callSite, idx);
      OpBuilder builder(call);
      SmallVector<Value> newOperands;
      for (size_t i = 0; i < call->getNumOperands(); i++) {
        if (i == idx) {
          newOperands.push_back(callSite.getArgument(i));
        }
        newOperands.push_back(call->getOperand(i));
      }
      func::CallOp newCall = builder.create<func::CallOp>(
          call.getLoc(), call.getCalleeAttr(), call.getResultTypes(),
          ValueRange(newOperands));
      call.replaceAllUsesWith(newCall);
      call->erase();
    }
  }

  void addFFTSBaseAddressToFuncParam(func::FuncOp funcOp, size_t startIdx) {
    if (hacc::utils::isKernelArg(funcOp, startIdx,
                                 hacc::KernelArgType::kFFTSBaseAddr))
      return;

    OpBuilder opBuilder(funcOp.getContext());
    // add argument && annotate
    auto loc = funcOp.getArgument(startIdx).getLoc();
    IntegerType argumentType = opBuilder.getIntegerType(64);
    NamedAttribute fftsBaseAddressAttr = hacc::createHACCKernelArgAttr(
        opBuilder.getContext(), hacc::KernelArgType::kFFTSBaseAddr);
    DictionaryAttr dictAttrs = opBuilder.getDictionaryAttr(
        SmallVector<NamedAttribute>{fftsBaseAddressAttr});
    funcOp.insertArgument(startIdx, argumentType, dictAttrs, loc);
    updateUsedFuncCalls(funcOp, startIdx);
  }

  /// check if it can insert to the pos of function arguments
  /// >=0 && =<len
  bool isInsertPosValid(func::FuncOp funcOp, int insertPos) {
    return insertPos >= 0 &&
           static_cast<unsigned>(insertPos) < funcOp.getNumArguments();
  }

  void handleFunc(func::FuncOp funcOp) {
    int insertPos = -1;
    if (this->forceAddFFTSAddr != -1) {
      insertPos = this->forceAddFFTSAddr;
    } else {
      auto fusionKindAttr =
          funcOp->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
      if (fusionKindAttr &&
          (fusionKindAttr.getFusionKind() == FusionKind::ShallowCV ||
           fusionKindAttr.getFusionKind() == FusionKind::MixCV ||
           fusionKindAttr.getFusionKind() == FusionKind::MixC2)) {
        insertPos = 0;
      }
    }
    if (isInsertPosValid(funcOp, insertPos))
      addFFTSBaseAddressToFuncParam(funcOp, 0);
  }
};

void AddFFTSAddrPass::runOnOperation() {
  auto module = getOperation();
  module.walk([&](func::FuncOp funcOp) {
    if (!hacc::utils::isDeviceEntry(funcOp))
      return;

    handleFunc(funcOp);
  });
}

std::unique_ptr<Pass>
mlir::hfusion::createAddFFTSAddrPass(const AddFFTSAddrOptions &options) {
  return std::make_unique<AddFFTSAddrPass>(options);
}
