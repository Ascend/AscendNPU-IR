//===- DivOptimization.cpp - Optimize some dpx division and remainders -*-
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

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
#define GEN_PASS_DEF_DPXDIVOPTIMIZATION
#include "bishengir/Dialect/AscendDPX/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct FuncInfo {
  LLVM::LLVMFunctionType funcType;
  StringAttr funcName;
  ArrayAttr funcArgAttr;
};

// copied over from LLVMTritonRemap
static LLVM::CallOp emitOrCreateLibCall(OpBuilder &builder, Location loc,
                                        llvm::StringRef funcName,
                                        ValueRange operands, Type returnType) {
  ModuleOp module;
  if (auto *defOp = operands[0].getDefiningOp()) {
    module = defOp->getParentOfType<mlir::ModuleOp>();
  } else {
    module = operands[0].getParentRegion()->getParentOfType<mlir::ModuleOp>();
  }
  LLVM::LLVMFuncOp func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);

  if (!func) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    SmallVector<Type> types;
    for (auto type : operands.getTypes()) {
      types.push_back(type);
    }
    auto funcType =
        LLVM::LLVMFunctionType::get(returnType, ArrayRef<Type>(types), false);

    func = builder.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);

    auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
    func->setAttr(haccAlwaysInlineAttr, builder.getUnitAttr());
    auto llvmEmitCAttr = LLVM::LLVMDialect::getEmitCWrapperAttrName();
    func->setAttr(llvmEmitCAttr, builder.getUnitAttr());
    func->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                  builder.getStringAttr("private"));
    auto ctx = builder.getContext();
    func->setAttr(mlir::hivm::TFuncCoreTypeAttr::name,
                  hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
  }

  auto calleeAttr = mlir::SymbolRefAttr::get(builder.getContext(), funcName);
  return builder.create<LLVM::CallOp>(loc, returnType, calleeAttr, operands);
}

/*
Under normal circumstances, when the main scalar calls a simt kernel,
every thread computes division operation if there is one, leading to a ton of
overhead. If the divisor of the operation is constant from the
perspective of the main scalar, we can precompute a magic and shift in the main
scalar to speed up division in the simt kernel.
*/
struct DPXDivOptimizationPass
    : public impl::DPXDivOptimizationBase<DPXDivOptimizationPass> {

  bishengir::TritonRemapOptions options;
  DPXDivOptimizationPass(bishengir::TritonRemapOptions opts) : options{opts} {}

  // Note: The \p funcOp may be a simt wrapper or a simt vf.
  Value getUBPtr(LLVM::LLVMFuncOp funcOp) {
    // Search for shared memory attr in the function parameters at first
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (DictionaryAttr dictAttr = funcOp.getArgAttrDict(idx)) {
        if (dictAttr.get(hivm::SharedMemoryAttr::name))
          return arg;
      }
    }

    // Then search for it in the ops
    for (Block &block : funcOp.getBody()) {
      for (Operation &op : block.getOperations()) {
        if (!op.getAttr(hivm::SharedMemoryAttr::name))
          continue;

        auto intToPtr = dyn_cast<LLVM::IntToPtrOp>(op);
        if (!intToPtr) {
          llvm::report_fatal_error("Unexpected op marked with shared mem attr");
        }
        return intToPtr.getResult();
      }
    }

    llvm::report_fatal_error("UB base address is missing");
  }

  // try get operands of division operation, and return whether or not op
  // is a division operation
  bool tryGetDivIn1In2(Operation *op, Value &in1, Value &in2) {
    if (auto dpxOp = dyn_cast<ascend_dpx::DivOp>(op)) {
      in1 = dpxOp.getIn1();
      in2 = dpxOp.getIn2();
      return true;
    }
    if (auto llvmOp = dyn_cast<LLVM::SDivOp>(op)) {
      in1 = llvmOp.getLhs();
      in2 = llvmOp.getRhs();
      return true;
    }
    if (auto llvmOp = dyn_cast<LLVM::UDivOp>(op)) {
      in1 = llvmOp.getLhs();
      in2 = llvmOp.getRhs();
      return true;
    }
    return false;
  }

  // try get operands of remainder operation, and return whether or not op
  // is a remainder operation
  bool tryGetRemIn1In2(Operation *op, Value &in1, Value &in2) {
    if (auto llvmOp = dyn_cast<LLVM::SRemOp>(op)) {
      in1 = llvmOp.getLhs();
      in2 = llvmOp.getRhs();
      return true;
    }
    if (auto llvmOp = dyn_cast<LLVM::URemOp>(op)) {
      in1 = llvmOp.getLhs();
      in2 = llvmOp.getRhs();
      return true;
    }
    return false;
  }

  // check types of operation input, and whether or not in2 is a function
  // argument to funcOp
  bool isOptimizableOperation(LLVM::LLVMFuncOp funcOp, Value in1, Value in2) {
    bool correctType =
        in1.getType().isInteger(32) && in2.getType().isInteger(32);
    if (!correctType)
      return false;
    auto funcArgsList = funcOp.getArguments();
    return std::any_of(funcArgsList.begin(), funcArgsList.end(),
                       [&](const Value &v) { return v == in2; });
  }

  // get position of value within funcOp args
  size_t argPos(LLVM::LLVMFuncOp funcOp, Value v) {
    auto funcArgsList = funcOp.getArguments();
    return std::find(funcArgsList.begin(), funcArgsList.end(), v) -
           funcOp.getArguments().begin();
  }

  void getMagicAndShiftPointers(OpBuilder &builder, Location loc,
                                MLIRContext *context, size_t opt_arg_number,
                                Value &ub_ptr, Value &ub_mul_offset,
                                Value &ub_shf_offset, int ttg_shared) {

    auto ub_ptr_type = LLVM::LLVMPointerType::get(context, 6);

    ub_mul_offset = builder.create<LLVM::ConstantOp>(
        loc, builder.getI32Type(),
        builder.getI32IntegerAttr(ttg_shared + opt_arg_number * 2));
    ub_shf_offset = builder.create<LLVM::ConstantOp>(
        loc, builder.getI32Type(),
        builder.getI32IntegerAttr(ttg_shared + opt_arg_number * 2 + 1));

    ub_mul_offset = builder.create<LLVM::GEPOp>(
        loc, ub_ptr_type, builder.getI32Type(), ub_ptr, ub_mul_offset);
    ub_shf_offset = builder.create<LLVM::GEPOp>(
        loc, ub_ptr_type, builder.getI32Type(), ub_ptr, ub_shf_offset);
  }

  // use builder to make umulhi and shift, and return result
  Value makeUmulhiShift(OpBuilder &builder, Location loc, MLIRContext *context,
                        Value &ub_ptr, Value in1, size_t opt_arg_number,
                        int ttg_shared) {

    // replace operation with load magic and shift from UB, then apply umulhi
    // and shift
    Value ub_mul_offset, ub_shf_offset;
    getMagicAndShiftPointers(builder, loc, context, opt_arg_number, ub_ptr,
                             ub_mul_offset, ub_shf_offset, ttg_shared);

    Value magic = builder.create<ascend_dpx::LoadOp>(
        loc, builder.getI32Type(), ub_mul_offset, Value(), Value());
    Value shift = builder.create<ascend_dpx::LoadOp>(
        loc, builder.getI32Type(), ub_shf_offset, Value(), Value());
    Value result =
        builder.create<LLVM::LShrOp>(loc,
                                     builder.create<ascend_dpx::UmulhiOp>(
                                         loc, builder.getI32Type(), in1, magic),
                                     shift);

    return result;
  }

  // attempt to optimize division operation, return success status
  bool tryOptimizeDivOp(LLVM::LLVMFuncOp funcOp, Operation *op,
                        SmallVector<size_t> &results, OpBuilder &builder,
                        MLIRContext *context, int ttg_shared) {
    Value in1, in2;
    if (!tryGetDivIn1In2(op, in1, in2))
      return false;
    if (!isOptimizableOperation(funcOp, in1, in2))
      return false;

    auto pos = argPos(funcOp, in2);
    size_t idx =
        std::find(results.begin(), results.end(), pos) - results.begin();
    if (idx == results.size()) {
      results.push_back(pos);
    }

    builder.setInsertionPoint(op);
    auto loc = op->getLoc();

    // this is completely reliant on AdaptGPUKernel to pass ub as the _last_ ptr
    Value ub_ptr = getUBPtr(funcOp);

    op->getResult(0).replaceAllUsesWith(
        makeUmulhiShift(builder, loc, context, ub_ptr, in1, idx, ttg_shared));
    op->erase();
    return true;
  }

  bool tryOptimizeRemOp(LLVM::LLVMFuncOp funcOp, Operation *op,
                        SmallVector<size_t> &results, OpBuilder &builder,
                        MLIRContext *context, int ttg_shared) {
    Value in1, in2;
    if (!tryGetRemIn1In2(op, in1, in2))
      return false;
    if (!isOptimizableOperation(funcOp, in1, in2))
      return false;

    auto pos = argPos(funcOp, in2);
    size_t idx =
        std::find(results.begin(), results.end(), pos) - results.begin();
    if (idx == results.size()) {
      results.push_back(pos);
    }

    builder.setInsertionPoint(op);
    auto loc = op->getLoc();

    Value ub_ptr = getUBPtr(funcOp);

    Value divResult =
        makeUmulhiShift(builder, loc, context, ub_ptr, in1, idx, ttg_shared);
    Value tmp = builder.create<LLVM::MulOp>(loc, divResult, in2);
    Value remResult = builder.create<LLVM::SubOp>(loc, in1, tmp);

    op->getResult(0).replaceAllUsesWith(remResult);
    op->erase();
    return true;
  }

  SmallVector<size_t> optimizeDivs(LLVM::LLVMFuncOp funcOp,
                                   MLIRContext *context, int ttg_shared) {
    // if an instance of ascend_dpx.div uses a function argument as a
    // denominator, we can treat the div as a const div and precompute the
    // magic+shift in the faster wrapper function This function will return a
    // vector of arg offsets that it wants the wrapper function to precompute
    // and pass as args.
    OpBuilder builder(context);

    SmallVector<size_t> results;

    funcOp.walk([&](Operation *op) {
      if (tryOptimizeDivOp(funcOp, op, results, builder, context, ttg_shared))
        return;
      if (tryOptimizeRemOp(funcOp, op, results, builder, context, ttg_shared))
        return;
      // op not optimized if both conditions above fail
    });

    return results;
  }

  void populateEntryWithPrecompute(LLVM::LLVMFuncOp funcOp,
                                   MLIRContext *context,
                                   const SmallVector<size_t> &optimizedDivs,
                                   int ttg_shared) {
    OpBuilder builder(context);
    funcOp.walk([&](Operation *op) {
      if (auto launchFunc = dyn_cast<hivm_regbaseintrins::LaunchFuncOp>(op)) {
        builder.setInsertionPoint(launchFunc);
        auto loc = launchFunc.getLoc();
        auto args = launchFunc.getOpnds();
        Value ub_ptr = getUBPtr(funcOp);
        for (size_t i = 0; i < optimizedDivs.size(); i++) {
          Value v = args[optimizedDivs[i]];
          Value shift =
              emitOrCreateLibCall(builder, loc,
                                  "_mlir_ciface_simt_div_magic_shift_uint32_t",
                                  {v}, builder.getI32Type())
                  .getResult();
          Value mul =
              emitOrCreateLibCall(builder, loc,
                                  "_mlir_ciface_simt_div_magic_mul_uint32_t",
                                  {v, shift}, builder.getI32Type())
                  .getResult();

          Value ub_mul_offset, ub_shf_offset;

          getMagicAndShiftPointers(builder, loc, context, i, ub_ptr,
                                   ub_mul_offset, ub_shf_offset, ttg_shared);

          builder.create<ascend_dpx::StoreOp>(loc, ub_mul_offset, mul, Value());
          builder.create<ascend_dpx::StoreOp>(loc, ub_shf_offset, shift,
                                              Value());
        }
      }
    });
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // if the optimization bit is not set, exit immediately
    if (!triton::util::getPassColumnDigit(m, "dpx-div-optimization"))
      return;

    SymbolTable symbolTable(m);
    auto *context = &getContext();
    context->loadDialect<hacc::HACCDialect>();

    int ttg_shared = 0;
    if (m->hasAttr("ttg.shared")) {
      ttg_shared = cast<IntegerAttr>(m->getAttr("ttg.shared")).getInt();
    }

    // map from kernel function name (with _simt_vf cut off)
    // to vector of argument indices
    // The point of this vector is to tell the wrapper function
    // to produce magic and shift values for some of the arguments
    // it passes to the simt_vf
    std::map<std::string, SmallVector<size_t>> optimizeDivMap;

    // maps from simt vf func name to set of arguments to be optimized
    m.walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {

        bool isNVVMKernel = funcOp->hasAttr("nvvm.kernel");
        if (!isNVVMKernel)
          return;

        auto funcName = funcOp.getSymNameAttr();
        if (funcName.size() <= utils::simtVFSuffix.size())
          return;

        auto sizeDiff = funcName.size() - utils::simtVFSuffix.size();
        if (funcName.str().substr(sizeDiff) != utils::simtVFSuffix.str())
          return;

        std::string name = funcOp.getSymNameAttr().str().substr(0, sizeDiff);
        auto optimizedDivs = optimizeDivs(funcOp, context, ttg_shared);

        if (optimizedDivs.size() > 0)
          optimizeDivMap[name] = optimizedDivs;
      }
    });

    // use the generated map and insert magic and shift computations in
    // the wrapper function
    m.walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        auto funcName = funcOp.getSymNameAttr().str();
        const auto optimizeDivs = optimizeDivMap.find(funcName);
        if (optimizeDivs == optimizeDivMap.end())
          return; // no divs to optimize
        populateEntryWithPrecompute(funcOp, context, optimizeDivs->second,
                                    ttg_shared);
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::ascend_dpx::createDPXDivOptimizationPass(
    bishengir::TritonRemapOptions options) {
  return std::make_unique<DPXDivOptimizationPass>(options);
}
