//===- HoistSimtScalarCallsToSimd.cpp -------------------------------------===//
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
//
// HoistCallScalarToCaller hoists each SIMT VF's scalar calls into the
// AdaptGPUKernel wrapper, stamping the shared-memory offset onto every store it
// emits there. This pass reads those ops back, rebuilds them in the SIMD caller
// in func/memref form, erases the wrapper's copy, and declares the VF's effect
// on the shared buffer so the sync solver orders the two.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMToTritonGPU/MemRefDescriptor.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/SimtScalarCalls.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTSIMTSCALARCALLSTOSIMD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

/// The scalar library is AIV-only and must be inlined at the call site.
static void setLibDeclAttrs(OpBuilder &b, func::FuncOp funcOp) {
  funcOp->setAttr(hacc::stringifyHACCToLLVMIRTranslateAttr(
                      hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE),
                  b.getUnitAttr());
  funcOp->setAttr(TFuncCoreTypeAttr::name,
                  TFuncCoreTypeAttr::get(b.getContext(), TFuncCoreType::AIV));
}

/// Create, or reuse, a private declaration of a scalar template-library
/// function
static func::FuncOp getOrCreateLibDecl(OpBuilder &b, ModuleOp mod,
                                       StringRef name, TypeRange argTys,
                                       TypeRange resTys) {
  if (auto existing = mod.lookupSymbol<func::FuncOp>(name)) {
    // A declaration created elsewhere carries neither attribute.
    setLibDeclAttrs(b, existing);
    return existing;
  }
  OpBuilder::InsertionGuard guard(b);
  // Append, so several declarations keep creation order.
  b.setInsertionPointToEnd(mod.getBody());
  auto funcOp = b.create<func::FuncOp>(mod.getLoc(), name,
                                       b.getFunctionType(argTys, resTys));
  funcOp.setPrivate();
  setLibDeclAttrs(b, funcOp);
  return funcOp;
}

/// Look through value-preserving casts, so an operand renamed by one of the
/// dialect conversions running before the hoist still resolves.
static Value peelCasts(Value v) {
  while (v) {
    auto cast = v.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast)
      break;
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      break;
    if (cast.getOperand(0).getType() != cast.getResult(0).getType())
      break;
    v = cast.getOperand(0);
  }
  return v;
}

/// Replay FuncToTriton's ABI expansion to map wrapper slots to original args.
/// Later wrapper arguments are appended and have no entry in this mapping.
static SmallVector<unsigned> getOriginalArgIndices(func::FuncOp decl) {
  TypeRange argTys = decl.getArgumentTypes();
  bool barePtr = llvm::all_of(argTys, [](Type type) {
    auto memTy = dyn_cast<BaseMemRefType>(type);
    return !memTy || LLVMTypeConverter::canConvertToBarePtr(memTy);
  });

  SmallVector<unsigned> indices;
  for (auto [idx, type] : llvm::enumerate(argTys)) {
    unsigned width = 1;
    if (!barePtr) {
      if (isa<UnrankedMemRefType>(type))
        break;
      if (auto memTy = dyn_cast<MemRefType>(type))
        width = getDescriptorSize(memTy.getRank());
    }
    indices.append(width, idx);
  }
  return indices;
}

/// Resolve one operand of a hoisted llvm.call to a caller-side Value.
static FailureOr<Value> resolveScalarCallArg(
    Value rawArg, LLVM::StoreOp store, LLVM::LLVMFuncOp wrapper,
    func::CallOp callOp, func::FuncOp calleeDecl, ArrayRef<unsigned> argIndices,
    const DenseMap<Value, Value> &recovered, OpBuilder &b, Location loc) {
  Value arg = peelCasts(rawArg);

  // Reuse the caller-side result of an earlier scalar helper.
  auto recoveredIt = recovered.find(arg);
  if (recoveredIt != recovered.end())
    return recoveredIt->second;

  if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
    if (blockArg.getOwner()->getParentOp() != wrapper.getOperation())
      return store.emitError()
             << "hoisted scalar call takes an argument of a block other than "
                "the wrapper's entry block";
    unsigned pos = blockArg.getArgNumber();
    if (pos >= argIndices.size())
      return callOp.emitError() << "scalar call operand at wrapper position "
                                << pos << " does not map to any of the call's "
                                << callOp.getNumOperands() << " operands";
    unsigned idx = argIndices[pos];
    Type declTy = calleeDecl.getArgumentTypes()[idx];
    if (isa<BaseMemRefType>(declTy))
      return callOp.emitError() << "scalar call operand at wrapper position "
                                << pos << " maps to argument " << idx << ", a "
                                << declTy << ", not a scalar";
    Value operand = callOp.getOperand(idx);
    if (!operand.getType().isInteger(32))
      return callOp.emitError() << "scalar call operand at wrapper position "
                                << pos << " maps to argument " << idx << ", a "
                                << operand.getType() << " operand, not i32";
    return operand;
  }

  llvm::APInt constVal;
  if (matchPattern(arg, m_ConstantInt(&constVal)))
    return b
        .create<arith::ConstantIntOp>(loc, constVal.getSExtValue(),
                                      /*width=*/32)
        .getResult();

  return store.emitError()
         << "hoisted scalar call operand is neither a wrapper argument, a "
            "constant, nor an earlier scalar call result";
}

/// Emit one scalar call: the library call, and the store of its
/// result into the shared buffer at offset bytes. Returns the result.
static Value emitScalarCall(OpBuilder &b, Location loc, ModuleOp mainMod,
                            StringRef calleeName, ValueRange operands,
                            Value sharedBuf, MemRefType shTy, int64_t offset) {
  SmallVector<Type> argTys(operands.getTypes());
  auto libDecl =
      getOrCreateLibDecl(b, mainMod, calleeName, argTys, {b.getI32Type()});
  auto libCall = b.create<func::CallOp>(loc, libDecl, operands);
  Value result = libCall.getResult(0);

  auto viewTy =
      MemRefType::get({1}, b.getI32Type(), AffineMap{}, shTy.getMemorySpace());
  Value byteShift = b.create<arith::ConstantIndexOp>(loc, offset);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value view =
      b.create<memref::ViewOp>(loc, viewTy, sharedBuf, byteShift, ValueRange{});
  b.create<memref::StoreOp>(loc, result, view, ValueRange{zero});
  return result;
}

/// OutlineScope creates one call site per uniquely named SIMT wrapper.
struct WrapperWork {
  LLVM::LLVMFuncOp wrapper;
  SmallVector<LLVM::StoreOp> stores;
  func::CallOp callSite = nullptr;
};

/// The hoisted scalar calls found in the SIMT modules, and the main module
/// they are hoisted into.
struct SimtScalarCallWork {
  ModuleOp mainMod = nullptr;
  /// Keyed by wrapper name. MapVector, so diagnostics follow IR order instead
  /// of a hash map's unspecified one.
  llvm::MapVector<StringRef, WrapperWork> byName;
};

/// Identify the main module and index every SIMT module's hoisted calls.
static FailureOr<SimtScalarCallWork> collectSimtScalarCalls(ModuleOp topM) {
  SimtScalarCallWork work;
  WalkResult walked = topM.walk([&](LLVM::StoreOp store) {
    if (!store->hasAttr(kShmemOffsetStampName))
      return WalkResult::advance();
    auto func = store->getParentOfType<LLVM::LLVMFuncOp>();
    auto parentMod = store->getParentOfType<ModuleOp>();
    if (!func || !parentMod ||
        !parentMod->hasAttr(hacc::SIMTModuleAttr::name)) {
      store.emitError() << "hoisted scalar store is not inside a SIMT wrapper "
                           "in a hacc.simt_module";
      return WalkResult::interrupt();
    }
    auto [it, inserted] = work.byName.insert({func.getName(), {func, {}}});
    if (!inserted && it->second.wrapper != func) {
      func.emitError() << "SIMT wrapper '" << func.getName()
                       << "' holds hoisted scalar calls under a name "
                          "another SIMT module already claimed";
      return WalkResult::interrupt();
    }
    it->second.stores.push_back(store);
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();

  // Nothing hoisted: fast-div off, or no division was optimized.
  if (work.byName.empty())
    return work;
  for (auto nested : topM.getOps<ModuleOp>()) {
    if (nested->hasAttr(hacc::SIMTModuleAttr::name))
      continue;
    if (work.mainMod) {
      nested.emitError() << "more than one main module exists, which "
                            "hoisting the SIMT scalar calls does not support";
      return failure();
    }
    work.mainMod = nested;
  }
  return work;
}

/// Find the unique call site for each wrapper.
static LogicalResult collectCallSites(SimtScalarCallWork &work) {
  work.mainMod.walk([&](func::CallOp callOp) {
    auto it = work.byName.find(callOp.getCallee());
    if (it == work.byName.end())
      return;
    assert(!it->second.callSite && "expected one call site per SIMT scope");
    it->second.callSite = callOp;
  });

  bool unhoisted = false;
  for (auto &[name, entry] : work.byName) {
    if (entry.callSite)
      continue;
    unhoisted = true;
    entry.wrapper.emitError()
        << "SIMT wrapper '" << name
        << "' holds hoisted scalar calls but no matching call site was "
           "found in the main module";
  }
  return failure(unhoisted);
}

/// Drop the wrapper's copy, the SIMD caller performs the write now
static void eraseHoistedOps(ArrayRef<LLVM::StoreOp> stores) {
  SmallVector<LLVM::CallOp> calls;
  SmallVector<LLVM::GEPOp> geps;
  for (LLVM::StoreOp store : stores) {
    if (auto call = store.getValue().getDefiningOp<LLVM::CallOp>())
      calls.push_back(call);
    if (auto gep = store.getAddr().getDefiningOp<LLVM::GEPOp>())
      geps.push_back(gep);
    store.erase();
  }
  for (LLVM::GEPOp gep : geps)
    if (gep->use_empty())
      gep.erase();
  // Reverse program order, so a call feeding a later call dies with it.
  for (LLVM::CallOp call : llvm::reverse(calls))
    if (call->use_empty())
      call.erase();
}

class HoistSimtScalarCallsToSimd
    : public impl::HoistSimtScalarCallsToSimdBase<HoistSimtScalarCallsToSimd> {
public:
  void runOnOperation() override {
    ModuleOp topM = getOperation();

    FailureOr<SimtScalarCallWork> work = collectSimtScalarCalls(topM);
    if (failed(work))
      return signalPassFailure();

    if (work->byName.empty())
      return;

    // Hoisted calls with nowhere to go. Reports every wrapper.
    if (!work->mainMod) {
      for (auto &[name, entry] : work->byName)
        entry.wrapper.emitError()
            << "SIMT wrapper '" << name
            << "' holds hoisted scalar calls but there is no main "
               "module to hoist them into";
      return signalPassFailure();
    }

    if (failed(collectCallSites(*work)))
      return signalPassFailure();

    for (auto &[name, entry] : work->byName) {
      if (failed(hoistAtCallSite(entry.callSite, entry.wrapper, entry.stores,
                                 work->mainMod)))
        return signalPassFailure();
      eraseHoistedOps(entry.stores);
    }
  }

private:
  /// Rebuilds one wrapper's hoisted scalar calls
  ///
  /// Before, in the SIMT module. The calls sit in the `llvm.func` wrapper and
  /// write the shared buffer through a raw pointer, which GSS cannot see.
  ///
  ///   llvm.func @k_scope_0(%arg1: i32,
  ///                        %arg3: !llvm.ptr<6> {hivm.shared_memory}, ...) {
  ///     %2 = llvm.call @..._magic_shift(%arg1) : (i32) -> i32
  ///     %3 = llvm.getelementptr %arg3[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
  ///     llvm.store %2, %3 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
  ///     %4 = llvm.call @..._magic_mul(%arg1, %2) : (i32, i32) -> i32
  ///     ...
  ///     hivm_regbaseintrins.intrins.launch_func @k_scope_0_vf_simt ...
  ///
  /// After, in the main module, immediately before the call:
  ///
  ///   %0 = call @..._magic_shift(%arg2) : (i32) -> i32
  ///   %view = memref.view %alloc_1[%c0][] : memref<20xi8> to memref<1xi32>
  ///   memref.store %0, %view[%c0] {hivm.tcore_type = <VECTOR>} : memref<1xi32>
  ///   %1 = call @..._magic_mul(%arg2, %0) : (i32, i32) -> i32
  ///   ...
  ///   call @k_scope_0(%alloc, %arg2, %alloc_0, %alloc_1)
  ///
  /// Op by op:
  ///   llvm.call            -> func.call, declared by getOrCreateLibDecl
  ///   getelementptr+store  -> memref.view + memref.store, the form GSS
  ///                           aliases; tcore_type<VECTOR> because otherwise
  ///                           getCoreType yields CUBE_OR_VECTOR and the
  ///                           cross-core solver asserts
  ///   !llvm.ptr<6>         -> the memref<20xi8> call operand carrying
  ///                           hivm.shared_memory
  ///   byte offset          -> memref.view shift, from the store's stamp
  ///
  LogicalResult hoistAtCallSite(func::CallOp callOp, LLVM::LLVMFuncOp wrapper,
                                ArrayRef<LLVM::StoreOp> stores,
                                ModuleOp mainMod) {
    MLIRContext *ctx = &getContext();
    Location loc = callOp.getLoc();

    auto calleeDecl = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!calleeDecl)
      return callOp.emitError() << "callee declaration not found";

    // Locate the shared buffer through the DECLARATION: WriteBackShared
    // replaces the alloc without re-setting hivm.shared_memory.
    std::optional<unsigned> sharedArgIdx;
    for (unsigned i = 0, e = calleeDecl.getNumArguments(); i < e; ++i)
      if (calleeDecl.getArgAttr(i, SharedMemoryAttr::name)) {
        sharedArgIdx = i;
        break;
      }
    if (!sharedArgIdx)
      return callOp.emitError() << "callee has hoisted scalar calls but no "
                                   "hivm.shared_memory argument";

    Value sharedBuf = callOp.getOperand(*sharedArgIdx);
    auto shTy = dyn_cast<MemRefType>(sharedBuf.getType());
    if (!shTy || !shTy.hasStaticShape() || shTy.getRank() != 1 ||
        !shTy.getElementType().isInteger(8) || !shTy.getLayout().isIdentity())
      return callOp.emitError()
             << "shared-memory buffer is not a static 1-D i8 identity memref";

    OpBuilder b(callOp);
    SmallVector<unsigned> argIndices = getOriginalArgIndices(calleeDecl);
    // Wrapper call result -> the caller-side value that replaces it.
    DenseMap<Value, Value> recovered;

    for (LLVM::StoreOp store : stores) {
      auto offsetAttr =
          store->getAttrOfType<IntegerAttr>(kShmemOffsetStampName);
      if (!offsetAttr)
        return store.emitError()
               << kShmemOffsetStampName << " is not an integer";
      auto libCall = store.getValue().getDefiningOp<LLVM::CallOp>();
      if (!libCall)
        return store.emitError()
               << "stamped store's value is not produced by an llvm.call";
      std::optional<StringRef> calleeName = libCall.getCallee();
      if (!calleeName)
        return store.emitError() << "hoisted scalar call is indirect";

      int64_t offset = offsetAttr.getInt();
      if (offset < 0 || offset % 4 != 0 || offset + 4 > shTy.getDimSize(0))
        return callOp.emitError() << "scalar call shmem_offset " << offset
                                  << " out of range for buffer of "
                                  << shTy.getDimSize(0) << " bytes";

      SmallVector<Value> operands;
      for (Value arg : libCall.getArgOperands()) {
        FailureOr<Value> operand =
            resolveScalarCallArg(arg, store, wrapper, callOp, calleeDecl,
                                 argIndices, recovered, b, loc);
        if (failed(operand))
          return failure();
        operands.push_back(*operand);
      }

      recovered[store.getValue()] = emitScalarCall(
          b, loc, mainMod, *calleeName, operands, sharedBuf, shTy, offset);
    }

    // Declare the VF's effect so the sync solver sees the dependency.
    calleeDecl.setArgAttr(*sharedArgIdx, MemoryEffectAttr::name,
                          MemoryEffectAttr::get(ctx, MemoryEffect::READ));
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createHoistSimtScalarCallsToSimdPass() {
  return std::make_unique<HoistSimtScalarCallsToSimd>();
}
