//===- HIVMAVEToStandard.cpp - conversion from HIVMAVE to Standard dialect ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <string>

#include "bishengir/Dialect/HIVMAVE/Interfaces/AVELibraryFunctionOpInterface.cpp.inc"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMAVETOSTANDARD
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

namespace mlir::hivmave {
static LLVM::LLVMStructType getCCEI64VectorStructType(OpBuilder &rewriter,
                                                      MLIRContext *ctx) {
  // struct vector_2xvl_s64 { [2 x vector<64xi32>] }
  Type i32Type = rewriter.getI32Type();
  Type vectorType = VectorType::get({64}, i32Type);
  Type arrayType = LLVM::LLVMArrayType::get(vectorType, 2);
  llvm::StringRef structName = "vector_2xvl_s64";
  auto identified = LLVM::LLVMStructType::getIdentified(ctx, structName);
  if (mlir::failed(identified.setBody({arrayType}, /*isPacked=*/false)))
    llvm_unreachable("creating struct failed!");
  return identified;
}

static inline bool isVL64xI64(VectorType vt) {
  return vt.getElementType().isInteger(64) &&
         vt.getShape() == ArrayRef<int64_t>({64});
}

// create a new llvm.alloca <struct> with count=1.
static Value createVLAlloca(OpBuilder &rewriter, Location loc) {
  MLIRContext *ctx = rewriter.getContext();
  auto structTy = getCCEI64VectorStructType(rewriter, ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  Value one =
      rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
  return rewriter.create<LLVM::AllocaOp>(loc, ptrTy, structTy, one);
}

static MemRefType makeStridedLayoutAndShapeDynamic(MemRefType type) {
  return MemRefType::Builder(type)
      .setLayout(StridedLayoutAttr::get(
          type.getContext(), ShapedType::kDynamic,
          SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)))
      .setShape(SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic));
}

static VectorType adjustVLVectorType(VectorType type) {
  if (type.getElementType().isInteger(1))
    return VectorType::get({hivm::util::PREDICATE_BITS}, type.getElementType());
  if (type.getElementType().isInteger(64))
    // already has special process for i64/u64
    return type;
  uint64_t totalSize = static_cast<uint64_t>(type.getNumElements());
  Type elementType = type.getElementType();
  auto dataWidth = elementType.getIntOrFloatBitWidth();
  auto vlLength = hivm::util::VL_BITS / dataWidth;
  if (totalSize != vlLength) {
    return VectorType::get(SmallVector<int64_t>{vlLength}, elementType);
  }
  return type;
}

static SmallVector<Type> convertResultTypes(TypeRange resultTypes) {
  return llvm::map_to_vector(resultTypes, [](Type type) -> Type {
    if (auto vt = dyn_cast<VectorType>(type))
      return adjustVLVectorType(vt);
    return type;
  });
}

// get the correct function arg types
static SmallVector<Type> computeDefaultLibArgTypes(ValueRange inputOperands) {
  SmallVector<Type> argTypes;
  argTypes.reserve(inputOperands.size());
  for (Value v : inputOperands) {
    Type t = v.getType();
    if (auto mt = dyn_cast<MemRefType>(t))
      argTypes.push_back(makeStridedLayoutAndShapeDynamic(mt));
    else if (auto vt = dyn_cast<VectorType>(t))
      argTypes.push_back(adjustVLVectorType(vt));
    else
      argTypes.push_back(t);
  }
  return argTypes;
}

// cast to match opreand to same type the function wants
static SmallVector<Value>
createOperandsTypeConversionTo(PatternRewriter &rewriter, Location loc,
                               ValueRange operands, TypeRange targetTypes) {
  SmallVector<Value> res;
  res.reserve(operands.size());

  for (auto it : llvm::enumerate(operands)) {
    Value v = it.value();
    Type srcTy = v.getType();
    Type tgtTy = targetTypes[it.index()];
    // identical -> nothing to do
    if (srcTy == tgtTy) {
      res.push_back(v);
      continue;
    }
    // MemRef -> MemRef: memref.cast is safe
    if (isa<MemRefType>(srcTy) && isa<MemRefType>(tgtTy)) {
      res.push_back(rewriter.create<memref::CastOp>(loc, tgtTy, v));
      continue;
    }
    // Fallback: UnrealizedConversionCastOp
    res.push_back(rewriter.create<UnrealizedConversionCastOp>(loc, tgtTy, v)
                      ->getResult(0));
  }
  return res;
}

// convert call results back to original expected types. Use UCC for ptr->vec.
static SmallVector<Value> createResultsTypeConversion(PatternRewriter &rewriter,
                                                      Location loc,
                                                      ValueRange callResults,
                                                      TypeRange origResTypes) {
  SmallVector<Value> res;
  res.reserve(callResults.size());
  for (auto it : llvm::enumerate(callResults)) {
    Value val = it.value();
    Type valTy = val.getType();
    Type wantTy = origResTypes[it.index()];

    // identical
    if (valTy == wantTy) {
      res.push_back(val);
      continue;
    }
    // UCC
    res.push_back(rewriter.create<UnrealizedConversionCastOp>(loc, wantTy, val)
                      ->getResult(0));
  }
  return res;
}

// function to create a matching func.func symbol in module so no error when
// creating a call op
static func::FuncOp ensureLibFunc(PatternRewriter &rewriter, ModuleOp mod,
                                  StringRef name, TypeRange argTypes,
                                  TypeRange resTypes, Location loc) {
  if (auto f = mod.lookupSymbol<func::FuncOp>(name))
    return f;
  OpBuilder::InsertionGuard g(rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto fileLoc = mlir::FileLineColLoc::get(ctx, "internal", 0, 0);
  rewriter.setInsertionPoint(mod.getBody(), std::prev(mod.getBody()->end()));

  auto fn = rewriter.create<func::FuncOp>(
      fileLoc, name, rewriter.getFunctionType(argTypes, resTypes));

  fn->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), UnitAttr::get(ctx));
  fn->setAttr(mlir::hivm::TFuncCoreTypeAttr::name,
              hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));

  if (!name.ends_with("_tmp")) {
    auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
        hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
    fn->setAttr(haccAlwaysInlineAttr, rewriter.getUnitAttr());
  }

  fn.setPrivate();
  return fn;
}

// This create the function calls
static SmallVector<Value> createLibCall(PatternRewriter &rewriter,
                                        Operation *op, ModuleOp mod,
                                        StringRef libCallName,
                                        ValueRange inputOperands,
                                        TypeRange resultTypes) {
  Location loc = op->getLoc();

  SmallVector<Type> defaultLibResTypes = convertResultTypes(resultTypes);

  SmallVector<Type> argTypesForCall;
  SmallVector<Type> resTypesForCall;

  argTypesForCall = computeDefaultLibArgTypes(inputOperands);
  resTypesForCall = defaultLibResTypes;

  SmallVector<Value> callArgs = createOperandsTypeConversionTo(
      rewriter, loc, inputOperands, argTypesForCall);

  (void)ensureLibFunc(rewriter, mod, libCallName, argTypesForCall,
                      resTypesForCall, loc);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto call = rewriter.create<func::CallOp>(loc, libCallName, resTypesForCall,
                                            callArgs);

  return createResultsTypeConversion(rewriter, loc, call.getResults(),
                                     resultTypes);
}

bool returnsI64Vec(Type t) {
  if (auto vt = dyn_cast<VectorType>(t))
    return isVL64xI64(vt);
  return false;
}

static void replaceWithLibCall(PatternRewriter &rewriter, Operation *op,
                               StringRef libCallName,
                               SmallVector<Value> inputOperands,
                               TypeRange resultTypes) {
  ModuleOp mod = op->template getParentOfType<ModuleOp>();
  Location loc = op->getLoc();

  // detect which results need sret (i64 VL vectors)
  SmallVector<char> isSRet(resultTypes.size(), 0);
  bool anySRet = false;
  for (size_t i = 0; i < resultTypes.size(); ++i) {
    if (returnsI64Vec(resultTypes[i])) {
      isSRet[i] = 1;
      anySRet = true;
    }
  }

  // If there are no sret results, keep the old (non-sret) path including the
  // special load-op handling.
  if (!anySRet) {
    // special case for load op, in hivm level the return type is 1xi64
    if (auto loadOp = dyn_cast<VFLoadOp>(op)) {
      if (loadOp.getPattern() == LoadDist::BRC_B64) {
        std::string newName = libCallName.str();
        newName += "_tmp";
        llvm::StringRef newLibCallName(newName);
        // make it return 64xi64 but this is just temporary funcion (after
        // pattern will replace with sret function)
        VectorType targetVt =
            VectorType::get({64}, rewriter.getIntegerType(64));
        auto newResults = createLibCall(rewriter, op, mod, newLibCallName,
                                        inputOperands, targetVt);
        Value ucc = rewriter
                        .create<UnrealizedConversionCastOp>(loc, resultTypes[0],
                                                            newResults[0])
                        ->getResult(0);
        rewriter.replaceOp(op, ucc);
        return;
      }
    }

    auto newResults = createLibCall(rewriter, op, mod, libCallName,
                                    inputOperands, resultTypes);
    rewriter.replaceOp(op, newResults);
    return;
  }

  // --- sret path for one-or-more sret results ---
  // Create an alloca per sret-result and keep them in a vector indexed by
  // result position so we can map them back afterwards.
  SmallVector<Value> sretPtrs(resultTypes.size(), Value());
  for (size_t i = 0; i < resultTypes.size(); ++i) {
    if (isSRet[i]) {
      sretPtrs[i] = createVLAlloca(rewriter, loc);
    }
  }

  // Insert the sret pointers at the beginning of the call arguments.
  // Insert from last to first so final order (front.. ) matches result order.
  for (int i = (int)resultTypes.size() - 1; i >= 0; --i) {
    if (isSRet[i]) {
      inputOperands.insert(inputOperands.begin(), sretPtrs[i]);
    }
  }

  // Build the call's return types: only those that are NOT sret.
  SmallVector<Type> resTypesForCall;
  resTypesForCall.reserve(resultTypes.size());
  for (size_t i = 0; i < resultTypes.size(); ++i) {
    if (!isSRet[i])
      resTypesForCall.push_back(resultTypes[i]);
  }

  // Make the call. For sret results the callee returns void and fills the
  // provided pointers; for others the call returns values.
  SmallVector<Value> callResults = createLibCall(
      rewriter, op, mod, libCallName, inputOperands, resTypesForCall);

  // Now assemble final results in the original order:
  // - for sret results: UCC from the corresponding sret pointer -> expected vec
  // - for non-sret results: consume from callResults in order
  SmallVector<Value> finalResults;
  finalResults.reserve(resultTypes.size());
  size_t callIdx = 0;
  for (size_t i = 0; i < resultTypes.size(); ++i) {
    if (isSRet[i]) {
      Value vec = rewriter
                      .create<UnrealizedConversionCastOp>(loc, resultTypes[i],
                                                          sretPtrs[i])
                      ->getResult(0);
      finalResults.push_back(vec);
    } else {
      // take next returned value from callResults
      finalResults.push_back(callResults[callIdx++]);
    }
  }

  // Replace op with exactly as many values as original results had.
  rewriter.replaceOp(op, finalResults);
}

static SmallVector<Value> adjustInputs(Operation *op, PatternRewriter &rewriter,
                                       SmallVector<Value> inputs) {
  SmallVector<Value> out;
  Location loc = op->getLoc();
  if (auto storeOp = dyn_cast<VFMaskedStoreOp>(op)) {
    if (storeOp.getPattern() == StoreDist::ONEPT_B64 && !inputs.empty()) {
      Value src = inputs.pop_back_val();
      Type i64Type = rewriter.getI64Type();
      Type vectorType = VectorType::get({64}, i64Type);
      Value structVal =
          rewriter.create<UnrealizedConversionCastOp>(loc, vectorType, src)
              ->getResult(0);
      inputs.push_back(structVal);
    }
  }
  for (Value v : inputs) {
    Type t = v.getType();
    if (auto vt = dyn_cast<VectorType>(t)) {
      // if this is a predicate (i1 element), leave it
      if (vt.getElementType().isInteger(1)) {
        out.push_back(v);
        continue;
      }
      if (isVL64xI64(vt)) {
        // convert vector 64xi64 to struct pointer type via UCC
        auto ctx = rewriter.getContext();
        auto ptrtype = LLVM::LLVMPointerType::get(ctx);
        Value structVal =
            rewriter.create<UnrealizedConversionCastOp>(loc, ptrtype, v)
                ->getResult(0);
        out.push_back(structVal);
        continue;
      }
      out.push_back(v);
    } else {
      out.push_back(v);
    }
  }

  if (auto storeOp = dyn_cast<VFMaskedStoreOp>(op)) {
    auto loc = storeOp->getLoc();
    if (storeOp.getPattern() != StoreDist::ONEPT_B64 && storeOp->hasAttr(UnalignedAttr::name)) {
      Value data = storeOp.getVal();
      VectorType dtype = dyn_cast<VectorType>(data.getType());
      Value elementCount = getElemSizeByStoreMask(
          storeOp.getMask(), dtype.getElementType(), loc, rewriter, true);
      out.push_back(elementCount);
    }
  }

  return out;
}

// ---------- AVEOpToLibraryCallPattern ----------

// helper function for AVEOpToLibraryCallPattern
namespace {
static TypeRange adjustResultType(Operation *op, PatternRewriter &rewriter,
                                  TypeRange resultTypes) {
  return resultTypes;
}
} // namespace

template <typename AVEOpTy>
struct AVEOpToLibraryCallPattern : public OpRewritePattern<AVEOpTy> {
  explicit AVEOpToLibraryCallPattern(MLIRContext *context,
                                     PatternBenefit benefit = 10)
      : OpRewritePattern<AVEOpTy>(context, benefit) {}

  LogicalResult matchAndRewrite(AVEOpTy op,
                                PatternRewriter &rewriter) const override {

    AVEOpWithLibraryFunction iface =
        dyn_cast<AVEOpWithLibraryFunction>(op.getOperation());
    if (!iface)
      return failure();
    std::string lib = iface.getOpLibraryCallName();
    if (lib.empty())
      return failure();
    SmallVector<Value> ins = iface.getOpLibraryCallInputs(rewriter);
    SmallVector<Value> adjustedInput = adjustInputs(op, rewriter, ins);
    TypeRange adjustedReturnType =
        adjustResultType(op, rewriter, op->getResultTypes());
    replaceWithLibCall(rewriter, op, lib, adjustedInput, adjustedReturnType);

    return success();
  }
};

template <>
struct AVEOpToLibraryCallPattern<VFDivOp> : public OpRewritePattern<VFDivOp> {
  explicit AVEOpToLibraryCallPattern(MLIRContext *context,
                                     PatternBenefit benefit = 10)
      : OpRewritePattern<VFDivOp>(context, benefit) {}

  LogicalResult matchAndRewrite(VFDivOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasInt(op.getOperation()))
      return failure();
    AVEOpWithLibraryFunction iface =
        dyn_cast<AVEOpWithLibraryFunction>(op.getOperation());
    if (!iface)
      return failure();
    std::string lib = iface.getOpLibraryCallName();
    if (lib.empty())
      return failure();
    if (lib.find("int") == std::string::npos)
      return failure();
    SmallVector<Value> ins = iface.getOpLibraryCallInputs(rewriter);
    SmallVector<Value> adjustedInput = adjustInputs(op, rewriter, ins);
    TypeRange adjustedReturnType =
        adjustResultType(op, rewriter, op->getResultTypes());
    replaceWithLibCall(rewriter, op, lib, adjustedInput, adjustedReturnType);

    return success();
  }

private:
  static bool hasInt(Operation *op) {
    for (Type t : op->getOperandTypes())
      if (isInt(t))
        return true;
    for (Type t : op->getResultTypes())
      if (isInt(t))
        return true;
    return false;
  }

  // i16/i32/i64/u16/u32/u64
  static bool isInt(Type type) {
    if (auto vec = dyn_cast<VectorType>(type)) {
      auto type = vec.getElementType();
      return type.isInteger(64) || type.isInteger(32) || type.isInteger(16);
    }
    return false;
  }
};

struct FixTempFuncCallPattern : public OpRewritePattern<func::CallOp> {
  explicit FixTempFuncCallPattern(MLIRContext *ctx, PatternBenefit b = 5)
      : OpRewritePattern<func::CallOp>(ctx, b) {}

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    StringRef callee = call.getCallee();
    if (!callee.ends_with("_tmp"))
      return failure();

    if (call.getNumResults() != 1)
      return failure();

    Type resTy = call.getResult(0).getType();
    auto vt = dyn_cast<VectorType>(resTy);
    if (!vt || !isVL64xI64(vt))
      return failure();

    StringRef realName = callee.drop_back(4);
    Location loc = call.getLoc();
    ModuleOp mod = call->getParentOfType<ModuleOp>();
    if (!mod)
      return failure();

    SmallVector<Value> inputs(call.getOperands().begin(),
                              call.getOperands().end());
    inputs = adjustInputs(call, rewriter, inputs);

    // sret: LLVM alloca first, call returns void
    Value sretPtr = createVLAlloca(rewriter, loc);
    inputs.insert(inputs.begin(), sretPtr);

    (void)createLibCall(rewriter, call.getOperation(), mod, realName, inputs,
                        TypeRange{});

    // Replace tmp call result by UCC ptr -> vector
    Value uccBack =
        rewriter.create<UnrealizedConversionCastOp>(loc, vt, sretPtr)
            ->getResult(0);
    rewriter.replaceOp(call, uccBack);
    return success();
  }
};

// Helper function to build LLVM struct from two vectors and return the alloca
// ptr.
Value buildLLVMStructFromTwoVecs(OpBuilder &builder, Location loc, Value vec0,
                                 Value vec1) {
  auto ctx = builder.getContext();

  // create an alloca for the struct
  Value structAlloca = createVLAlloca(builder, loc);

  // types
  Type vec64Ty = getCCEI64VectorStructType(builder, ctx);
  auto ptrToVec32Ty = LLVM::LLVMPointerType::get(ctx);

  // constants (i32 indices)
  Value i32zero =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  Value i32one =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32IntegerAttr(1));

  // get the pointer to the first 32 bit vector from the struct
  Value vec0ptr =
      builder.create<LLVM::GEPOp>(loc, ptrToVec32Ty, vec64Ty, structAlloca,
                                  ValueRange{i32zero, i32zero, i32zero}, true);

  // store vec0
  builder.create<LLVM::StoreOp>(loc, vec0, vec0ptr);

  // get the pointer to the second 32 bit vector from the struct
  Value vec1ptr =
      builder.create<LLVM::GEPOp>(loc, ptrToVec32Ty, vec64Ty, structAlloca,
                                  ValueRange{i32zero, i32zero, i32one}, true);

  // store vec1
  builder.create<LLVM::StoreOp>(loc, vec1, vec1ptr);

  return structAlloca;
}

// Load a struct from pointer via GEP and extract the two vector<64xi32> lanes.
static std::pair<Value, Value>
loadTwoVec32FromPtr(OpBuilder &builder, Location loc, Value structPtr) {

  auto ctx = builder.getContext();
  Type vec32Ty = VectorType::get({64}, builder.getI32Type());
  Type vec64Ty = getCCEI64VectorStructType(builder, ctx);
  auto ptrToVec32Ty = LLVM::LLVMPointerType::get(ctx);

  Value i32zero =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  Value i32one =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32IntegerAttr(1));

  // get pointer to the first vector struct
  Value vec0ptr =
      builder.create<LLVM::GEPOp>(loc, ptrToVec32Ty, vec64Ty, structPtr,
                                  ValueRange{i32zero, i32zero, i32zero}, true);
  // load it to vector<64xi32>
  Value vec0 = builder.create<LLVM::LoadOp>(loc, vec32Ty, vec0ptr);

  // get pointer to the second vector struct
  Value vec1ptr =
      builder.create<LLVM::GEPOp>(loc, ptrToVec32Ty, vec64Ty, structPtr,
                                  ValueRange{i32zero, i32zero, i32one}, true);
  // load it to vector<64xi32>

  Value vec1 = builder.create<LLVM::LoadOp>(loc, vec32Ty, vec1ptr);

  return {vec0, vec1};
}

struct SplitVectorForPattern : public OpRewritePattern<scf::ForOp> {
  explicit SplitVectorForPattern(MLIRContext *ctx, PatternBenefit benefit = 1)
      : OpRewritePattern<scf::ForOp>(ctx, benefit) {}

  LogicalResult matchAndRewrite(scf::ForOp oldFor,
                                PatternRewriter &rewriter) const override {
    Location loc = oldFor.getLoc();

    // get the loop body block and initial values (modern scf API)
    Block &oldBody = oldFor.getRegion().front();
    SmallVector<Value, 8> oldInitArgs(oldFor.getInitArgs().begin(),
                                      oldFor.getInitArgs().end());
    unsigned oldIterCount = oldInitArgs.size();

    // build new iter arg types & init args:
    SmallVector<Type, 8> newIterArgTypes;
    SmallVector<Value, 8> newInitArgs;
    bool anyExpanded = false;

    // expand init args
    expandInitArgs(rewriter, loc, oldInitArgs, newIterArgTypes, newInitArgs,
                   anyExpanded);
    if (!anyExpanded)
      return failure();

    // create the new scf.for with new init args (same bounds and step)
    Value lb = oldFor.getLowerBound();
    Value ub = oldFor.getUpperBound();
    Value step = oldFor.getStep();

    rewriter.setInsertionPoint(oldFor);
    scf::ForOp newFor =
        rewriter.create<scf::ForOp>(loc, lb, ub, step, newInitArgs);


    // clone body and map arguments
    cloneAndMapBody(rewriter, oldFor, newFor, oldBody, oldIterCount);

    // replace old results with recombined new results
    replaceOldResultsWithNew(rewriter, loc, oldFor, newFor, oldBody,
                             oldIterCount);

    rewriter.eraseOp(oldFor);
    return success();
  }

private:
  void expandInitArgs(PatternRewriter &rewriter, Location loc,
                      const SmallVector<Value, 8> &oldInitArgs,
                      SmallVector<Type, 8> &newIterArgTypes,
                      SmallVector<Value, 8> &newInitArgs,
                      bool &anyExpanded) const {
    for (Value oldInit : oldInitArgs) {
      Type t = oldInit.getType();
      if (auto vt = dyn_cast<VectorType>(t); vt && isVL64xI64(vt)) {
        VectorType vec32Ty = VectorType::get({64}, rewriter.getI32Type());
        newIterArgTypes.push_back(vec32Ty);
        newIterArgTypes.push_back(vec32Ty);

        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value maybePtr =
            rewriter.create<UnrealizedConversionCastOp>(loc, ptrType, oldInit)
                ->getResult(0);

        auto vec = loadTwoVec32FromPtr(rewriter, loc, maybePtr);
        newInitArgs.push_back(vec.first);
        newInitArgs.push_back(vec.second);
        anyExpanded = true;
      } else {
        newIterArgTypes.push_back(t);
        newInitArgs.push_back(oldInit);
      }
    }
  }

  void cloneAndMapBody(PatternRewriter &rewriter, scf::ForOp oldFor,
                       scf::ForOp newFor, Block &oldBody,
                       unsigned oldIterCount) const {
    Location loc = oldFor.getLoc();
    Block *newBody = newFor.getBody();
    rewriter.setInsertionPointToStart(newBody);

    mlir::IRMapping mapping;
    mapping.map(oldBody.getArgument(0), newBody->getArgument(0));

    unsigned newArgIdx = 1;
    for (unsigned i = 0; i < oldIterCount; ++i) {
      Value oldBlockArg = oldBody.getArgument(1 + i);
      Type oldT = oldBlockArg.getType();
      if (auto vt = dyn_cast<VectorType>(oldT); vt && isVL64xI64(vt)) {
        Value vec0 = newBody->getArgument(newArgIdx);
        Value vec1 = newBody->getArgument(newArgIdx + 1);
        newArgIdx += 2;

        Value structAlloca =
            buildLLVMStructFromTwoVecs(rewriter, loc, vec0, vec1);
        Value uccVec =
            rewriter.create<UnrealizedConversionCastOp>(loc, vt, structAlloca)
                ->getResult(0);
        mapping.map(oldBlockArg, uccVec);
      } else {
        mapping.map(oldBlockArg, newBody->getArgument(newArgIdx));
        newArgIdx++;
      }
    }

    for (Operation &op : oldBody) {
      if (isa<scf::YieldOp>(op))
        continue;
      rewriter.clone(op, mapping);
    }

    scf::YieldOp oldYield = nullptr;
    for (Operation &op : oldBody) {
      if ((oldYield = dyn_cast<scf::YieldOp>(&op)))
        break;
    }

    if (!oldYield)
      return;

    SmallVector<Value, 8> newYieldVals;
    for (Value oldVal : oldYield.getOperands()) {
      Value mappedVal = mapping.lookupOrDefault(oldVal);
      Type oldT = oldVal.getType();
      if (auto vt = dyn_cast<VectorType>(oldT); vt && isVL64xI64(vt)) {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        Value maybePtr =
            rewriter
                .create<UnrealizedConversionCastOp>(loc, ptrType, mappedVal)
                ->getResult(0);
        auto lanes = loadTwoVec32FromPtr(rewriter, loc, maybePtr);
        newYieldVals.push_back(lanes.first);
        newYieldVals.push_back(lanes.second);
      } else {
        newYieldVals.push_back(mappedVal);
      }
    }

    rewriter.setInsertionPointToEnd(newBody);
    rewriter.create<scf::YieldOp>(loc, newYieldVals);
    rewriter.setInsertionPointAfter(newFor);
  }

  void replaceOldResultsWithNew(PatternRewriter &rewriter, Location loc,
                                scf::ForOp oldFor, scf::ForOp newFor,
                                Block &oldBody, unsigned oldIterCount) const {
    SmallVector<Value, 8> newForResults(newFor.getResults().begin(),
                                        newFor.getResults().end());
    SmallVector<Value, 8> replacementResults;
    replacementResults.resize(oldIterCount);

    unsigned curPos = 0;
    for (unsigned i = 0; i < oldIterCount; ++i) {
      Type oldT = oldBody.getArgument(1 + i).getType();
      if (auto vt = dyn_cast<VectorType>(oldT); vt && isVL64xI64(vt)) {
        if (curPos + 1 >= newForResults.size()) {
          replacementResults[i] = rewriter.create<LLVM::UndefOp>(loc, oldT);
        } else {
          Value vec0 = newForResults[curPos];
          Value vec1 = newForResults[curPos + 1];
          Value structAlloca =
              buildLLVMStructFromTwoVecs(rewriter, loc, vec0, vec1);
          Value uccVec =
              rewriter
                  .create<UnrealizedConversionCastOp>(loc, vt, structAlloca)
                  ->getResult(0);
          replacementResults[i] = uccVec;
        }
        curPos += 2;
      } else {
        if (curPos >= newForResults.size())
          replacementResults[i] = rewriter.create<LLVM::UndefOp>(loc, oldT);
        else
          replacementResults[i] = newForResults[curPos];
        curPos += 1;
      }
    }

    for (unsigned i = 0; i < oldFor.getNumResults(); ++i)
      oldFor.getResult(i).replaceAllUsesWith(replacementResults[i]);
  }
};

// adding the patterns

void addSplitVectorForPattern(RewritePatternSet &patterns) {
  patterns.add<SplitVectorForPattern>(patterns.getContext(), 10);
}

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  PatternBenefit benefit = 10;
  patterns.add<AVEOpToLibraryCallPattern<OpType>>(patterns.getContext(),
                                                  benefit);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateHIVMAVEToStandardConversionPatterns(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVMAVE/IR/AVEOps.cpp.inc"
      >(patterns);
}
} // namespace mlir::hivmave

using namespace mlir;
using namespace mlir::hivmave;

namespace {
struct ConvertHIVMAVEToStandardPass
    : public impl::ConvertHIVMAVEToStandardBase<ConvertHIVMAVEToStandardPass> {

  void runOnOperation() override;
};
} // namespace

static LogicalResult applyAVELibCallPatterns(ModuleOp module,
                                             ConversionTarget target,
                                             MLIRContext *context) {
  RewritePatternSet patterns(context);
  populateHIVMAVEToStandardConversionPatterns(patterns);
  return applyPartialConversion(module, target, std::move(patterns));
}

static LogicalResult applyFixTempPatterns(ModuleOp module,
                                          MLIRContext *context) {
  RewritePatternSet patterns(context);
  patterns.add<FixTempFuncCallPattern>(context);
  return applyPatternsGreedily(module, std::move(patterns));
}

static LogicalResult applySplitVectorPatterns(ModuleOp module,
                                              MLIRContext *context) {
  RewritePatternSet patterns(context);
  addSplitVectorForPattern(patterns);
  return applyPatternsGreedily(module, std::move(patterns));
}

static LogicalResult applyCanonicalizationPatterns(ModuleOp module) {
  RewritePatternSet patterns(module->getContext());
  memref::SubViewOp::getCanonicalizationPatterns(patterns,
                                                 patterns.getContext());
  return applyPatternsGreedily(module, std::move(patterns));
}

void ConvertHIVMAVEToStandardPass::runOnOperation() {
  auto module = getOperation();
  auto *context = &getContext();

  ConversionTarget target(getContext());
  if (!hacc::utils::isAscend950(module)) {
    return;
  }

  // legal dialects and dynamic legality for AVE types (unchanged).
  target.addLegalDialect<func::FuncDialect, memref::MemRefDialect,
                         arith::ArithDialect, scf::SCFDialect,
                         LLVM::LLVMDialect>();

  target.addDynamicallyLegalDialect<AVEDialect>([](Operation *op) {
    auto isIllegalTypeI64 = [](Type type) {
      if (auto vecType = dyn_cast<VectorType>(type)) {
        return vecType.getElementType().isInteger(64);
      }
      return false;
    };

    auto isIllegalTypeI32I16 = [](Type type) {
      if (auto vecType = dyn_cast<VectorType>(type)) {
        return vecType.getElementType().isInteger(32) ||
               vecType.getElementType().isInteger(16);
      }
      return false;
    };

    if (isa<hivmave::VFDivOp>(*op)) {
      for (Type operandType : op->getOperandTypes()) {
        if (isIllegalTypeI32I16(operandType))
          return false;
      }
    }

    for (Type operandType : op->getOperandTypes()) {
      if (isIllegalTypeI64(operandType))
        return false;
    }
    for (Type resultType : op->getResultTypes()) {
      if (isIllegalTypeI64(resultType))
        return false;
    }
    return true;
  });

  // legal helper ops.
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<LLVM::LoadOp>();
  target.addLegalOp<LLVM::AllocaOp>();
  target.addLegalOp<LLVM::StoreOp>();

  // ops replaced here
  target.addIllegalOp<hivmave::VFDivFHPOp>();
  target.addIllegalOp<hivmave::VFModOp>();
  target.addIllegalOp<hivmave::VFModUIOp>();
  // apply all patterns
  if (failed(applyAVELibCallPatterns(module, target, context))) {
    module.emitWarning("Greedy AVE->libcall rewrite failed");
    return signalPassFailure();
  }

  if (failed(applyFixTempPatterns(module, context))) {
    module.emitWarning("FixTempFuncCallPattern did not fully apply");
  }

  if (failed(applySplitVectorPatterns(module, context))) {
    module.emitWarning("SplitVectorForPattern did not fully apply");
  }

  if (failed(applyCanonicalizationPatterns(module))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createConvertHIVMAVEToStandardPass() {
  return std::make_unique<ConvertHIVMAVEToStandardPass>();
}
