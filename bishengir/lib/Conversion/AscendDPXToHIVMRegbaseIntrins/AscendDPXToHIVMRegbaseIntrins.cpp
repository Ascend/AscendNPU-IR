//===- AscendDPXToHIVMRegbaseIntrins.cpp -===//
//===- Convert Ascend DPX dialect to HIVMRegbaseIntrins dialect -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------------------------===//

#include "bishengir/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXToHIVMRegbaseIntrins.h"
#include "bishengir/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXMathOpsLowering.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTASCENDDPXTOHIVMREGBASEINTRINS
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm_regbaseintrins;

const int GLOBAL_ADDR_SPACE = 1;
const int SHARED_ADDR_SPACE = 6;
const int MAX_LOADSTORE_BITWIDTH =
    64; // TODO: get this to 128 (stg.b128 doesn't work right now)

static int typeBitWidth(Type t) {
  if (t.isIntOrFloat())
    return t.getIntOrFloatBitWidth();
  else if (auto vt = llvm::dyn_cast<VectorType>(t)) {
    int numElements = vt.getNumElements();
    Type elementType = vt.getElementType();
    return typeBitWidth(elementType) * numElements;
  }
  return 0; // invalid
}

// takes a desired result and builds a single load to load to that result
static Value buildSingleLoad(ConversionPatternRewriter &rewriter, Location loc,
                             Type resultType,
                             TypedValue<LLVM::LLVMPointerType> ptr,
                             uint32_t cacheHint, uint32_t cacheOption,
                             uint32_t volatileOption) {
  int bitWidth = typeBitWidth(resultType);
  auto ptrType = ptr.getType();
  auto addrSpace = ptrType.getAddressSpace();
  if (addrSpace == SHARED_ADDR_SPACE || cacheHint == 0 || volatileOption) {
    // cache hints not supported
    return rewriter.create<LLVM::LoadOp>(loc, resultType, ptr, 0, volatileOption);
  } else if (addrSpace == GLOBAL_ADDR_SPACE) {
    std::string optionString = cacheOption ? "cache" : "uncache";
    std::string opName =
        "llvm.hivm.ldg." + optionString + ".s" + std::to_string(bitWidth);

    Value cacheHintConstant = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(cacheHint));

    SmallVector<Value> args = {ptr, cacheHintConstant};

    Value loadResult =
        rewriter
            .create<LLVM::CallIntrinsicOp>(
                loc, rewriter.getIntegerType(std::max(32, bitWidth)),
                rewriter.getStringAttr(opName), args)
            .getResult(0);

    if (bitWidth < 32) {
      // The result of the intrinsic op is an i32, which doesn't match the
      // desired bitwidth. Truncate
      loadResult = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getIntegerType(bitWidth), loadResult);
    }

    loadResult = rewriter.create<LLVM::BitcastOp>(loc, resultType, loadResult)
                     .getResult();

    return loadResult;
  }
  llvm_unreachable("ERROR: matchAndRewrite should have given match failure "
                   "before reaching this point!");
}

// forward declaration
static Value buildLoad(ConversionPatternRewriter &rewriter, Location loc,
                       Type resultType, TypedValue<LLVM::LLVMPointerType> ptr,
                       uint32_t cacheHint, uint32_t cacheOption,
                       uint32_t volatileOption);

// extracts values from src and inserts them to dst[offset:offset+len(src)]
static TypedValue<VectorType>
extractAndInsert_Load(ConversionPatternRewriter &rewriter, Location loc,
                      TypedValue<VectorType> src, VectorType srcType,
                      TypedValue<VectorType> dst, VectorType dstType,
                      int offset) {
  for (int i = 0; i < srcType.getNumElements(); i++) {
    Value const_i =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), i);
    Value item = rewriter.create<LLVM::ExtractElementOp>(
        loc, srcType.getElementType(), src, const_i);
    Value const_i_plus_offset = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), i + offset);
    dst = dyn_cast<TypedValue<VectorType>>(
        rewriter
            .create<LLVM::InsertElementOp>(loc, dstType, dst, item,
                                           const_i_plus_offset)
            .getResult());
  }
  return dst;
}

// this function takes in a desired result that is a vector of values and builds
// multiple loads to load the entire vector
static Value buildMultiLoad(ConversionPatternRewriter &rewriter, Location loc,
                            VectorType resultType,
                            TypedValue<LLVM::LLVMPointerType> ptr,
                            uint32_t cacheHint, uint32_t cacheOption,
                            uint32_t volatileOption) {
  int elemBitWidth = typeBitWidth(resultType.getElementType());
  int numElems = resultType.getNumElements();
  Type elemType = resultType.getElementType();
  Type ptrType = ptr.getType();
  // the number of elements we can fit into a single load is
  // MAX_LOADSTORE_WIDTH / elemBitWidth vector elements
  int strideLength = std::max(1, MAX_LOADSTORE_BITWIDTH / elemBitWidth);
  TypedValue<VectorType> finalVec = dyn_cast<TypedValue<VectorType>>(
      rewriter.create<LLVM::UndefOp>(loc, resultType).getResult());
  for (int i = 0; i < numElems; i += strideLength) {
    int strideEnd = std::min(numElems - i, strideLength);
    VectorType subLoadType = VectorType::get(strideEnd, elemType, false);
    LLVM::GEPOp GEPOp = rewriter.create<LLVM::GEPOp>( // getElementPtr
        loc, ptrType, elemType, ptr, ArrayRef<LLVM::GEPArg>(i));
    TypedValue<LLVM::LLVMPointerType> subLoadOffset =
        dyn_cast<TypedValue<LLVM::LLVMPointerType>>(GEPOp.getResult());
    assert(subLoadOffset != nullptr);
    Value subLoadResult = buildLoad(rewriter, loc, subLoadType, subLoadOffset,
                                    cacheHint, cacheOption, volatileOption);
    finalVec = extractAndInsert_Load(
        rewriter, loc, dyn_cast<TypedValue<VectorType>>(subLoadResult),
        subLoadType, finalVec, resultType, i);
  }
  return finalVec;
}

// if load width is greater than MAX_LOADSTORE_BITWIDTH, generate multiple
// single loads otherwise generate a single load
static Value buildLoad(ConversionPatternRewriter &rewriter, Location loc,
                       Type resultType, TypedValue<LLVM::LLVMPointerType> ptr,
                       uint32_t cacheHint, uint32_t cacheOption,
                       uint32_t volatileOption) {
  // currently it *appears* that large loads arent generated by
  // the upper triton to dpx, meaning this isn't fully tested
  int bitWidth = typeBitWidth(resultType);
  if (bitWidth <= MAX_LOADSTORE_BITWIDTH) {
    return buildSingleLoad(rewriter, loc, resultType, ptr, cacheHint,
                           cacheOption, volatileOption);
  } else if (auto vResult = dyn_cast<VectorType>(resultType)) {
    return buildMultiLoad(rewriter, loc, vResult, ptr, cacheHint, cacheOption,
                          volatileOption);
  } else {
    // bit cast from vector of int32s
    VectorType castFrom =
        VectorType::get(bitWidth / 32, rewriter.getI32Type(), false);
    Value loadValue = buildMultiLoad(rewriter, loc, castFrom, ptr, cacheHint,
                                     cacheOption, volatileOption);
    return rewriter.create<LLVM::BitcastOp>(loc, resultType, loadValue)
        .getResult();
  }
}

static DenseElementsAttr getDenseZeroAttr(VectorType vecType,
                                          ConversionPatternRewriter &rewriter) {
  auto elemType = vecType.getElementType();
  if (elemType.isInteger(1))
    return DenseElementsAttr::get(vecType, false);
  else if (elemType.isInteger())
    return DenseElementsAttr::get(vecType,
                                  rewriter.getIntegerAttr(elemType, 0ll));
  else if (auto f16Type = dyn_cast<Float16Type>(elemType)) {
    // For f16, use APFloat with proper semantics
    llvm::APFloat zeroVal(0.0f);
    bool losesInfo = false;
    zeroVal.convert(llvm::APFloatBase::IEEEhalf(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
    return DenseFPElementsAttr::get(vecType, zeroVal);
  } else if (auto bf16Type = dyn_cast<BFloat16Type>(elemType)) {
    // For bf16, use APFloat with proper semantics
    llvm::APFloat zeroVal(0.0f);
    bool losesInfo = false;
    zeroVal.convert(llvm::APFloatBase::BFloat(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
    return DenseFPElementsAttr::get(vecType, zeroVal);
  } else {
    return DenseElementsAttr::get(vecType,
                                  rewriter.getFloatAttr(elemType, 0.0f));
  }
}

struct AscendDPXLoadOpLowering
    : public ConvertOpToLLVMPattern<ascend_dpx::LoadOp> {
  explicit AscendDPXLoadOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<ascend_dpx::LoadOp>(converter) {}
  LogicalResult
  matchAndRewrite(ascend_dpx::LoadOp load, ascend_dpx::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = load.getLoc();
    auto ptr = cast<TypedValue<LLVM::LLVMPointerType>>(adaptor.getPtr());
    auto ptrType = cast<LLVM::LLVMPointerType>(ptr.getType());
    auto resultType = load.getResult().getType();
    auto mask = load.getMask();
    uint32_t cacheHint =
        static_cast<uint32_t>(load.getCacheHintAttr().getValue());
    uint32_t cacheOption =
        static_cast<uint32_t>(load.getCacheOptionAttr().getValue());
    uint32_t volatileOption =
        static_cast<uint32_t>(load.getVolatileOptionAttr().getValue());
    auto addrSpace = ptrType.getAddressSpace();
    if (addrSpace != 1 && addrSpace != 6)
      return rewriter.notifyMatchFailure(load,
                                         "address space must be either 1 or 6");

    if (mask) {
      auto ifOp = rewriter.create<scf::IfOp>(loc, resultType, mask, true);

      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto loadOp = buildLoad(rewriter, loc, resultType, ptr, cacheHint,
                              cacheOption, volatileOption);
      rewriter.create<scf::YieldOp>(loc, loadOp);

      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto falseVal = load.getFalseVal();
      if (!falseVal) {
        if (auto vrt = dyn_cast<VectorType>(resultType)) {
          falseVal = rewriter.create<LLVM::ConstantOp>(
              loc, resultType, getDenseZeroAttr(vrt, rewriter));
        } else {
          falseVal = rewriter.create<LLVM::ConstantOp>(loc, resultType, 0);
        }
      }
      rewriter.create<scf::YieldOp>(loc, falseVal);

      rewriter.replaceOp(load, ifOp);

    } else {
      auto loadOp = buildLoad(rewriter, loc, resultType, ptr, cacheHint,
                              cacheOption, volatileOption);
      rewriter.replaceOp(load, loadOp);
    }
    return success();
  }
};

// this function takes a value of bit width < MAX_LOADSTORE_BITWIDTH
// and simply stores it
static void buildSingleStore(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, TypedValue<LLVM::LLVMPointerType> ptr,
                             uint32_t cacheHint, uint32_t cacheOption) {
  Type valType = val.getType();
  LLVM::LLVMPointerType ptrType = ptr.getType();
  unsigned int addrSpace = ptrType.getAddressSpace();
  int storeWidth = typeBitWidth(valType);
  if (addrSpace == SHARED_ADDR_SPACE || cacheHint == 0) {
    // cache hints not supported
    rewriter.create<LLVM::StoreOp>(loc, val, ptr);
  } else if (addrSpace == GLOBAL_ADDR_SPACE) {
    std::string isCached = cacheOption ? "cache" : "uncache";
    std::string opName =
        "llvm.hivm.stg." + isCached + ".b" + std::to_string(storeWidth);

    Value cacheHintConstant = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(cacheHint));

    val = rewriter.create<LLVM::BitcastOp>(
        loc, rewriter.getIntegerType(storeWidth), val);

    if (storeWidth == 8) {
      val = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), val);
    } else if (storeWidth == 16) {
      val = rewriter.create<LLVM::BitcastOp>(loc, rewriter.getF16Type(), val);
    }

    SmallVector<Value> args = {ptr, val, cacheHintConstant};

    rewriter.create<LLVM::CallIntrinsicOp>(
        loc, LLVM::LLVMVoidType(), rewriter.getStringAttr(opName), args);
  } else {
    llvm_unreachable("ERROR: matchAndRewrite should have given match failure "
                     "before reaching this point!");
  }
}

// forward declaration
static void buildStore(ConversionPatternRewriter &rewriter, Location loc,
                       Value val, TypedValue<LLVM::LLVMPointerType> ptr,
                       uint32_t cacheHint, uint32_t cacheOption);

// extracts values from src[offset:offsetlen(dst)] and inserts them to dst
static TypedValue<VectorType>
extractAndInsert_Store(ConversionPatternRewriter &rewriter, Location loc,
                       TypedValue<VectorType> src, VectorType srcType,
                       TypedValue<VectorType> dst, VectorType dstType,
                       int offset) {
  for (int i = 0; i < dstType.getNumElements(); i++) {
    Value const_i =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), i);
    Value const_i_plus_offset = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), i + offset);
    Value item = rewriter.create<LLVM::ExtractElementOp>(
        loc, srcType.getElementType(), src, const_i_plus_offset);
    dst = dyn_cast<TypedValue<VectorType>>(
        rewriter.create<LLVM::InsertElementOp>(loc, dstType, dst, item, const_i)
            .getResult());
  }
  return dst;
}

// this function takes a vector of values then partitions the vector and
// builds a single store for each segment.
static void buildMultiStore(ConversionPatternRewriter &rewriter, Location loc,
                            TypedValue<VectorType> val,
                            TypedValue<LLVM::LLVMPointerType> ptr,
                            uint32_t cacheHint, uint32_t cacheOption) {
  VectorType valType = val.getType();
  int elemBitWidth = typeBitWidth(valType.getElementType());
  int numElems = valType.getNumElements();
  Type elemType = valType.getElementType();
  Type ptrType = ptr.getType();
  int strideLength = std::max(1, MAX_LOADSTORE_BITWIDTH / elemBitWidth);
  for (int i = 0; i < numElems; i += strideLength) {
    int strideEnd = std::min(numElems - i, strideLength);
    VectorType subStoreType = VectorType::get(strideEnd, elemType, false);
    TypedValue<VectorType> subStoreValue = dyn_cast<TypedValue<VectorType>>(
        rewriter.create<LLVM::UndefOp>(loc, subStoreType).getResult());
    LLVM::GEPOp GEPOp = rewriter.create<LLVM::GEPOp>( // getElementPtr
        loc, ptrType, elemType, ptr, ArrayRef<LLVM::GEPArg>(i));
    TypedValue<LLVM::LLVMPointerType> subStoreOffset =
        dyn_cast<TypedValue<LLVM::LLVMPointerType>>(GEPOp.getResult());
    assert(subStoreOffset != nullptr);
    subStoreValue = extractAndInsert_Store(rewriter, loc, val, valType,
                                           subStoreValue, subStoreType, i);
    buildStore(rewriter, loc, subStoreValue, subStoreOffset, cacheHint,
               cacheOption);
  }
}

// This function decides whether to build a single store, build multiple stores,
// or bitcast to a more favourable type and try build again
static void buildStore(ConversionPatternRewriter &rewriter, Location loc,
                       Value val, TypedValue<LLVM::LLVMPointerType> ptr,
                       uint32_t cacheHint, uint32_t cacheOption) {
  int bitWidth = typeBitWidth(val.getType());
  if (bitWidth <= MAX_LOADSTORE_BITWIDTH) {
    buildSingleStore(rewriter, loc, val, ptr, cacheHint, cacheOption);
  } else if (auto vecVal = dyn_cast<TypedValue<VectorType>>(val)) {
    buildMultiStore(rewriter, loc, vecVal, ptr, cacheHint, cacheOption);
  } else {
    // cast to vector of i32s and try again
    VectorType castTo =
        VectorType::get(bitWidth / 32, rewriter.getI32Type(), false);
    TypedValue<VectorType> castedVal = dyn_cast<TypedValue<VectorType>>(
        rewriter.create<LLVM::BitcastOp>(loc, castTo, val).getResult());
    buildMultiStore(rewriter, loc, castedVal, ptr, cacheHint, cacheOption);
  }
}

struct AscendDPXStoreOpLowering
    : public ConvertOpToLLVMPattern<ascend_dpx::StoreOp> {
  explicit AscendDPXStoreOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<ascend_dpx::StoreOp>(converter) {}
  LogicalResult
  matchAndRewrite(ascend_dpx::StoreOp store,
                  ascend_dpx::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = store.getLoc();
    auto ptr = cast<TypedValue<LLVM::LLVMPointerType>>(adaptor.getPtr());
    auto ptrType = cast<LLVM::LLVMPointerType>(ptr.getType());
    auto val = store.getValue();
    auto mask = store.getMask();
    auto cacheHint = static_cast<uint32_t>(store.getCacheHintAttr().getValue());
    auto cacheOption =
        static_cast<uint32_t>(store.getCacheOptionAttr().getValue());
    auto addrSpace = ptrType.getAddressSpace();
    if (addrSpace != 1 && addrSpace != 6)
      return rewriter.notifyMatchFailure(store,
                                         "address space must be either 1 or 6");

    if (mask) {
      auto ifOp = rewriter.create<scf::IfOp>(loc, mask, false);

      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      buildStore(rewriter, loc, val, ptr, cacheHint, cacheOption);
      // it seems like with an empty result type, the yield is automatically
      // inserted. Therefore if you manually insert yield, compilation fails

      rewriter.replaceOp(store, ifOp);

    } else {

      rewriter.setInsertionPointAfter(store);

      buildStore(rewriter, loc, val, ptr, cacheHint, cacheOption);

      rewriter.eraseOp(store);
    }

    return success();
  }
};

struct AscendDPXCastOpLowering
    : public ConvertOpToLLVMPattern<ascend_dpx::CastOp> {
  explicit AscendDPXCastOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<ascend_dpx::CastOp>(converter) {}
  LogicalResult
  matchAndRewrite(ascend_dpx::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inType = op.getIn().getType();
    Type outType = op.getOut().getType();

    auto inVecTy = dyn_cast<VectorType>(inType);
    auto outVecTy = dyn_cast<VectorType>(outType);

    if (inVecTy && outVecTy && inVecTy.getShape()[0] == 4 &&
        outVecTy.getShape()[0] == 4 && inVecTy.getElementType().isInteger(8) &&
        outVecTy.getElementType().isBF16() &&
        op.getKind() == ascend_dpx::AscendDPXCastKind::SIGNED_TO_FLOAT) {

      return lowerVector4i8ToBf16(op, adaptor, rewriter);
    }

    return lowerScalarOrGenericCast(op, adaptor, rewriter);
  }

private:
  LogicalResult
  lowerVector4i8ToBf16(ascend_dpx::CastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    Location loc = op.getLoc();
    Value inputVec = adaptor.getIn(); // vector<4xi8>

    Type srcVectorF32Type = VectorType::get({2}, rewriter.getF32Type());
    Type dstVectorBF16Type = VectorType::get({2}, rewriter.getBF16Type());

    auto idx0 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0);
    auto idx1 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
    auto idx2 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 2);
    auto idx3 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 3);

    Value elem0 = rewriter.create<LLVM::ExtractElementOp>(loc, inputVec, idx0);
    Value elem1 = rewriter.create<LLVM::ExtractElementOp>(loc, inputVec, idx1);
    Value elem2 = rewriter.create<LLVM::ExtractElementOp>(loc, inputVec, idx2);
    Value elem3 = rewriter.create<LLVM::ExtractElementOp>(loc, inputVec, idx3);

    Value f32_0 =
        rewriter.create<LLVM::SIToFPOp>(loc, rewriter.getF32Type(), elem0);
    Value f32_1 =
        rewriter.create<LLVM::SIToFPOp>(loc, rewriter.getF32Type(), elem1);

    Value vec0 = rewriter.create<LLVM::UndefOp>(loc, srcVectorF32Type);
    vec0 = rewriter.create<LLVM::InsertElementOp>(loc, srcVectorF32Type, vec0,
                                                  f32_0, idx0);
    vec0 = rewriter.create<LLVM::InsertElementOp>(loc, srcVectorF32Type, vec0,
                                                  f32_1, idx1);

    SmallVector<Value> args0 = {vec0, idx0, idx0};
    auto call0 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, TypeRange{dstVectorBF16Type}, "llvm.hivm.f32x2.to.bf16x2", args0);

    Value f32_2 =
        rewriter.create<LLVM::SIToFPOp>(loc, rewriter.getF32Type(), elem2);
    Value f32_3 =
        rewriter.create<LLVM::SIToFPOp>(loc, rewriter.getF32Type(), elem3);

    Value vec1 = rewriter.create<LLVM::UndefOp>(loc, srcVectorF32Type);
    vec1 = rewriter.create<LLVM::InsertElementOp>(loc, srcVectorF32Type, vec1,
                                                  f32_2, idx0);
    vec1 = rewriter.create<LLVM::InsertElementOp>(loc, srcVectorF32Type, vec1,
                                                  f32_3, idx1);

    SmallVector<Value> args1 = {vec1, idx0, idx0};
    auto call1 = rewriter.create<LLVM::CallIntrinsicOp>(
        loc, TypeRange{dstVectorBF16Type}, "llvm.hivm.f32x2.to.bf16x2", args1);

    Value result0 = call0.getResult(0);
    Value result1 = call1.getResult(0);

    Value bf16_0 = rewriter.create<LLVM::ExtractElementOp>(loc, result0, idx0);
    Value bf16_1 = rewriter.create<LLVM::ExtractElementOp>(loc, result0, idx1);
    Value bf16_2 = rewriter.create<LLVM::ExtractElementOp>(loc, result1, idx0);
    Value bf16_3 = rewriter.create<LLVM::ExtractElementOp>(loc, result1, idx1);

    Type outVecType = VectorType::get({4}, rewriter.getBF16Type());
    Value outVec = rewriter.create<LLVM::UndefOp>(loc, outVecType);
    outVec = rewriter.create<LLVM::InsertElementOp>(loc, outVecType, outVec,
                                                    bf16_0, idx0);
    outVec = rewriter.create<LLVM::InsertElementOp>(loc, outVecType, outVec,
                                                    bf16_1, idx1);
    outVec = rewriter.create<LLVM::InsertElementOp>(loc, outVecType, outVec,
                                                    bf16_2, idx2);
    outVec = rewriter.create<LLVM::InsertElementOp>(loc, outVecType, outVec,
                                                    bf16_3, idx3);

    rewriter.replaceOp(op, outVec);
    return success();
  }

  LogicalResult
  lowerScalarOrGenericCast(ascend_dpx::CastOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    Location loc = op.getLoc();
    Value input = adaptor.getIn();
    Type outType = op.getOut().getType();
    Value result;
    switch (op.getKind()) {
    case ascend_dpx::AscendDPXCastKind::SIGNED_TO_FLOAT:
      result = rewriter.create<LLVM::SIToFPOp>(loc, outType, input);
      break;
    case ascend_dpx::AscendDPXCastKind::UNSIGNED_TO_FLOAT:
      result = rewriter.create<LLVM::UIToFPOp>(loc, outType, input);
      break;
    case ascend_dpx::AscendDPXCastKind::FLOAT_TO_SIGNED:
      result = rewriter.create<LLVM::FPToSIOp>(loc, outType, input);
      break;
    case ascend_dpx::AscendDPXCastKind::FLOAT_TO_UNSIGNED:
      result = rewriter.create<LLVM::FPToUIOp>(loc, outType, input);
      break;
    case ascend_dpx::AscendDPXCastKind::FLOAT_TO_FLOAT: {
      Type inType = input.getType();

      Type V2F32Type = VectorType::get({2}, rewriter.getF32Type());
      Type V2F16Type = VectorType::get({2}, rewriter.getF16Type());
      Type V2BF16Type = VectorType::get({2}, rewriter.getBF16Type());
      auto idx0 =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0);

      result = input;
      SmallVector<Value> args = {input, idx0, idx0};
      if (inType == V2BF16Type) {
        result = rewriter
                     .create<LLVM::CallIntrinsicOp>(loc, TypeRange{V2F32Type},
                                                    "llvm.hivm.bf16x2.to.f32x2",
                                                    args)
                     .getResult(0);
        if (outType == V2F32Type)
          break;
      } else if (inType == V2F16Type && outType == V2F32Type) {
        result = rewriter
                     .create<LLVM::CallIntrinsicOp>(loc, TypeRange{V2F32Type},
                                                    "llvm.hivm.f16x2.to.f32x2",
                                                    args)
                     .getResult(0);
        break;
      } else if (inType == V2F32Type && outType == V2F16Type) {
        result = rewriter
                     .create<LLVM::CallIntrinsicOp>(loc, TypeRange{V2F16Type},
                                                    "llvm.hivm.f32x2.to.f16x2",
                                                    args)
                     .getResult(0);
        break;
      } else if (inType == V2F32Type && outType == V2BF16Type) {
        result = rewriter
                     .create<LLVM::CallIntrinsicOp>(loc, TypeRange{V2BF16Type},
                                                    "llvm.hivm.f32x2.to.bf16x2",
                                                    args)
                     .getResult(0);
        break;
      }

      StringRef opStr;
      auto outVecTy = dyn_cast<mlir::VectorType>(outType);
      auto inVecTy = dyn_cast<mlir::VectorType>(inType);
      if ((!outVecTy) && (!inVecTy)) {
        return rewriter.notifyMatchFailure(op, "dpx.cast matches no type!");
      }
      if (isa<mlir::Float8E4M3FNType>(outVecTy.getElementType())) {
        opStr = "llvm.hivm.f32x2.to.f8e4m3x2";
      } else if (isa<mlir::Float8E5M2Type>(outVecTy.getElementType())) {
        opStr = "llvm.hivm.f32x2.to.f8e5m2x2";
      } else if (isa<mlir::Float8E4M3FNType>(inVecTy.getElementType())) {
        opStr = "llvm.hivm.f8e4m3x2.to.f32x2";
      } else if (isa<mlir::Float8E5M2Type>(inVecTy.getElementType())) {
        opStr = "llvm.hivm.f8e5m2x2.to.f32x2";
      } else {
        return rewriter.notifyMatchFailure(op, "dpx.cast matches no type!");
      }

      result = rewriter
                   .create<LLVM::CallIntrinsicOp>(loc, TypeRange{outType},
                                                  opStr, args)
                   .getResult(0);

      if (outType == V2BF16Type) {
        SmallVector<Value> args = {result, idx0, idx0};
        result = rewriter
                     .create<LLVM::CallIntrinsicOp>(loc, TypeRange{V2BF16Type},
                                                    "llvm.hivm.f32x2.to.bf16x2",
                                                    args)
                     .getResult(0);
      }
      break;
    }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

static bool isV2F16(Type type) {
  auto vType = dyn_cast<VectorType>(type);
  return vType && vType.getElementType().isF16() && vType.getNumElements() == 2;
}

static bool isV2BF16(Type type) {
  auto vType = dyn_cast<VectorType>(type);
  return vType && vType.getElementType().isBF16() &&
         vType.getNumElements() == 2;
}

template <typename ShflOp, typename ShflOpAdaptor, typename ShflI32Op,
          typename ShflI64Op, typename ShflF32Op, typename ShflF16Op,
          typename ShflV2F16Op>
struct AscendDPXShflOpLowering : public ConvertOpToLLVMPattern<ShflOp> {
  explicit AscendDPXShflOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<ShflOp>(converter) {}
  LogicalResult
  matchAndRewrite(ShflOp shfl, ShflOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = shfl.getLoc();
    auto src = shfl.getSrc();
    auto laneMask = shfl.getLaneMask();
    auto clamp = shfl.getClamp();
    auto offset = shfl.getIndex();

    Type type = src.getType();

    // compute control value ctrl
    // ctrl[4:0] = source lane idx/offset
    // ctrl[12:8] = clamp
    // ctrl[20:16] = lane mask

    auto const31 = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0b11111));
    auto const8 = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(8));
    auto const16 = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(16));

    auto cutOffset = rewriter.create<arith::AndIOp>(loc, offset, const31);
    auto cutClamp = rewriter.create<arith::AndIOp>(loc, clamp, const31);
    auto cutLane = rewriter.create<arith::AndIOp>(loc, laneMask, const31);

    auto shiftClamp = rewriter.create<arith::ShLIOp>(loc, cutClamp, const8);
    auto shiftLane = rewriter.create<arith::ShLIOp>(loc, cutLane, const16);

    auto ctrlIntermediate =
        rewriter.create<arith::OrIOp>(loc, cutOffset, shiftClamp);
    auto ctrl = rewriter.create<arith::OrIOp>(loc, ctrlIntermediate, shiftLane);

    if (type.isSignedInteger(32) || type.isSignlessInteger(32)) {
      rewriter.replaceOpWithNewOp<ShflI32Op>(shfl, type, src, ctrl);
    } else if (type.isSignedInteger(64) || type.isSignlessInteger(64)) {
      rewriter.replaceOpWithNewOp<ShflI64Op>(shfl, type, src, ctrl);
    } else if (type.isF32()) {
      rewriter.replaceOpWithNewOp<ShflF32Op>(shfl, type, src, ctrl);
    } else if (type.isF16()) {
      rewriter.replaceOpWithNewOp<ShflF16Op>(shfl, type, src, ctrl);
    } else if (isV2F16(type)) {
      rewriter.replaceOpWithNewOp<ShflV2F16Op>(shfl, type, src, ctrl);
    } else {
      return rewriter.notifyMatchFailure(
          shfl, "source type must be one of i32, i64, f32, f16");
    }

    return success();
  }
};
using AscendDPXShflUpOpLowering =
    AscendDPXShflOpLowering<ascend_dpx::ShflUpOp, ascend_dpx::ShflUpOp::Adaptor,
                            ShflUpOpI32, ShflUpOpI64, ShflUpOpF32, ShflUpOpF16,
                            ShflUpOpV2F16>;
using AscendDPXShflDownOpLowering = AscendDPXShflOpLowering<
    ascend_dpx::ShflDownOp, ascend_dpx::ShflDownOp::Adaptor, ShflDownOpI32,
    ShflDownOpI64, ShflDownOpF32, ShflDownOpF16, ShflDownOpV2F16>;
using AscendDPXShflIndexOpLowering = AscendDPXShflOpLowering<
    ascend_dpx::ShflIndexOp, ascend_dpx::ShflIndexOp::Adaptor, ShflIdxOpI32,
    ShflIdxOpI64, ShflIdxOpF32, ShflIdxOpF16, ShflIdxOpV2F16>;
using AscendDPXShflButterflyOpLowering =
    AscendDPXShflOpLowering<ascend_dpx::ShflButterflyOp,
                            ascend_dpx::ShflButterflyOp::Adaptor, ShflBflyOpI32,
                            ShflBflyOpI64, ShflBflyOpF32, ShflBflyOpF16,
                            ShflBflyOpV2F16>;

template <typename ReduxOp, typename ReduxOpI32, typename ReduxOpF32,
          typename ReduxOpF16>
struct AscendDPXReduxOpLowering : public ConvertOpToLLVMPattern<ReduxOp> {
  explicit AscendDPXReduxOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<ReduxOp>(converter) {}
  LogicalResult
  matchAndRewrite(ReduxOp red, typename ReduxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = red.getSrc();
    Type type = src.getType();
    if (type.isInteger(32))
      rewriter.replaceOpWithNewOp<ReduxOpI32>(red, type, src);
    else if (isa<Float32Type>(type))
      rewriter.replaceOpWithNewOp<ReduxOpF32>(red, type, src);
    else if (isa<Float16Type>(type))
      rewriter.replaceOpWithNewOp<ReduxOpF16>(red, type, src);
    else
      return rewriter.notifyMatchFailure(
          red, "Op type must be one of I32, F32, F16");
    return success();
  }
};
using AscendDPXReduceAddOpLowering = AscendDPXReduxOpLowering<
    ascend_dpx::ReduceAdd, hivm_regbaseintrins::ReduceAddOpS32,
    hivm_regbaseintrins::ReduceAddOpF32, hivm_regbaseintrins::ReduceAddOpF16>;
using AscendDPXReduceMaxOpLowering = AscendDPXReduxOpLowering<
    ascend_dpx::ReduceMax, hivm_regbaseintrins::ReduceMaxOpS32,
    hivm_regbaseintrins::ReduceMaxOpF32, hivm_regbaseintrins::ReduceMaxOpF16>;
using AscendDPXReduceMinOpLowering = AscendDPXReduxOpLowering<
    ascend_dpx::ReduceMin, hivm_regbaseintrins::ReduceMinOpS32,
    hivm_regbaseintrins::ReduceMinOpF32, hivm_regbaseintrins::ReduceMinOpF16>;
using AscendDPXReduceUMaxOpLowering = AscendDPXReduxOpLowering<
    ascend_dpx::ReduceUMax, hivm_regbaseintrins::ReduceMaxOpU32,
    hivm_regbaseintrins::ReduceMaxOpF32, hivm_regbaseintrins::ReduceMaxOpF16>;
using AscendDPXReduceUMinOpLowering = AscendDPXReduxOpLowering<
    ascend_dpx::ReduceUMin, hivm_regbaseintrins::ReduceMinOpU32,
    hivm_regbaseintrins::ReduceMinOpF32, hivm_regbaseintrins::ReduceMinOpF16>;

// Template for lowering logic that just calls for a direct swap in (no args)
// Used for thread_idx_(x|y|z), block_dim_(x|y|z)
template <typename DPX_OP, typename DPX_OP_ADAPTOR, typename INTRIN_OP>
struct DirectConversion : public ConvertOpToLLVMPattern<DPX_OP> {
  explicit DirectConversion(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<DPX_OP>(converter) {}
  LogicalResult
  matchAndRewrite(DPX_OP dpx_op, DPX_OP_ADAPTOR adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = dpx_op.getRes().getType();
    rewriter.replaceOpWithNewOp<INTRIN_OP>(dpx_op, resType);
    return success();
  }
};
using AscendDPXThreadIdxXOpLowering =
    DirectConversion<ascend_dpx::ThreadIdXOp, ascend_dpx::ThreadIdXOp::Adaptor,
                     hivm_regbaseintrins::ThreadIdXOp>;
using AscendDPXThreadIdxYOpLowering =
    DirectConversion<ascend_dpx::ThreadIdYOp, ascend_dpx::ThreadIdYOp::Adaptor,
                     hivm_regbaseintrins::ThreadIdYOp>;
using AscendDPXThreadIdxZOpLowering =
    DirectConversion<ascend_dpx::ThreadIdZOp, ascend_dpx::ThreadIdZOp::Adaptor,
                     hivm_regbaseintrins::ThreadIdZOp>;
using AscendDPXBlockDimXOpLowering =
    DirectConversion<ascend_dpx::BlockDimXOp, ascend_dpx::BlockDimXOp::Adaptor,
                     hivm_regbaseintrins::BlockDimXOp>;
using AscendDPXBlockDimYOpLowering =
    DirectConversion<ascend_dpx::BlockDimYOp, ascend_dpx::BlockDimYOp::Adaptor,
                     hivm_regbaseintrins::BlockDimYOp>;
using AscendDPXBlockDimZOpLowering =
    DirectConversion<ascend_dpx::BlockDimZOp, ascend_dpx::BlockDimZOp::Adaptor,
                     hivm_regbaseintrins::BlockDimZOp>;

// if you are looking for the following conversions, look in AdaptGPUKernel.cpp
// convertGridDPXToArgs
// using AscendDPXBlockIdxXOpLowering =
//     DirectConversion<ascend_dpx::BlockIdxXOp,
//     ascend_dpx::BlockIdxXOp::Adaptor,
//                      hivm_regbaseintrins::BlockIdxXOp>;
// using AscendDPXBlockIdxYOpLowering =
//     DirectConversion<ascend_dpx::BlockIdxYOp,
//     ascend_dpx::BlockIdxYOp::Adaptor,
//                      hivm_regbaseintrins::BlockIdxYOp>;
// using AscendDPXBlockIdxZOpLowering =
//     DirectConversion<ascend_dpx::BlockIdxZOp,
//     ascend_dpx::BlockIdxZOp::Adaptor,
//                      hivm_regbaseintrins::BlockIdxZOp>;
// AscendDPXGridDimXOpLowering
// AscendDPXGridDimYOpLowering
// AscendDPXGridDimZOpLowering

using AscendDPXBlockIdxOpLowering =
    DirectConversion<ascend_dpx::BlockIdxOp, ascend_dpx::BlockIdxOp::Adaptor,
                     hivm_regbaseintrins::BlockIdxOp>;
using AscendDPXClock32OpLowering =
    DirectConversion<ascend_dpx::Clock32Op, ascend_dpx::Clock32Op::Adaptor,
                     hivm_regbaseintrins::Clock32Op>;
using AscendDPXClock64OpLowering =
    DirectConversion<ascend_dpx::Clock64Op, ascend_dpx::Clock64Op::Adaptor,
                     hivm_regbaseintrins::Clock64Op>;

using AscendDPXCoreIdOpLowering =
    DirectConversion<ascend_dpx::CoreIdOp, ascend_dpx::CoreIdOp::Adaptor,
                     hivm_regbaseintrins::CoreIdOp>;

struct AscendDPXSyncThreadsOpLowering
    : public ConvertOpToLLVMPattern<ascend_dpx::SyncThreadsOp> {
  explicit AscendDPXSyncThreadsOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<ascend_dpx::SyncThreadsOp>(converter) {}
  LogicalResult
  matchAndRewrite(ascend_dpx::SyncThreadsOp dpx_op,
                  ascend_dpx::SyncThreadsOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm_regbaseintrins::SyncThreadsOp>(dpx_op);
    return success();
  }
};

template <typename AtomicOp> static std::string getAtomicOpType;
template <> std::string getAtomicOpType<ascend_dpx::AtomicAndOp> = "AND";
template <> std::string getAtomicOpType<ascend_dpx::AtomicOrOp> = "OR";
template <> std::string getAtomicOpType<ascend_dpx::AtomicXorOp> = "XOR";
template <> std::string getAtomicOpType<ascend_dpx::AtomicIncOp> = "INC";
template <> std::string getAtomicOpType<ascend_dpx::AtomicDecOp> = "DEC";
template <> std::string getAtomicOpType<ascend_dpx::AtomicMaxOp> = "MAX";
template <> std::string getAtomicOpType<ascend_dpx::AtomicMinOp> = "MIN";
template <> std::string getAtomicOpType<ascend_dpx::AtomicUMaxOp> = "MAX";
template <> std::string getAtomicOpType<ascend_dpx::AtomicUMinOp> = "MIN";
template <> std::string getAtomicOpType<ascend_dpx::AtomicAddOp> = "ADD";
template <> std::string getAtomicOpType<ascend_dpx::AtomicSubOp> = "SUB";
template <> std::string getAtomicOpType<ascend_dpx::AtomicExchangeOp> = "EXCH";
template <> std::string getAtomicOpType<ascend_dpx::AtomicCASOp> = "CAS";

template <bool unsignedOp>
static std::string atomicOpTypeString(const Type &type) {
  if (type.isInteger()) {
    if constexpr (unsignedOp) {
      return "u" + std::to_string(type.getIntOrFloatBitWidth());
    } else {
      return "s" + std::to_string(type.getIntOrFloatBitWidth());
    }
  } else if (type.isBF16())
    return "bf16";
  else if (isa<FloatType>(
               type)) // for some reason there isn't a general isFloat function
    return "fp" + std::to_string(type.getIntOrFloatBitWidth());
  else if (isV2F16(type))
    return "f16x2";
  else if (isV2BF16(type))
    return "bf16x2";
  else
    return "";
}

static std::string constructAtomicOpString(const std::string &op,
                                           const LLVM::LLVMPointerType &ptrType,
                                           const std::string &dType) {
  std::string addrSpace = ptrType.getAddressSpace() == 1 ? "G" : "S";
  return "llvm.hivm.atom." + op + "." + addrSpace + "." + dType;
}

template <typename AtomicOp>
struct AscendDPXAtomicOpLowering : public ConvertOpToLLVMPattern<AtomicOp> {
  static constexpr bool unsignedOp =
      std::is_same_v<ascend_dpx::AtomicUMaxOp, AtomicOp> ||
      std::is_same_v<ascend_dpx::AtomicUMinOp, AtomicOp>;

  explicit AscendDPXAtomicOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<AtomicOp>(converter) {}

  bool needSoftImpl(AtomicOp op) const {
    auto resultType = op.getResult().getType();
    size_t opNBits;
    if (auto vecTy = mlir::dyn_cast<VectorType>(resultType)) {
      unsigned bitwidth = vecTy.getElementTypeBitWidth();
      int64_t numElements = vecTy.getNumElements();
      opNBits = bitwidth * static_cast<size_t>(numElements);
      return false;
    } else {
      opNBits = resultType.getIntOrFloatBitWidth();
    }

    if (opNBits < 16)
      return true;

    auto operationType = getAtomicOpType<AtomicOp>;
    if (opNBits == 16) {
      if (mlir::dyn_cast<IntegerType>(resultType))
        return true;
      if (mlir::isa<FloatType>(resultType))
        return true;
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(AtomicOp op, typename AtomicOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto data = op.getData();
    auto resultType = op.getResult().getType();
    auto ptrType = src.getType();
    auto operationType = getAtomicOpType<AtomicOp>;

    auto cacheHint = static_cast<uint32_t>(op.getCacheAttr().getValue());
    Value cacheHintConstant = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(cacheHint));

    size_t opNBits;
    if (auto vecTy = mlir::dyn_cast<VectorType>(resultType)) {
      unsigned bitwidth = vecTy.getElementTypeBitWidth();
      int64_t numElements = vecTy.getNumElements();
      opNBits = bitwidth * static_cast<size_t>(numElements);
    } else {
      opNBits = resultType.getIntOrFloatBitWidth();
    }
    auto *context = rewriter.getContext();

    /* software implement for unsupported atomic intrinsics.
      @TODO: Support pack.
      @TODO: optimize software impl if necessary: use shfl to save some
      atomic_cas instrs.
    */
    if (needSoftImpl(op)) {
      Type i8Ty = IntegerType::get(context, 8);
      Type i16Ty = IntegerType::get(context, 16);
      Type i32Ty = IntegerType::get(context, 32);
      Type i64Ty = IntegerType::get(context, 64);

      auto dataType = data.getType();

      Value alignedPtr = src;
      Type casTy, orgTy;
      Value offsetBits, invMask;
      if (opNBits < 32) {
        Value intPtr = rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, src);

        orgTy = opNBits == 8 ? i8Ty : i16Ty;
        casTy = i32Ty;

        // Align ptr to 32bits;
        Value mask = rewriter.create<LLVM::ConstantOp>(
            loc, i64Ty, rewriter.getI64IntegerAttr(3));
        Value offset = rewriter.create<LLVM::AndOp>(loc, intPtr, mask);
        Value wordBaseInt = rewriter.create<LLVM::SubOp>(loc, intPtr, offset);
        alignedPtr =
            rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, wordBaseInt);

        offsetBits = rewriter.create<LLVM::MulOp>(
            loc, offset,
            rewriter.create<LLVM::ConstantOp>(loc, i64Ty,
                                              rewriter.getI64IntegerAttr(8)));
        offsetBits = rewriter.create<LLVM::TruncOp>(loc, i32Ty, offsetBits);

        uint32_t maskBits = opNBits == 8 ? 0xFF : 0xFFFF;
        Value byteMask = rewriter.create<LLVM::ConstantOp>(
            loc, i32Ty, rewriter.getI32IntegerAttr(maskBits));
        Value shiftedMask =
            rewriter.create<LLVM::ShlOp>(loc, byteMask, offsetBits);
        invMask = rewriter.create<LLVM::XOrOp>(
            loc, shiftedMask,
            rewriter.create<LLVM::ConstantOp>(
                loc, i32Ty, rewriter.getI32IntegerAttr(0xFFFFFFFF)));
      } else {
        casTy = opNBits == 32 ? i32Ty : i64Ty;
        orgTy = opNBits == 32 ? i32Ty : i64Ty;
      }

      Value initVal =
          rewriter.create<ascend_dpx::LoadOp>(loc, casTy, alignedPtr);

      auto whileOp = rewriter.create<scf::WhileOp>(loc, casTy, initVal);

      rewriter.setInsertionPointToStart(&whileOp.getBefore().emplaceBlock());
      whileOp.getBefore().addArguments(casTy, loc);
      Value oldVal = whileOp.getBefore().getArgument(0);

      Value opVal, clearedWord;
      if (opNBits < 32) {
        Value currentByteShifted =
            rewriter.create<LLVM::LShrOp>(loc, oldVal, offsetBits);
        opVal = rewriter.create<LLVM::TruncOp>(loc, orgTy, currentByteShifted);
        clearedWord = rewriter.create<LLVM::AndOp>(loc, oldVal, invMask);
      }

      if (orgTy != dataType)
        opVal = rewriter.create<arith::BitcastOp>(loc, dataType, opVal);

      Value opRes;
      Value casCond;
      if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicAndOp>)
        opRes = rewriter.create<LLVM::AndOp>(loc, opVal, data);
      else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicOrOp>)
        opRes = rewriter.create<LLVM::OrOp>(loc, opVal, data);
      else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicXorOp>)
        opRes = rewriter.create<LLVM::XOrOp>(loc, opVal, data);
      else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicIncOp>) {
        auto oneConstant =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, dataType, 1);
        opRes = rewriter.create<LLVM::AddOp>(loc, opVal, oneConstant);
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicDecOp>) {
        auto oneConstant =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, dataType, 1);
        opRes = rewriter.create<LLVM::SubOp>(loc, opVal, oneConstant);
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicMaxOp>) {
        if (mlir::isa<FloatType>(dataType))
          opRes = rewriter.create<arith::MaximumFOp>(loc, opVal, data);

        if (auto intTy = mlir::dyn_cast<IntegerType>(dataType)) {
          if (intTy.isUnsigned()) {
            opRes = rewriter.create<arith::MaxUIOp>(loc, opVal, data);
          } else {
            opRes = rewriter.create<arith::MaxSIOp>(loc, opVal, data);
          }
        }
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicMinOp>) {
        if (mlir::isa<FloatType>(dataType))
          opRes = rewriter.create<arith::MinimumFOp>(loc, opVal, data);

        if (auto intTy = mlir::dyn_cast<IntegerType>(dataType)) {
          if (intTy.isUnsigned()) {
            opRes = rewriter.create<arith::MinUIOp>(loc, opVal, data);
          } else {
            opRes = rewriter.create<arith::MinSIOp>(loc, opVal, data);
          }
        }
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicUMaxOp>) {
        if (mlir::isa<FloatType>(dataType))
          opRes = rewriter.create<arith::MaximumFOp>(loc, opVal, data);

        if (auto intTy = mlir::dyn_cast<IntegerType>(dataType)) {
          opRes = rewriter.create<arith::MaxUIOp>(loc, opVal, data);
        }
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicUMinOp>) {
        if (mlir::isa<FloatType>(dataType))
          opRes = rewriter.create<arith::MinimumFOp>(loc, opVal, data);

        if (auto intTy = mlir::dyn_cast<IntegerType>(dataType)) {
          opRes = rewriter.create<arith::MinUIOp>(loc, opVal, data);
        }
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicAddOp>) {
        if (mlir::isa<FloatType>(dataType))
          opRes = rewriter.create<arith::AddFOp>(loc, opVal, data);
        else
          opRes = rewriter.create<arith::AddIOp>(loc, opVal, data);
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicSubOp>) {
        if (mlir::isa<FloatType>(dataType))
          opRes = rewriter.create<arith::SubFOp>(loc, opVal, data);
        else
          opRes = rewriter.create<arith::SubIOp>(loc, opVal, data);
      } else if constexpr (std::is_same_v<AtomicOp,
                                          ascend_dpx::AtomicExchangeOp>)
        opRes = data;
      else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicCASOp>) {
        // Simulate op: if (opVal == data) { opRes = other; }
        if (mlir::isa<FloatType>(dataType)) {
          data = rewriter.create<arith::BitcastOp>(loc, orgTy, data);
          opVal = rewriter.create<arith::BitcastOp>(loc, orgTy, opVal);
        }
        casCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 opVal, data);
        opRes = op.getOther();
      } else {
        return failure();
      }

      if (orgTy != dataType)
        opRes = rewriter.create<arith::BitcastOp>(loc, orgTy, opRes);

      if (opNBits < 32) {
        Value resExt = rewriter.create<LLVM::ZExtOp>(loc, casTy, opRes);
        Value resShifted =
            rewriter.create<LLVM::ShlOp>(loc, resExt, offsetBits);
        opRes = rewriter.create<LLVM::OrOp>(loc, clearedWord, resShifted);
      }

      Value prevVal;
      auto casTypeStr = atomicOpTypeString<true>(casTy);
      auto casOpString = constructAtomicOpString("CAS", ptrType, casTypeStr);
      SmallVector<Value> args = {alignedPtr, oldVal, opRes, cacheHintConstant};
      if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicCASOp>) {
        auto ifOp = rewriter.create<scf::IfOp>(loc, casTy, casCond, true);

        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto thenRes =
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, casTy, rewriter.getStringAttr(casOpString), args)
                .getResult(0);
        rewriter.create<scf::YieldOp>(loc, thenRes);

        rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
        rewriter.create<scf::YieldOp>(loc, oldVal);

        rewriter.setInsertionPointAfter(ifOp);
        prevVal = ifOp.getResult(0);
      } else {
        prevVal = rewriter
                      .create<LLVM::CallIntrinsicOp>(
                          loc, casTy, rewriter.getStringAttr(casOpString), args)
                      .getResult(0);
      }

      Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  prevVal, oldVal);
      rewriter.create<scf::ConditionOp>(loc, cond, prevVal);

      rewriter.setInsertionPointToStart(&whileOp.getAfter().emplaceBlock());
      whileOp.getAfter().addArguments(casTy, loc);
      Value oldVal_ = whileOp.getAfter().getArgument(0);
      rewriter.create<scf::YieldOp>(loc, oldVal_);

      rewriter.setInsertionPointAfter(whileOp);
      Value result = whileOp.getResult(0);
      if (opNBits < 32) {
        Value currentByteShifted =
            rewriter.create<LLVM::LShrOp>(loc, result, offsetBits);
        result = rewriter.create<LLVM::TruncOp>(loc, orgTy, currentByteShifted);
      }
      if (orgTy != resultType)
        result = rewriter.create<arith::BitcastOp>(loc, resultType, result);

      rewriter.replaceOp(op, result);
      return success();
    }

    if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicCASOp>) {
      auto other = op.getOther();
      if (mlir::isa<FloatType>(resultType)) {
        auto casTy = IntegerType::get(context, opNBits);
        auto dTypeStr = atomicOpTypeString<true>(casTy);
        auto opString =
            constructAtomicOpString(operationType, ptrType, dTypeStr);
        data = rewriter.create<arith::BitcastOp>(loc, casTy, data);
        other = rewriter.create<arith::BitcastOp>(loc, casTy, other);
        SmallVector<Value> args = {src, data, other, cacheHintConstant};
        auto result =
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, casTy, rewriter.getStringAttr(opString), args)
                .getResult(0);
        result = rewriter.create<arith::BitcastOp>(loc, resultType, result);
        rewriter.replaceOp(op, result);
      } else {
        auto dTypeStr = atomicOpTypeString<false>(data.getType());
        auto opString =
            constructAtomicOpString(operationType, ptrType, dTypeStr);
        SmallVector<Value> args = {src, data, other, cacheHintConstant};

        auto result =
            rewriter
                .create<LLVM::CallIntrinsicOp>(
                    loc, resultType, rewriter.getStringAttr(opString), args)
                .getResult(0);
        rewriter.replaceOp(op, result);
      }
    } else {
      auto dTypeStr = atomicOpTypeString<false>(data.getType());

      if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicUMaxOp>) {
        dTypeStr = atomicOpTypeString<true>(data.getType());
      } else if constexpr (std::is_same_v<AtomicOp, ascend_dpx::AtomicUMinOp>) {
        dTypeStr = atomicOpTypeString<true>(data.getType());
      }
      auto opString = constructAtomicOpString(operationType, ptrType, dTypeStr);
      SmallVector<Value> args = {src, data, cacheHintConstant};
      rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
          op, resultType, rewriter.getStringAttr(opString), args);
    }
    return success();
  }
};

using AscendDPXAtomicAndOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicAndOp>;
using AscendDPXAtomicOrOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicOrOp>;
using AscendDPXAtomicXorOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicXorOp>;
using AscendDPXAtomicIncOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicIncOp>;
using AscendDPXAtomicDecOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicDecOp>;
using AscendDPXAtomicMaxOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicMaxOp>;
using AscendDPXAtomicMinOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicMinOp>;
using AscendDPXAtomicUMaxOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicUMaxOp>;
using AscendDPXAtomicUMinOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicUMinOp>;
using AscendDPXAtomicAddOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicAddOp>;
using AscendDPXAtomicSubOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicSubOp>;
using AscendDPXAtomicExchangeOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicExchangeOp>;
using AscendDPXAtomicCASOpLowering =
    AscendDPXAtomicOpLowering<ascend_dpx::AtomicCASOp>;

struct AscendDPXToHIVMRegbaseIntrins
    : public ::impl::ConvertAscendDPXToHIVMRegbaseIntrinsBase<
          AscendDPXToHIVMRegbaseIntrins> {

  /// Set VF target attr to the global entry func so that bisheng works on
  /// the RegBased arch.
  LogicalResult setVFTargetAttr(ModuleOp moduleOp) {
    if (!hacc::utils::isRegBasedArch(moduleOp)) {
      return success();
    }

    auto maybeSpecInterface = hacc::utils::getNPUTargetSpec(moduleOp);
    if (!maybeSpecInterface.has_value()) {
      return failure();
    }
    auto specInterface = maybeSpecInterface.value();
    auto aArch = specInterface.getSpecForIdentifierEnum(hacc::DeviceSpec::ARCH);
    auto archStr = cast<StringAttr>(aArch.getValue()).str();
    // Here we use regbaseintrins's TargetAttr so that
    // HIVMRegbaseIntrinsDialectLLVMIRTranslationInterface would convert
    // the target attr to target-cpu and target-feature in the LLVM IR.
    // In the future, we could use hacc's target attr to replace
    // hivm_regbaseintrins::SIMT_TargetAttr.
    auto targetAttr = hivm_regbaseintrins::SIMT_TargetAttr::get(
        moduleOp.getContext(), archStr);

    moduleOp->walk([&](func::FuncOp funcOp) {
      if (!hacc::utils::isDevice(funcOp))
        return WalkResult::skip();
      funcOp->setAttr(hivm_regbaseintrins::kDavinciTargetAttrName, targetAttr);
      return WalkResult::advance();
    });

    // To be compatible with triton, addrspace 3 is used for shared memory
    // space. Mapping it to UB addrspace 6 here.
    moduleOp.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(result.getType())) {
          if (ptrType.getAddressSpace() == 3) {
            Type newType =
                LLVM::LLVMPointerType::get(moduleOp->getContext(), 6);
            result.setType(newType);
          }
        }
      }

      for (Value operand : op->getOperands()) {
        if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(operand.getType())) {
          if (ptrType.getAddressSpace() == 3) {
            Type newType =
                LLVM::LLVMPointerType::get(moduleOp->getContext(), 6);
            operand.setType(newType);
          }
        }
      }
    });
    return success();
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    LLVMConversionTarget target(ctx);
    ctx.loadDialect<hivm::HIVMDialect>();
    target.addLegalDialect<hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
    target.addLegalDialect<hivm::HIVMDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    RewritePatternSet patterns(&ctx);

    LLVMTypeConverter converter(&ctx);
    converter.addConversion([](LLVM::LLVMPointerType type) {
      return LLVM::LLVMPointerType::get(
          type.getContext(),
          type.getAddressSpace() == NVVM::NVVMMemorySpace::kSharedMemorySpace
              ? (uint32_t)hivm::AddressSpace::UB
              : type.getAddressSpace());
    });
    patterns
        .add<AscendDPXLoadOpLowering, AscendDPXStoreOpLowering,
             AscendDPXBlockDimXOpLowering, AscendDPXBlockDimYOpLowering,
             AscendDPXBlockDimZOpLowering, AscendDPXShflUpOpLowering,
             AscendDPXShflDownOpLowering, AscendDPXShflButterflyOpLowering,
             AscendDPXShflIndexOpLowering, AscendDPXThreadIdxXOpLowering,
             AscendDPXThreadIdxYOpLowering, AscendDPXThreadIdxZOpLowering,
             AscendDPXBlockIdxOpLowering, AscendDPXCoreIdOpLowering,
             AscendDPXClock32OpLowering, AscendDPXClock64OpLowering,
             AscendDPXSyncThreadsOpLowering, AscendDPXAtomicAndOpLowering,
             AscendDPXAtomicOrOpLowering, AscendDPXAtomicXorOpLowering,
             AscendDPXAtomicIncOpLowering, AscendDPXAtomicDecOpLowering,
             AscendDPXAtomicMaxOpLowering, AscendDPXAtomicMinOpLowering,
             AscendDPXAtomicUMaxOpLowering, AscendDPXAtomicUMinOpLowering,
             AscendDPXAtomicAddOpLowering, AscendDPXAtomicSubOpLowering,
             AscendDPXAtomicExchangeOpLowering, AscendDPXAtomicCASOpLowering,
             AscendDPXReduceAddOpLowering, AscendDPXReduceMaxOpLowering,
             AscendDPXReduceMinOpLowering, AscendDPXReduceUMaxOpLowering,
             AscendDPXReduceUMinOpLowering, AscendDPXCastOpLowering>(converter);
    addAscendDPXMathOpsLoweringPatterns(patterns, converter);
    auto moduleOp = getOperation();
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    if (failed(setVFTargetAttr(moduleOp)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createConvertAscendDPXToHIVMRegbaseIntrinPass() {
  return std::make_unique<AscendDPXToHIVMRegbaseIntrins>();
}
