//===--LoadStoreOpToLLVM.cpp - Load/Store Op to LLVM Conversion ---*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {
constexpr unsigned kMaxTransferWidthBits = 128;
constexpr unsigned kMinWordWidthBits = 32;

StringAttr getRegStrAttr(MLIRContext *ctx) { return str_attr("reg"); }

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (a && b) {
    return tb.and_(a, b);
  }
  return a ? a : b;
}

Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const ascend::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred;
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = maybeAnd(rewriter, loc, pred, dimPred);
    }
  }
  return pred;
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

static void emitVectorizationWarning(Operation *op, unsigned vec,
                                     unsigned vecOrig, unsigned numElems,
                                     int maskValue) {
  op->emitRemark() << "Warning: vectorization fails vec = " << vec
                   << " origin vec = " << vecOrig << " numElems = " << numElems
                   << " mask is " << maskValue << "\n";
}

// Contains helper functions for both Load and Store conversions
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const ascend::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    return std::min<unsigned>(kMaxTransferWidthBits / pointeeBitWidth,
                              contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  static unsigned getMaxWordWidth(unsigned valueElemNBits) {
    return std::max<size_t>(kMinWordWidthBits, valueElemNBits);
  }

protected:
  const ascend::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const ascend::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    triton::EvictionPolicy evict = op.getEvict();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    unsigned vecOrig = vec;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
    }

    if (vec == 1 && numElems > 1) {
      int maskValue = !llMask ? -1 : static_cast<int>(getMaskAlignment(mask));
      emitVectorizationWarning(op, vec, vecOrig, numElems, maskValue);
    }

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    uint64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = static_cast<uint64_t>(
          constAttr.getSplatValue<APInt>().getSExtValue());
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());

    // Load redundantly in all dims except reg
    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    auto kReg = getRegStrAttr(ctx);
    uint32_t regMask = static_cast<uint32_t>(freeVarMasks[kReg]);

    auto cachePolicy = [&]() -> ascend_dpx::AscendDPXLoadCachePolicy {
      switch (evict) {
      case EvictionPolicy::NORMAL:
        return ascend_dpx::AscendDPXLoadCachePolicy::L2_CACHE_HINT_NORMAL_FV;
      case EvictionPolicy::EVICT_FIRST:
        return ascend_dpx::AscendDPXLoadCachePolicy::L2_CACHE_HINT_NORMAL_FV;
      case EvictionPolicy::EVICT_LAST:
        return ascend_dpx::AscendDPXLoadCachePolicy::L2_CACHE_HINT_NORMAL_LV;
      }
      llvm_unreachable("switch on EvictionPolicy is not exhaustive");
    }();

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (auto canonicalVecStart = getCanonicalIndex(vecStart, regMask);
          vecStart != canonicalVecStart) {
        // For redundant registers, refer back to the canonical load
        for (unsigned iVec = 0; iVec < vec; ++iVec) {
          loadedVals.push_back(loadedVals[canonicalVecStart + iVec]);
        }
        continue;
      }

      Type retVecTy = LLVM::getVectorType(valueElemTy, vec);

      Value falseVal;
      if (other) {
        Value v = b.undef(retVecTy);
        for (size_t s = 0; s < vec; ++s) {
          Value elem = otherIsSplatConstInt
                           ? b.int_val(valueElemNBits, splatVal)
                           : otherElems[vecStart + s];
          elem = b.bitcast(elem, valueElemTy);
          Value sVal = createIndexAttrConstant(
            rewriter, loc, typeConverter->getIndexType(), s);
          v = b.insert_element(retVecTy, v, elem, sVal);
        }
        falseVal = v;
      }
      Value maskVal = mask ? maskElems[vecStart] : Value();

      Value loadResult = rewriter.create<ascend_dpx::LoadOp>(
        loc, retVecTy, ptrElems[vecStart], maskVal, falseVal, cachePolicy);
      
      if (vec == 1) {
        Value loaded = b.bitcast(loadResult, valueElemTy);
        loadedVals.push_back(loaded);
      } else {
        for (unsigned i = 0; i < vec; i++) {
          auto idx = b.i32_val(i);
          auto elem = b.extract_element(loadResult, idx);
          loadedVals.push_back(elem);
        }
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const ascend::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    unsigned vecOrig = vec;
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    if (vec == 1 && elemsPerThread > 1) {
      int mask =
          !llMask ? -1 : static_cast<int>(getMaskAlignment(op.getMask()));
      emitVectorizationWarning(op, vec, vecOrig, elemsPerThread, mask);
    }

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto kReg = getRegStrAttr(ctx);
    uint32_t regMask = static_cast<uint32_t>(freeVarMasks[kReg]);

    triton::EvictionPolicy evict = op.getEvict();
    auto cachePolicy = [&]() -> ascend_dpx::AscendDPXStoreCachePolicy {
      switch (evict) {
      case EvictionPolicy::NORMAL:
        return ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_FV;
      case EvictionPolicy::EVICT_FIRST:
        return ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_FV;
      case EvictionPolicy::EVICT_LAST:
        return ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_LV;
      }
      llvm_unreachable("switch on EvictionPolicy is not exhaustive");
    }();

    const int numVecs = static_cast<int>(elemsPerThread / vec);
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if ((vecStart & regMask) != 0) {
        continue;
      }

      const size_t maxWordWidth = getMaxWordWidth(valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      Value pred = threadPred;
      if (llMask) {
        auto mask = maskElems[vecStart];
        pred = maybeAnd(rewriter, loc, pred, mask);
      }

      Type storeValueTy =
          VectorType::get({static_cast<long>(nWords)}, valArgTy);
      Value toBeStored = b.undef(storeValueTy);

      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        Value llWord = b.undef(wordTy);
        if (wordNElems == 1) {
          Value elem = valueElems[vecStart + wordIdx];
          elem = b.bitcast(elem, valueElemTy);
          llWord = b.bitcast(elem, valArgTy);
        } else {
          for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
            const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
            assert(elemOffset < valueElems.size());
            Value elem = valueElems[elemOffset];
            if (elem.getType().isInteger(1))
              elem = b.sext(i8_ty, elem);
            elem = b.bitcast(elem, valueElemTy);

            llWord = b.insert_element(wordTy, llWord, elem, b.i32_val(elemIdx));
          }
          llWord = b.bitcast(llWord, valArgTy);
        }
        toBeStored = b.insert_element(storeValueTy, toBeStored, llWord,
                                      b.i32_val(wordIdx));
      }

      rewriter.create<ascend_dpx::StoreOp>(loc, ptrElems[vecStart], toBeStored,
                                           pred, cachePolicy);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {

  AtomicCASOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto cmpElems = unpackLLElements(loc, adaptor.getCmp(), rewriter);
    auto valElems = unpackLLElements(loc, adaptor.getVal(), rewriter);

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        getTypeConverter()->convertType(tensorTy.getElementType());

    unsigned elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i++) {
      Value casPtr = ptrElems[i];
      Value casCmp = cmpElems[i];
      Value casVal = valElems[i];

      auto casOp = rewriter.create<mlir::ascend_dpx::AtomicCASOp>(
          loc, valueElemTy, casPtr, casCmp, casVal);

      resultVals[i] = casOp.getRes();
    }

    Value result =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, tensorTy);
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {

  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const ascend::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto ptrElems = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto valElems = unpackLLElements(loc, adaptor.getVal(), rewriter);

    SmallVector<Value> maskElems;
    if (adaptor.getMask())
      maskElems = unpackLLElements(loc, adaptor.getMask(), rewriter);

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        getTypeConverter()->convertType(tensorTy.getElementType());

    unsigned elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    // Compute warp/lane/block guard: only threads that hold unique data execute
    // the atomic. This matches the warp guard emitted by the remap path.
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    auto kReg = getRegStrAttr(ctx);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = static_cast<uint32_t>(freeVarMasks[kReg]);

    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i++) {
      if (auto canonicalStart = getCanonicalIndex(i, regMask);
          canonicalStart != i) {
        resultVals[i] = resultVals[canonicalStart];
        continue;
      }

      Value ptr = ptrElems[i];
      Value val = valElems[i];
      Value pred = !maskElems.empty()
                       ? maybeAnd(rewriter, loc, threadPred, maskElems[i])
                       : threadPred;

      Value result;
      if (pred) {
        // Wrap the atomic in a conditional branch so that redundant threads
        // (warp guard = false) do not execute it at all.
        Value undefVal = b.undef(valueElemTy);
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock =
            curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));
        endBlock->addArgument({valueElemTy}, {loc});

        rewriter.setInsertionPointToEnd(curBlock);
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock,
                                        undefVal);

        rewriter.setInsertionPointToEnd(atomicBlock);
        auto atomOrFailure = emitAtomicOp(rewriter, loc, valueElemTy,
                                          op.getAtomicRmwOp(), ptr, val);
        if (failed(atomOrFailure))
          return op.emitError("unhandled atomic operation");
        rewriter.create<LLVM::BrOp>(loc, *atomOrFailure, endBlock);

        rewriter.setInsertionPointToStart(endBlock);
        result = endBlock->getArgument(0);
      } else {
        auto atomOrFailure = emitAtomicOp(rewriter, loc, valueElemTy,
                                          op.getAtomicRmwOp(), ptr, val);
        if (failed(atomOrFailure))
          return op.emitError("unhandled atomic operation");
        result = *atomOrFailure;
      }
      resultVals[i] = result;
    }

    Value result =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, tensorTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  FailureOr<Value> emitAtomicOp(ConversionPatternRewriter &rewriter,
                                Location loc, Type valueElemTy,
                                triton::RMWOp rmwOp, Value ptr,
                                Value val) const {
    switch (rmwOp) {
    case triton::RMWOp::AND:
      return rewriter.create<ascend_dpx::AtomicAndOp>(loc, valueElemTy, ptr,
                                                      val)
          .getRes();
    case triton::RMWOp::OR:
      return rewriter
          .create<ascend_dpx::AtomicOrOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::XOR:
      return rewriter
          .create<ascend_dpx::AtomicXorOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::ADD:
    case triton::RMWOp::FADD:
      return rewriter
          .create<ascend_dpx::AtomicAddOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::MAX:
      return rewriter
          .create<ascend_dpx::AtomicMaxOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::MIN:
      return rewriter
          .create<ascend_dpx::AtomicMinOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::UMAX:
      return rewriter
          .create<ascend_dpx::AtomicUMaxOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::UMIN:
      return rewriter
          .create<ascend_dpx::AtomicUMinOp>(loc, valueElemTy, ptr, val)
          .getRes();
    case triton::RMWOp::XCHG:
      return rewriter
          .create<ascend_dpx::AtomicExchangeOp>(loc, valueElemTy, ptr, val)
          .getRes();
    }
  }

  const ascend::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::ascend {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion>(typeConverter, targetInfo,
                                                    axisInfoAnalysis, benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::ascend
