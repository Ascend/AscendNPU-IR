//===--TargetInfo.cpp - TritonAscendGPU Target Info -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
constexpr unsigned kMinElementWidthBits = 8;
constexpr unsigned kMinWordWidthBits = 32;
constexpr unsigned kMaxTransferWidthBits = 128;
constexpr unsigned kMaxVectorSize = 4;
constexpr int kSharedMemoryAddressSpace =
    static_cast<int>(ascend_dpx::AscendDPXAddressSpace::SHARED_MEM);
constexpr int64_t kConstantTruePredValue = -1;

// Helper function to check if predicate is constant true
static bool isConstantTruePred(Value pred) {
  if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>()) {
    return cast<IntegerAttr>(constOp.getValue()).getInt() ==
           kConstantTruePredValue;
  }
  return false;
}

} // namespace

namespace mlir::triton::ascend {

template <typename ShflOpType>
static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl<ShflOpType>(loc, rewriter, val0, i, clamp);
    val1 = shuffleCommonImpl<ShflOpType>(loc, rewriter, val1, i, clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  Type type = val.getType();
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }

  // all lanes in one group, no sub-warp splitting
  Value lanemask = b.i32_val(0);
  Value result =
      rewriter.create<ShflOpType>(loc, i32_ty, val, lanemask, clamp, i);

  if (type != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, type);
  }
  return result;
}

template <typename ShflOpType>
static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl<ShflOpType>(loc, rewriter, val, i, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  auto int32Ty = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                           rewriter.getI32IntegerAttr(0));
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == kSharedMemoryAddressSpace &&
         "Invalid addr space for store_dsmem");

  if (!isa<VectorType>(val.getType())) {
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, {val}, rewriter),
                 pred);
    return;
  }

  auto vecTy = cast<VectorType>(val.getType());
  Type elemTy = vecTy.getElementType();
  unsigned int vec = static_cast<unsigned int>(vecTy.getNumElements());
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < kMinElementWidthBits) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = b.zext(int_ty(kMinElementWidthBits),
                 b.bitcast(v, int_ty(elemBitwidth)));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  if (!elemTy.isInteger()) {
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = b.bitcast(v, int_ty(elemBitwidth));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  // If vec > 4 and elemBitwidth < 32, pack into b32's
  if (vec > kMaxVectorSize && elemBitwidth < kMinWordWidthBits) {
    assert(llvm::isPowerOf2_32(vec));
    unsigned int elemsPerPack = kMinWordWidthBits / elemBitwidth;
    SmallVector<Value> oldVals = unpackLLVector(loc, val, rewriter);

    SmallVector<Value> newVals;
    for (unsigned int i = 0; i < vec / elemsPerPack; i++) {
      Value v = packLLVector(
          loc, ArrayRef(oldVals).slice(i * elemsPerPack, elemsPerPack),
          rewriter);
      newVals.push_back(b.bitcast(v, i32_ty));
    }
    storeDShared(rewriter, loc, ptr, ctaId,
                 packLLVector(loc, newVals, rewriter), pred);
    return;
  }

  // If total width > 128, split into multiple stores
  if (vec * elemBitwidth > kMaxTransferWidthBits) {
    assert(llvm::isPowerOf2_32(vec));
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    unsigned int maxVec = kMaxTransferWidthBits / elemBitwidth;

    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (unsigned int i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec));
      storeDShared(
          rewriter, loc, newPtr, ctaId,
          packLLVector(loc, ArrayRef(vals).slice(i * maxVec, maxVec), rewriter),
          pred);
    }
    return;
  }

  assert(elemBitwidth >= kMinElementWidthBits);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= kMaxVectorSize);
  assert(vec * elemBitwidth <= kMaxTransferWidthBits);

  Value maskToUse = (pred && !isConstantTruePred(pred)) ? pred : Value();

  rewriter.create<ascend_dpx::StoreOp>(
      loc, ptr, val, maskToUse,
      ascend_dpx::AscendDPXStoreCachePolicy::L2_CACHE_HINT_NORMAL_FV);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type loadTy,
                              Value pred, Operation *localLoadOp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == kSharedMemoryAddressSpace &&
         "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(loadTy)) {
    SmallVector<Value> values = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, vec_ty(loadTy, 1), pred),
        rewriter);
    assert(values.size() == 1);
    return values[0];
  }

  auto vecTy = cast<VectorType>(loadTy);
  Type elemTy = vecTy.getElementType();
  auto vec = vecTy.getNumElements();
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < kMinElementWidthBits) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals =
        unpackLLVector(loc,
                       loadDShared(rewriter, loc, ptr, ctaId,
                                   int_ty(kMinElementWidthBits), pred),
                       rewriter);
    assert(vals.size() == 1);
    return b.bitcast(b.trunc(int_ty(elemBitwidth), vals[0]), elemTy);
  }

  // We only know how to load integers
  if (!elemTy.isInteger()) {
    Type newLoadTy = vec_ty(int_ty(elemBitwidth), vec);
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, newLoadTy, pred), rewriter);
    for (Value &v : vals) {
      v = b.bitcast(v, elemTy);
    }
    return packLLVector(loc, vals, rewriter);
  }

  // If vec > 4 and elemBitwidth < 32, load b32's instead
  if (vec > kMaxVectorSize && elemBitwidth < kMinWordWidthBits) {
    auto newVec = vec / (int64_t)(kMinWordWidthBits / elemBitwidth);
    auto newVecTy = vec_ty(i32_ty, newVec);
    auto res = loadDShared(rewriter, loc, ptr, ctaId, newVecTy, pred);

    // Unpack the b32's into the original vector type
    SmallVector<Value> vals;
    for (Value v : unpackLLVector(loc, res, rewriter)) {
      Value vv = b.bitcast(v, vec_ty(elemTy, kMinWordWidthBits / elemBitwidth));
      for (Value vvv : unpackLLVector(loc, vv, rewriter)) {
        vals.push_back(vvv);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  // If total width > 128, split into multiple loads
  if (static_cast<unsigned int>(vec) * elemBitwidth > kMaxTransferWidthBits) {
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    assert(llvm::isPowerOf2_32(vec));
    unsigned int maxVec = kMaxTransferWidthBits / elemBitwidth;

    SmallVector<Value> vals;
    for (unsigned int i = 0; i < vec / (int)maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec));
      auto newVal = loadDShared(rewriter, loc, newPtr, ctaId,
                                vec_ty(elemTy, maxVec), pred);
      for (Value v : unpackLLVector(loc, newVal, rewriter)) {
        vals.push_back(v);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  assert(elemBitwidth >= kMinElementWidthBits);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= kMaxVectorSize);
  assert(static_cast<unsigned int>(vec) * elemBitwidth <= kMaxTransferWidthBits);

  Type loadResTy =
      vec > 1 ? VectorType::get({static_cast<long>(vec)}, elemTy) : elemTy;

  Value falseVal;
  if (!isConstantTruePred(pred)) {
    Value zeroVal = b.int_val(elemBitwidth, 0);
    zeroVal = b.bitcast(zeroVal, elemTy);

    if (vec == 1) {
      falseVal = zeroVal;
    } else {
      Type yieldedVecTy = LLVM::getVectorType(elemTy, vec);
      falseVal = b.undef(yieldedVecTy);
      for (int64_t ii = 0; ii < vec; ++ii) {
        Value idx = b.i32_val(ii);
        falseVal = b.insert_element(yieldedVecTy, falseVal, zeroVal, idx);
      }
    }
  }

  Value maskToUse = isConstantTruePred(pred) ? Value() : pred;

  Value loadResult = rewriter.create<ascend_dpx::LoadOp>(
      loc, loadResTy, ptr, maskToUse, falseVal,
      ascend_dpx::AscendDPXLoadCachePolicy::L2_CACHE_HINT_NORMAL_FV);

  return loadResult;
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon<ascend_dpx::ShflButterflyOp>(
      loc, rewriter, val, b.i32_val(i), b.i32_val(0x1f));
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon<ascend_dpx::ShflUpOp>(loc, rewriter, val, b.i32_val(i),
                                             b.i32_val(0x0));
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon<ascend_dpx::ShflIndexOp>(loc, rewriter, val,
                                                b.i32_val(i), b.i32_val(0x1f));
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon<ascend_dpx::ShflIndexOp>(loc, rewriter, val, i,
                                                b.i32_val(0x1f));
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  rewriter.create<ascend_dpx::SyncThreadsOp>(loc);
}

} // namespace mlir::triton::ascend
