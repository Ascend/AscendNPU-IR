//===-- FMADotUtility.cpp - K-outer FMA microkernel with lazy extraction --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/FMADotUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir::triton::ascend {

LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier) {
  auto loc = op.getLoc();

  auto A = op.getA();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  SmallVector<int64_t> aShapePerCTA =
      gpu::expandMatrixShapeWithBatch(ArrayRef(gpu::getShapePerCTA(aTensorTy)));

  // Accumulator: fully unpacked upfront — must remain live across all k steps.
  SmallVector<Value> acc = unpackLLElements(loc, adaptor.getC(), rewriter);

  // A and B kept as raw LLVM struct values. Elements are extracted lazily
  // inside the k-outer loop so each extracted SSA value is live for only one
  // k-step, keeping register pressure proportional to mTile + nTile + acc-tile
  // rather than totalM*K + totalN*K.
  Value aStruct = adaptor.getA();
  Value bStruct = adaptor.getB();

  // Determine per-thread element counts from the LLVM struct sizes.
  // (Same ordinal count as unpackLLElements would return.)
  unsigned totalAElems = 1, totalBElems = 1;
  if (auto sTy = dyn_cast<LLVM::LLVMStructType>(aStruct.getType()))
    totalAElems = static_cast<unsigned>(sTy.getBody().size());
  if (auto sTy = dyn_cast<LLVM::LLVMStructType>(bStruct.getType()))
    totalBElems = static_cast<unsigned>(sTy.getBody().size());

  const unsigned K = static_cast<unsigned>(aShapePerCTA[2]);

  if (K == 0 || totalAElems % K != 0) {
    (void)rewriter.notifyMatchFailure(op, "A element count must be a multiple of K");
    return failure();
  }
  if (K == 0 || totalBElems % K != 0) {
    (void)rewriter.notifyMatchFailure(op, "B element count must be a multiple of K");
    return failure();
  }

  const unsigned totalM = totalAElems / K;
  const unsigned totalN = totalBElems / K;
  const unsigned totalB =
      (totalM * totalN > 0)
          ? static_cast<unsigned>(acc.size()) / (totalM * totalN)
          : 1;

  if (acc.size() != totalB * totalM * totalN) {
    (void)rewriter.notifyMatchFailure(
        op, "FMA layout mismatch: acc.size() != totalB * totalM * totalN. "
            "The linear layouts for A/B are inconsistent with D's blocked layout.");
    return failure();
  }

  Type accTy = acc[0].getType();

  // Compute tile sizes (nTile, mTile) to bound simultaneously-live values.
  //
  // At the innermost point the live set is:
  //   aSlice: mTile  values  (promoted A elements for current k, M-strip)
  //   bSlice: nTile  values  (promoted B elements for current k, N-strip)
  //   acc:    mTile * nTile  values  (accumulator tile)
  // Target: mTile + nTile + mTile*nTile <= budget
  //
  // kTotalCTARegs: total 32-bit register words in the Ascend CTA register file.
  // Observation: 32 warps * 32 threads = 1024 threads -> 32 regs/thread
  //   => 1024 * 32 = 32768 total registers.
  // Budget = registers/thread + DCache spill allowance/thread.
  // DCache = 256KB - 8KB(OS) - max(128KB, bishengir.shared-mem-dynamic-size),
  //          clamped to [32KB, 120KB], shared across all threads in the CTA.
  constexpr unsigned kTotalCTARegs = 32768;
  unsigned nTile = totalN;
  unsigned mTile = totalM;
  if (auto mod = op->getParentOfType<ModuleOp>()) {
    int numWarps = gpu::lookupNumWarps(op.getOperation());
    int tpw = gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (numWarps > 0 && tpw > 0) {
      unsigned threadsPerCTA =
          static_cast<unsigned>(numWarps) * static_cast<unsigned>(tpw);
      unsigned regsPerThread = kTotalCTARegs / threadsPerCTA;

      unsigned sharedMem = 0;
      if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
              "bishengir.shared-mem-dynamic-size"))
        sharedMem = static_cast<unsigned>(attr.getInt());

      // Allow per-kernel dcache budget override for tuning.
      unsigned dcacheKB;
      if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
              "bishengir.fma-dcache-budget-kb")) {
        dcacheKB = static_cast<unsigned>(attr.getInt());
      } else {
        constexpr unsigned kTotalKB = 256, kOsKB = 8, kMinSharedKB = 128;
        constexpr unsigned kDCacheMinKB = 32, kDCacheMaxKB = 120;
        unsigned sharedKB = (sharedMem + 1023u) / 1024u;
        dcacheKB = kTotalKB - kOsKB - std::max(kMinSharedKB, sharedKB);
        dcacheKB = std::max(kDCacheMinKB, std::min(kDCacheMaxKB, dcacheKB));
      }
      unsigned dcacheF32PerThread = (dcacheKB * 1024u) / (4u * threadsPerCTA);

      unsigned rawBudget = regsPerThread + dcacheF32PerThread;
      // Apply headroom factor (default 100%) to reserve registers for loop
      // induction variables, address registers, and other non-FMA live values.
      unsigned headroomPct = 100u;
      if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
              "bishengir.fma-budget-headroom-pct"))
        headroomPct = static_cast<unsigned>(attr.getInt());
      unsigned budget = (rawBudget * headroomPct) / 100u;

      // Solve nTile first (N-dimension), then mTile (M-dimension).
      // Constraint: mTile + nTile + mTile*nTile <= budget
      if (budget > totalM && totalM * totalN + totalM + totalN > budget) {
        unsigned maxN = (budget - totalM) / (totalM + 1);
        unsigned floorP2 = (maxN > 0) ? (1u << llvm::Log2_32(maxN)) : 1u;
        nTile = std::min(totalN, std::max(1u, floorP2));
      }
      // With nTile fixed, solve for mTile.
      if (budget > nTile) {
        unsigned maxM = (budget - nTile) / (nTile + 1);
        unsigned floorP2M = (maxM > 0) ? (1u << llvm::Log2_32(maxM)) : 1u;
        mTile = std::min(totalM, std::max(1u, floorP2M));
      }
    }
  }

  // ── K-outer microkernel with lazy element extraction ──────────────────────
  //
  // Loop order: batch → k → M-tile → N-tile → m → n
  //
  // Ordinal conventions (must match layouts from ConvertDotInputToLinearLayout):
  //   A is K-innermost: ordinal(bi, mi, k) = (bi*totalM + mi)*K + k
  //   B is N-innermost: ordinal(bi, k, ni) = (bi*K + k)*totalN + ni
  //
  // A ordinal has no niBase dependence, so A is extracted once per (bi, k,
  // miBase) and reused across all N-tile iterations — eliminating redundant
  // ExtractValue+FPExt pairs. B ordinal has no miBase dependence, so B is
  // re-extracted per (k, miBase) iteration (same trade-off N-tiling makes).
  //
  for (unsigned bi = 0; bi < totalB; ++bi) {
    for (unsigned k = 0; k < K; ++k) {
      for (unsigned miBase = 0; miBase < totalM; miBase += mTile) {
        const unsigned miEnd = std::min(miBase + mTile, totalM);

        // Lazily extract and promote A[bi, mi, k] for mi in [miBase, miEnd).
        // Live across all N-tile iterations at this (bi, k, miBase).
        SmallVector<Value, 4> aSlice;
        aSlice.reserve(miEnd - miBase);
        for (unsigned mi = miBase; mi < miEnd; ++mi) {
          unsigned aOrdinal = (bi * totalM + mi) * K + k;
          Value raw =
              rewriter.create<LLVM::ExtractValueOp>(loc, aStruct, aOrdinal);
          aSlice.push_back(multiplier.promoteToAccType(raw, accTy));
        }

        for (unsigned niBase = 0; niBase < totalN; niBase += nTile) {
          const unsigned niEnd = std::min(niBase + nTile, totalN);

          // Lazily extract and promote B[bi, k, ni] for ni in [niBase, niEnd).
          SmallVector<Value, 16> bSlice;
          bSlice.reserve(niEnd - niBase);
          for (unsigned ni = niBase; ni < niEnd; ++ni) {
            unsigned bOrdinal = (bi * K + k) * totalN + ni;
            Value raw =
                rewriter.create<LLVM::ExtractValueOp>(loc, bStruct, bOrdinal);
            bSlice.push_back(multiplier.promoteToAccType(raw, accTy));
          }

          // Emit mTile * nTile FMAs.  Only aSlice + bSlice + acc-tile live.
          for (unsigned mi = miBase; mi < miEnd; ++mi) {
            for (unsigned ni = niBase; ni < niEnd; ++ni) {
              unsigned accIdx = bi * totalM * totalN + mi * totalN + ni;
              acc[accIdx] = multiplier.emitSingleFMA(aSlice[mi - miBase],
                                                     bSlice[ni - niBase],
                                                     acc[accIdx]);
            }
          }
          // bSlice is dead here — LLVM recycles its registers.
        }
        // aSlice is dead here — LLVM recycles its registers.
      }
    }
  }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);
  return success();
}

} // namespace mlir::triton::ascend
