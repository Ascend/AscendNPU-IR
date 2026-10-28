//===- SharedMemorySwizzle.cpp - Ascend scratch swizzles
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/TritonAscendGPUToLLVM/SharedMemorySwizzle.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::ascend {
namespace {

// Match the register permutations and vector width used by lowerLdStShared.
// Repetition bits do not contribute to the address of the reused scratch tile.
std::optional<std::pair<LinearLayout, unsigned>>
getScratchTransfer(const LinearLayout &layout, const LinearLayout &smem,
                   int32_t bitwidth) {
  auto *ctx = layout.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kReps = StringAttr::get(ctx, "reps");
  auto kOffset = StringAttr::get(ctx, "offset");
  auto cvt = layout.invertAndCompose(smem);
  if (smem.hasInDim(kReps) && smem.getInDimSize(kReps) > 1) {
    auto reps = LinearLayout::identity1D(smem.getInDimSize(kReps), kReg, kReps);
    auto permutation = regPermForDivide(cvt, reps, /*left=*/false);
    if (!permutation)
      return std::nullopt;
    auto quotient = divideRight(permutation->apply(cvt), reps);
    if (!quotient)
      return std::nullopt;
    cvt = *quotient;
  }
  cvt = cvt.reshapeOuts({{kOffset, cvt.getTotalOutDimSize()}});
  for (unsigned vec = 128 / bitwidth; vec >= 1; vec /= 2) {
    auto tile = LinearLayout::identity1D(vec, kReg, kOffset);
    auto permutation = regPermForDivide(cvt, tile, /*left=*/true);
    if (!permutation)
      continue;
    auto permuted = permutation->apply(cvt);
    if (divideLeft(permuted, tile))
      return std::make_pair(permuted, vec);
  }
  return std::nullopt;
}

unsigned getBinaryRank(ArrayRef<uint32_t> vectors) {
  std::array<uint32_t, 32> pivots{};
  unsigned rank = 0;
  for (uint32_t value : vectors) {
    while (value) {
      unsigned bit = llvm::Log2_32(value);
      if (pivots[bit]) {
        value ^= pivots[bit];
      } else {
        pivots[bit] = value;
        ++rank;
        break;
      }
    }
  }
  return rank;
}

int getBankConflictLog2(const LinearLayout &transfer, unsigned vec,
                        int32_t bitwidth,
                        const SharedMemoryBankInfo &bankInfo) {
  assert(llvm::isPowerOf2_32(bankInfo.numBanks) &&
         llvm::isPowerOf2_32(bankInfo.bankWidthBits) &&
         llvm::isPowerOf2_32(bitwidth) &&
         static_cast<unsigned>(bitwidth) <= bankInfo.bankWidthBits &&
         "bank words must contain a power-of-two number of elements");
  auto *ctx = transfer.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kOffset = *transfer.getOutDimNames().begin();
  unsigned wordShift = llvm::Log2_32(bankInfo.bankWidthBits / bitwidth);
  unsigned transactionBits = bankInfo.numBanks * bankInfo.bankWidthBits;
  unsigned numLanes = transfer.getInDimSize(kLane);
  unsigned phases = std::max(1u, numLanes * vec * bitwidth / transactionBits);
  assert(llvm::isPowerOf2_32(phases) && phases <= numLanes &&
         "a service transaction must accommodate a thread's vector");
  unsigned laneBits = llvm::Log2_32(numLanes / phases);
  SmallVector<uint32_t> wordBases, bankBases;
  auto addBasis = [&](uint32_t basis) {
    uint32_t word = basis >> wordShift;
    wordBases.push_back(word);
    bankBases.push_back(word & (bankInfo.numBanks - 1));
  };
  for (unsigned i = 0; i < llvm::Log2_32(vec); ++i)
    addBasis(transfer.getBasis(kReg, i, kOffset));
  for (unsigned i = 0; i < laneBits; ++i)
    addBasis(transfer.getBasis(kLane, i, kOffset));

  // The bank projection's kernel counts distinct aligned words per resource.
  // Taking rank after the word shift merges subword accesses and broadcasts.
  return getBinaryRank(wordBases) - getBinaryRank(bankBases);
}

} // namespace

std::pair<int, int> getLog2BankConflicts(const LinearLayout &src,
                                         const LinearLayout &dst,
                                         const LinearLayout &smem,
                                         int32_t bitwidth,
                                         const SharedMemoryBankInfo &bankInfo) {
  auto store = getScratchTransfer(src, smem, bitwidth);
  auto load = getScratchTransfer(dst, smem, bitwidth);
  if (!store || !load)
    llvm::report_fatal_error("unsupported scratch transfer in bank cost model");
  return {getBankConflictLog2(load->first, load->second, bitwidth, bankInfo),
          getBankConflictLog2(store->first, store->second, bitwidth, bankInfo)};
}

LinearLayout refineSharedMemorySwizzle(const LinearLayout &src,
                                       const LinearLayout &dst,
                                       const LinearLayout &baseline,
                                       int32_t bitwidth,
                                       const SharedMemoryBankInfo &bankInfo) {
  if (bitwidth != 32)
    return baseline;
  auto best = baseline;
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kVec = StringAttr::get(ctx, "vector");
  auto kBank = StringAttr::get(ctx, "bank");

  // Refine the case where the low bank basis supplies the second element of
  // a bank word (32-bit elements in an 8-byte word on Ascend). Higher bank-bit
  // permutations only rename resources and cannot affect conflicts. Other
  // word partitions keep the existing planner.
  if (bankInfo.bankWidthBits / bitwidth != 2u * best.getInDimSize(kVec))
    return best;
  auto store = getScratchTransfer(src, best, bitwidth);
  auto load = getScratchTransfer(dst, best, bitwidth);
  if (!store || !load)
    return best;
  auto bestConflicts = std::make_pair(
      getBankConflictLog2(load->first, load->second, bitwidth, bankInfo),
      getBankConflictLog2(store->first, store->second, bitwidth, bankInfo));
  const auto base = best;
  unsigned bankBits = base.getInDimSizeLog2(kBank);
  for (unsigned lowWordBasis = 1; lowWordBasis < bankBits; ++lowWordBasis) {
    SmallVector<size_t> order;
    for (unsigned i = 0; i < bankBits; ++i)
      order.push_back(i);
    std::swap(order[0], order[lowWordBasis]);
    auto candidate = ColumnAction(order, kBank, bankBits).apply(base);
    auto candidateStore = getScratchTransfer(src, candidate, bitwidth);
    auto candidateLoad = getScratchTransfer(dst, candidate, bitwidth);
    if (!candidateStore || !candidateLoad ||
        candidateStore->second != store->second ||
        candidateLoad->second != load->second)
      continue;
    auto conflicts = std::make_pair(
        getBankConflictLog2(candidateLoad->first, candidateLoad->second,
                            bitwidth, bankInfo),
        getBankConflictLog2(candidateStore->first, candidateStore->second,
                            bitwidth, bankInfo));
    if (conflicts.first <= bestConflicts.first &&
        conflicts.second <= bestConflicts.second &&
        conflicts != bestConflicts) {
      best = candidate;
      bestConflicts = conflicts;
    }
  }
  // Allocation uses the default layout. Bank-column permutations preserve
  // every input dimension, including repetitions, so the same scratch size
  // and reuse/barrier schedule remain valid.
  return best;
}

} // namespace mlir::triton::ascend
