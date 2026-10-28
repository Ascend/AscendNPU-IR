//===- SharedMemorySwizzle.h - Ascend scratch swizzles ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_TRITONASCENDGPUTOLLVM_SHAREDMEMORYSWIZZLE_H
#define BISHENGIR_CONVERSION_TRITONASCENDGPUTOLLVM_SHAREDMEMORYSWIZZLE_H

#include <cstdint>
#include <utility>

namespace mlir::triton {
class LinearLayout;

namespace ascend {

// Effective shared-memory service resources, not the physical SRAM structure.
// Repeated accesses within one aligned word count once. A transaction spans
// numBanks * bankWidthBits bits.
struct SharedMemoryBankInfo {
  unsigned numBanks = 32;
  unsigned bankWidthBits = 32;
};

// Log2 of the maximum distinct aligned words per resource, in {read, write}
// order. Accounts for emitted vector widths, register permutations, scratch
// repetitions and transaction lane groups. Requires power-of-two geometry and
// an element width no greater than the bank word.
std::pair<int, int> getLog2BankConflicts(const LinearLayout &src,
                                         const LinearLayout &dst,
                                         const LinearLayout &smem,
                                         int32_t bitwidth,
                                         const SharedMemoryBankInfo &bankInfo);

// Refine a default 32-bit scratch layout by permuting its bank bases. Preserve
// allocation size, repetitions and the emitted vector widths in both
// directions. Keep the baseline on ties, conflicting tradeoffs or unsupported
// transfers.
LinearLayout refineSharedMemorySwizzle(const LinearLayout &src,
                                       const LinearLayout &dst,
                                       const LinearLayout &baseline,
                                       int32_t bitwidth,
                                       const SharedMemoryBankInfo &bankInfo);

} // namespace ascend
} // namespace mlir::triton

#endif // BISHENGIR_CONVERSION_TRITONASCENDGPUTOLLVM_SHAREDMEMORYSWIZZLE_H
