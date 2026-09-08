/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#if defined(__DAV_C310__)

#include "__clang_cce_simt_intrinsics.h"
#include "RegBase/VecUtils.h"
#include "Vector/Histogram/HistogramUtils.h"

constexpr unsigned int MAX_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_histogram_1d(__ubuf__ T *inputs, __ubuf__ int32_t *bins,
                  int64_t input_size, int64_t input_stride,
                  int64_t bins_stride, int64_t num_bins) {
  using U = typename std::make_unsigned<T>::type;
  for (int64_t i = threadIdx.x; i < input_size; i += blockDim.x) {
    uint64_t value = static_cast<uint64_t>(static_cast<U>(inputs[i * input_stride]));
    if (value >= num_bins) {
      continue;
    }
    atomicAdd(bins + value * bins_stride, static_cast<int32_t>(1));
  }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
__aiv__ __attribute__((always_inline)) static void
simt_histogram_1d_masked(__ubuf__ T *inputs, __ubuf__ int32_t *bins, __ubuf__ bool *mask,
                         int64_t input_size, int64_t input_stride,
                         int64_t bins_stride,
                         int64_t mask_stride,
                         int64_t num_bins) {
  using U = typename std::make_unsigned<T>::type;
  // Reinterpret the mask as a byte (uint8_t) pointer to enable efficient
  // byte-level (8-bit) memory accesses. Since the mask is a bit-stream, loading
  // one byte (8 bits) at a time and extracting individual bits via bitwise operations.
  __ubuf__ const uint8_t *mask_bytes = reinterpret_cast<__ubuf__ const uint8_t *>(mask);

  for (uint32_t i = threadIdx.x; i < input_size; i += blockDim.x) {
    int64_t bit_idx = i * mask_stride;
    // Check if the current element's corresponding mask bit is 1.
    // `bit_idx >> 3` calculates the byte index (floor division by 8).
    // `bit_idx & 7` calculates the bit offset within that byte (modulo 8).
    // If the bit is 0 (mask is false), skip the current element.
    if (!((mask_bytes[bit_idx >> 3] >> (bit_idx & 7)) & 1)) {
      continue;
    }
    uint64_t value = static_cast<uint64_t>(static_cast<U>(inputs[i * input_stride]));
    if (value >= num_bins) {
      continue;
    }
    atomicAdd(bins + value * bins_stride, static_cast<int32_t>(1));
  }
}

//===-------------------------------------------------------------------===//
// dhistv2 SIMD fast path for u8/s8 histograms
//===-------------------------------------------------------------------===//
// The A5 (dav-c310) vector unit provides the `dhistv2` instruction: it adds
// the per-value frequency of a 256-lane u8 vector into a u16 accumulator
// vector, 128 bins per call (Bin_N0: bins 0-127, Bin_N1: bins 128-255). Two
// calls per 256-element chunk keep the whole 256-bin histogram in registers,
// replacing the per-element atomicAdd loop of the SIMT path.
//
// Since u8 values cannot exceed 255, bins >= 256 never receive counts; when
// num_bins > 256 those bins are simply left untouched (the caller pre-zeroes
// dst, the same contract the SIMT path relies on). s8 inputs have the same
// bit pattern (the SIMT path also casts through the unsigned type), so they
// reuse this path unchanged.

// u8 elements consumed per dhistv2 chunk (one full vector register).
constexpr int kDhistLanes = 256;
// Flush cadence: 255 * 256 = 65280 < 65536, so no u16 lane can overflow
// between two flushes into the u32 totals.
constexpr int kDhistFlushChunks = 255;

// Register accumulators for the 256 dhistv2 bins.
//   half[k]: u16 partial counts for bins [k*128, k*128 + 128)
//   total[j]: i32 totals for bins [j*64, j*64 + 64), updated by dhistv2Flush
struct DhistBins {
  vector_u16 half[2];
  vector_u32 total[4];
};

__aiv__ __attribute__((always_inline)) static void
dhistv2Init(DhistBins &acc) {
  vector_bool all = pset_b8(PAT_ALL);
  vdup(acc.half[0], static_cast<uint16_t>(0), all, MODE_ZEROING);
  vdup(acc.half[1], static_cast<uint16_t>(0), all, MODE_ZEROING);
  vdup(acc.total[0], static_cast<uint32_t>(0), all, MODE_ZEROING);
  vdup(acc.total[1], static_cast<uint32_t>(0), all, MODE_ZEROING);
  vdup(acc.total[2], static_cast<uint32_t>(0), all, MODE_ZEROING);
  vdup(acc.total[3], static_cast<uint32_t>(0), all, MODE_ZEROING);
}

// Add the frequencies of one full 256-lane chunk into the accumulators.
// `active` selects the lanes (elements) that take part in the histogram.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessChunk(DhistBins &acc, __ubuf__ uint8_t *src, vector_bool active) {
  vector_u8 values;
  vlds(values, src, 0, NORM);
  INTRINSIC(dhistv2, acc.half[0], values, active, Bin_N0);
  INTRINSIC(dhistv2, acc.half[1], values, active, Bin_N1);
}

// Fold the u16 partial counts into the u32 totals and reset them. Called at
// least every kDhistFlushChunks chunks so the u16 lanes cannot overflow.
__aiv__ __attribute__((always_inline)) static void
dhistv2Flush(DhistBins &acc) {
  vector_bool all = pset_b8(PAT_ALL);
  vector_u32 part;
  vunpack(part, acc.half[0], LOWER);
  vadd(acc.total[0], acc.total[0], part, all);
  vunpack(part, acc.half[0], HIGHER);
  vadd(acc.total[1], acc.total[1], part, all);
  vunpack(part, acc.half[1], LOWER);
  vadd(acc.total[2], acc.total[2], part, all);
  vunpack(part, acc.half[1], HIGHER);
  vadd(acc.total[3], acc.total[3], part, all);
  vdup(acc.half[0], static_cast<uint16_t>(0), all, MODE_ZEROING);
  vdup(acc.half[1], static_cast<uint16_t>(0), all, MODE_ZEROING);
}

// Clamp helper for the per-64-bin store masks below.
__aiv__ __attribute__((always_inline)) static uint32_t
dhistv2BinCount(int64_t remain) {
  return remain > 64 ? 64 : static_cast<uint32_t>(remain < 0 ? 0 : remain);
}

// Store the totals of bins [0, min(num_bins, 256)). Empty lane masks make
// vsts a no-op, so no runtime branch is needed for partial bin ranges.
__aiv__ __attribute__((always_inline)) static void
dhistv2StoreBins(DhistBins &acc, __ubuf__ int32_t *bins, int64_t num_bins) {
  dhistv2Flush(acc);
  __ubuf__ uint32_t *out = reinterpret_cast<__ubuf__ uint32_t *>(bins);
  uint32_t c0 = dhistv2BinCount(num_bins);
  uint32_t c1 = dhistv2BinCount(num_bins - 64);
  uint32_t c2 = dhistv2BinCount(num_bins - 128);
  uint32_t c3 = dhistv2BinCount(num_bins - 192);
  vector_bool m0, m1, m2, m3;
  CREATE_MASK_BY_SIZE(m0, uint32_t, c0);
  CREATE_MASK_BY_SIZE(m1, uint32_t, c1);
  CREATE_MASK_BY_SIZE(m2, uint32_t, c2);
  CREATE_MASK_BY_SIZE(m3, uint32_t, c3);
  vsts(acc.total[0], out, 0, NORM_B32, m0);
  vsts(acc.total[1], out, 64, NORM_B32, m1);
  vsts(acc.total[2], out, 128, NORM_B32, m2);
  vsts(acc.total[3], out, 192, NORM_B32, m3);
}

// Load the packed i1 bitstream (LSB-first, 1 bit per element) covering 256
// elements into a predicate register: predicate bit i enables lane i.
// `byteOffset` is the byte offset of the 32 bytes covering the chunk; it is
// always a multiple of 32 (one chunk per 32 mask bytes), which plds requires
// (the 256-bit predicate load must stay 32-byte aligned). plds loads the
// whole predicate with one vector instruction, so the vector scope contains
// no scalar __ubuf__ accesses.
__aiv__ __attribute__((always_inline)) static vector_bool
dhistv2LoadPackedMask(__ubuf__ uint8_t *maskBytes, int64_t byteOffset) {
  __ubuf__ uint32_t *words = reinterpret_cast<__ubuf__ uint32_t *>(maskBytes);
  vector_bool m;
  plds(m, words, static_cast<int32_t>(byteOffset), NORM);
  return m;
}

// Accumulate all full 256-element chunks, flushing every
// kDhistFlushChunks chunks. Masked variant: `maskBytes` is the packed i1
// bitstream of the whole tensor (LSB-first, 1 bit per element).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateMasked(DhistBins &acc, __ubuf__ uint8_t *src,
                        int64_t fullChunks, __ubuf__ uint8_t *maskBytes) {
  int64_t tiles = (fullChunks + kDhistFlushChunks - 1) / kDhistFlushChunks;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunk(acc, src + (done + c) * kDhistLanes,
                          dhistv2LoadPackedMask(maskBytes, (done + c) * 32));
    dhistv2Flush(acc);
  }
}

// Unmasked counterpart of dhistv2AccumulateMasked.
__aiv__ __attribute__((always_inline)) static void
dhistv2Accumulate(DhistBins &acc, __ubuf__ uint8_t *src, int64_t fullChunks) {
  vector_bool all = pset_b8(PAT_ALL);
  int64_t tiles = (fullChunks + kDhistFlushChunks - 1) / kDhistFlushChunks;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunk(acc, src + (done + c) * kDhistLanes, all);
    dhistv2Flush(acc);
  }
}

// Tail (< kDhistLanes trailing elements): processed with one more dhistv2
// chunk whose lane predicate keeps only the first `lanes` elements. The
// vlds of that chunk still reads a full 256-byte vector (address stays
// 32B aligned because tailStart is a multiple of kDhistLanes), so up to
// 255 bytes past the tensor end are read but never counted (predicate
// gates dhistv2 lanes). This stays entirely inside the vector scope: no
// pipe_barrier and no queued SIMT task.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunk(DhistBins &acc, __ubuf__ uint8_t *src, int64_t lanes) {
  uint32_t count = static_cast<uint32_t>(lanes);
  vector_bool active;
  CREATE_MASK_BY_SIZE(active, uint8_t, count); // plt_b8: lane i < count
  dhistv2ProcessChunk(acc, src, active);
}

// Masked tail: AND the count predicate with the packed mask bits. The mask
// bitstream only defines ceil(n/8) valid bytes; garbage bits beyond `lanes`
// are cut by the count predicate.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkMasked(DhistBins &acc, __ubuf__ uint8_t *src,
                              int64_t lanes, __ubuf__ uint8_t *maskBytes,
                              int64_t chunkIdx) {
  uint32_t count = static_cast<uint32_t>(lanes);
  vector_bool active;
  CREATE_MASK_BY_SIZE(active, uint8_t, count);
  vector_bool data = dhistv2LoadPackedMask(maskBytes, chunkIdx * 32);
  vector_bool all = pset_b8(PAT_ALL);
  pand(active, active, data, all);
  dhistv2ProcessChunk(acc, src, active);
}

// u8 fast path (unmasked): full chunks + one predicate-masked tail chunk.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DU8(memref_t<__ubuf__ uint8_t, 1> *src,
                     memref_t<__ubuf__ int32_t, 1> *dst, int64_t num_bins) {
  __ubuf__ uint8_t *srcPtr = src->aligned + src->offset;
  __ubuf__ int32_t *binsPtr = dst->aligned + dst->offset;
  int64_t fullChunks = src->sizes[0] / kDhistLanes;
  int64_t tail = src->sizes[0] - fullChunks * kDhistLanes;
  __VEC_SCOPE__ {
    DhistBins acc;
    dhistv2Init(acc);
    dhistv2Accumulate(acc, srcPtr, fullChunks);
    if (tail > 0)
      dhistv2ProcessTailChunk(acc, srcPtr + fullChunks * kDhistLanes, tail);
    dhistv2StoreBins(acc, binsPtr, num_bins);
  }
}

// u8 fast path (masked): the packed mask bitstream gates each chunk.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DMaskedU8(memref_t<__ubuf__ uint8_t, 1> *src,
                           memref_t<__ubuf__ int32_t, 1> *dst,
                           memref_t<__ubuf__ bool, 1> *mask, int64_t num_bins) {
  __ubuf__ uint8_t *srcPtr = src->aligned + src->offset;
  __ubuf__ int32_t *binsPtr = dst->aligned + dst->offset;
  __ubuf__ uint8_t *maskBytes =
      reinterpret_cast<__ubuf__ uint8_t *>(mask->aligned + mask->offset);
  int64_t fullChunks = src->sizes[0] / kDhistLanes;
  int64_t tail = src->sizes[0] - fullChunks * kDhistLanes;
  __VEC_SCOPE__ {
    DhistBins acc;
    dhistv2Init(acc);
    dhistv2AccumulateMasked(acc, srcPtr, fullChunks, maskBytes);
    if (tail > 0)
      dhistv2ProcessTailChunkMasked(acc, srcPtr + fullChunks * kDhistLanes,
                                    tail, maskBytes, fullChunks);
    dhistv2StoreBins(acc, binsPtr, num_bins);
  }
}

//===-------------------------------------------------------------------===//
// dhistv2 SIMD fast path for u16/s16 histograms (segmented, any num_bins)
//===-------------------------------------------------------------------===//
// dhistv2 only covers the 256 u8 bins, so wider inputs are processed in
// segments of 256 bins: segment s covers values [s*256, (s+1)*256), and the
// counted bin of a value v in that segment is v - s*256 = the low byte of
// (v - s*256). Per segment the chunk pipeline is:
//   1. shift the segment down: w = v + (65536 - s*256) mod 65536, so
//      w < 256 selects exactly the values of segment s (values below the
//      segment wrap around to >= 65280 and are excluded by the compare),
//   2. predicate: w < 256 (per 128-lane register, merged with ppack/psel
//      into the 256-bit byte predicate; por does not exist on c310),
//   3. vpack the low bytes of both shifted registers into one 256-lane u8
//      vector (LOWER from the first, HIGHER from the second),
//   4. two dhistv2 calls accumulate the 256 segment bins.
// The segment loop re-reads the source once per segment; the cost per
// element therefore grows linearly with the segment count, so very wide
// histograms (the per-width segment limit exceeded) stay on the SIMT path.
// s16 shares the u16 bit pattern.

// Segment limit: beyond this even large inputs fall back to SIMT (the cost
// grows linearly with the segment count, calibrated on 950PR). Since the
// counted values are < num_bins, this is effectively a bins threshold: it
// applies to any input width, not just to a specific dtype. The u32 chunk
// loop processes four registers per chunk (vs two for u16), so its
// per-segment cost is ~2x higher and the profitable segment range halves.
constexpr int64_t kDhistMaxSegmentsU16 = 16;
constexpr int64_t kDhistMaxSegmentsU32 = 8;
// Segment count always profitable regardless of the input size: up to this
// many segments the dhistv2 path beats SIMT atomics even for tiny inputs.
constexpr int64_t kDhistAlwaysSegments = 4;
// Full chunks per segment needed to amortize the per-segment fixed costs
// (accumulator init + bin store) when kDhistAlwaysSegments is exceeded.
constexpr int64_t kDhistChunksPerSegment = 2;

// Segment count of a histogram: values >= num_bins are never counted, so
// only ceil(num_bins / 256) segments can receive counts. Computed with
// shifts only: the dav-c310 scalar pipeline has no fast integer division,
// and a divide here would tax every eligibility check (the fallback path
// measured ~50 cycles slower per call before this).
__aiv__ __attribute__((always_inline)) static int64_t
dhistv2Segments(int64_t num_bins) {
  return num_bins > 0 ? (num_bins + kDhistLanes - 1) >> 8 : 0;
}

// Size-aware segment gating shared by the u16 and u32 paths (calibrated on
// 950PR): few segments always win; many segments re-read the whole source
// per segment, so they need a large enough input to amortize the
// per-segment fixed costs, otherwise the SIMT atomics are faster (e.g. 16
// segments on 1024 elements run at 0.45x). `maxSegments` carries the
// per-width limit (see the constants above).
__aiv__ __attribute__((always_inline)) static bool
dhistv2SegmentEligible(int64_t num_bins, int64_t input_size,
                       int64_t maxSegments) {
  int64_t segs = dhistv2Segments(num_bins);
  if (segs > maxSegments)
    return false;
  if (segs <= kDhistAlwaysSegments)
    return true;
  return (input_size >> 8) >= kDhistChunksPerSegment * segs;
}

// Count predicate keeping the first `lanes` byte lanes of a u16 chunk.
// plt_b16 only reaches 128 lanes, so each half gets its own count predicate
// and they are merged like the value predicates.
__aiv__ __attribute__((always_inline)) static vector_bool
dhistv2CountPredU16(uint32_t lanes) {
  uint32_t lo = lanes > 128 ? 128 : lanes;
  uint32_t hi = lanes > 128 ? lanes - 128 : 0;
  vector_bool cl, ch, pl, ph, active;
  CREATE_MASK_BY_SIZE(cl, uint16_t, lo);
  CREATE_MASK_BY_SIZE(ch, uint16_t, hi);
  ppack(pl, cl, LOWER);
  ppack(ph, ch, HIGHER);
  psel(active, pl, ph, pset_b8(PAT_H));
  return active;
}

// Add the segment-base offset applied to one 128-lane register. s == 0 keeps
// the original values (adds 0), so the single-segment case pays no extra
// shift.
__aiv__ __attribute__((always_inline)) static void
dhistv2ShiftSegment(vector_u16 &w, const vector_u16 &v, uint16_t negBase,
                    vector_bool all) {
  vadds(w, v, negBase, all, MODE_ZEROING);
}

// Add the frequencies of one 256-element u16 chunk of segment `negBase`
// (see dhistv2ShiftSegment) into the accumulators. `active` selects the
// counted elements and is ANDed with the segment predicate. Both vlds
// always read a full 512 bytes (the tail chunk may over-read up to 510
// bytes past the tensor end, gated by the predicate — the same contract as
// the u8 path's 255 bytes).
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessChunkU16(DhistBins &acc, __ubuf__ uint16_t *src,
                       vector_bool active, uint16_t negBase) {
  vector_bool all = pset_b8(PAT_ALL);
  vector_u16 v0, v1, w0, w1;
  vlds(v0, src, 0, NORM);       // elements 0..127
  vlds(v1, src + 128, 0, NORM); // elements 128..255
  dhistv2ShiftSegment(w0, v0, negBase, all);
  dhistv2ShiftSegment(w1, v1, negBase, all);
  // Countable iff the shifted value fits in a byte: segment member.
  vector_bool p0, p1, pl, ph, counted, pred;
  vcmps_lt(p0, w0, static_cast<uint16_t>(256), all);
  vcmps_lt(p1, w1, static_cast<uint16_t>(256), all);
  ppack(pl, p0, LOWER);
  ppack(ph, p1, HIGHER);
  psel(counted, pl, ph, pset_b8(PAT_H));
  pand(pred, counted, active, all);
  vector_u8 bytes;
  vpack(bytes, w0, LOWER, MODE_ZEROING);  // segment bins of elements 0..127
  vpack(bytes, w1, HIGHER, MODE_MERGING); // segment bins of elements 128..255
  INTRINSIC(dhistv2, acc.half[0], bytes, pred, Bin_N0);
  INTRINSIC(dhistv2, acc.half[1], bytes, pred, Bin_N1);
}

// Unmasked accumulation of all full 256-element chunks (512-byte stride).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateU16(DhistBins &acc, __ubuf__ uint16_t *src, int64_t fullChunks,
                     uint16_t negBase) {
  vector_bool all = pset_b8(PAT_ALL);
  int64_t tiles = (fullChunks + kDhistFlushChunks - 1) / kDhistFlushChunks;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkU16(acc, src + (done + c) * kDhistLanes, all, negBase);
    dhistv2Flush(acc);
  }
}

// Masked accumulation: the packed mask bitstream gates each chunk (32 bytes
// per 256 elements, same layout as the u8 path).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateMaskedU16(DhistBins &acc, __ubuf__ uint16_t *src,
                           int64_t fullChunks, __ubuf__ uint8_t *maskBytes,
                           uint16_t negBase) {
  int64_t tiles = (fullChunks + kDhistFlushChunks - 1) / kDhistFlushChunks;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkU16(acc, src + (done + c) * kDhistLanes,
                             dhistv2LoadPackedMask(maskBytes, (done + c) * 32),
                             negBase);
    dhistv2Flush(acc);
  }
}

// u16 tail (< kDhistLanes trailing elements): one more chunk whose count
// predicate keeps only the first `lanes` elements (see dhistv2CountPredU16).
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkU16(DhistBins &acc, __ubuf__ uint16_t *src,
                           int64_t lanes, uint16_t negBase) {
  dhistv2ProcessChunkU16(acc, src,
                         dhistv2CountPredU16(static_cast<uint32_t>(lanes)),
                         negBase);
}

// Masked u16 tail: count predicate AND packed mask bits.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkMaskedU16(DhistBins &acc, __ubuf__ uint16_t *src,
                                 int64_t lanes, __ubuf__ uint8_t *maskBytes,
                                 int64_t chunkIdx, uint16_t negBase) {
  vector_bool active = dhistv2CountPredU16(static_cast<uint32_t>(lanes));
  vector_bool data = dhistv2LoadPackedMask(maskBytes, chunkIdx * 32);
  vector_bool all = pset_b8(PAT_ALL);
  vector_bool combined;
  pand(combined, active, data, all);
  dhistv2ProcessChunkU16(acc, src, combined, negBase);
}

// Run one segment pass: accumulate the whole source for segment `seg`, then
// store the [0, segBins) bins at dst + seg*256. segBins is clamped inside
// dhistv2StoreBins; bins past 65536 never exist for u16 inputs.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DU16Segment(DhistBins &acc, __ubuf__ uint16_t *srcPtr,
                             int64_t fullChunks, int64_t tail,
                             __ubuf__ int32_t *binsPtr, int64_t num_bins,
                             int64_t seg) {
  uint16_t negBase = static_cast<uint16_t>(0 - seg * kDhistLanes);
  dhistv2Init(acc);
  dhistv2AccumulateU16(acc, srcPtr, fullChunks, negBase);
  if (tail > 0)
    dhistv2ProcessTailChunkU16(acc, srcPtr + fullChunks * kDhistLanes, tail,
                               negBase);
  dhistv2StoreBins(acc, binsPtr + seg * kDhistLanes, num_bins - seg * kDhistLanes);
}

__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DMaskedU16Segment(DhistBins &acc, __ubuf__ uint16_t *srcPtr,
                                   int64_t fullChunks, int64_t tail,
                                   __ubuf__ uint8_t *maskBytes,
                                   __ubuf__ int32_t *binsPtr, int64_t num_bins,
                                   int64_t seg) {
  uint16_t negBase = static_cast<uint16_t>(0 - seg * kDhistLanes);
  dhistv2Init(acc);
  dhistv2AccumulateMaskedU16(acc, srcPtr, fullChunks, maskBytes, negBase);
  if (tail > 0)
    dhistv2ProcessTailChunkMaskedU16(acc, srcPtr + fullChunks * kDhistLanes,
                                     tail, maskBytes, fullChunks, negBase);
  dhistv2StoreBins(acc, binsPtr + seg * kDhistLanes, num_bins - seg * kDhistLanes);
}

//===-------------------------------------------------------------------===//
// Narrow-bins clamp fast path (u16/u32, num_bins < 256)
//===-------------------------------------------------------------------===//
// When num_bins < 256 the 256 dhistv2 bins have one spare slot past the last
// exported bin: the sentinel index num_bins. Values >= num_bins (including
// negative s16/s32, whose bit patterns are >= 2^15 / 2^31 as unsigned) are
// then CLAMPED to the sentinel instead of filtered by a value predicate:
// dhistv2StoreBins never exports bin num_bins, so clamped elements are
// silently dropped — the exact "values >= num_bins are not counted" contract
// of the SIMT path. num_bins == 256 leaves no spare bin and keeps the
// predicate segment path.
//
// The clamp is an unsigned min(v, num_bins), but c310 only provides a SIGNED
// vector-scalar min (vmins). Adding the sign bit first (vadds with 0x8000 /
// 0x80000000 — adding the top bit flips only the MSB) maps the unsigned
// order onto the signed order, so the signed min against (num_bins | sign
// bit) performs the unsigned clamp. The bias only touches the top bit, so
// the low byte of the result is already the clamped bin (or the sentinel):
// no un-biasing step is needed before the vpack byte extraction.
//
// Per 128-lane (u16) / 64-lane (u32) register this costs one vadds + one
// vmins and removes the segment shift, the vcmps_lt compare and the whole
// ppack/psel predicate merge chain: a u16 chunk drops from ~16 to ~10 vector
// instructions, a u32 chunk from ~28 to ~20.

// Clamp selector: the sentinel bin index for num_bins < 256, or -1 when the
// predicate segment path must be used (num_bins >= 256 leaves no spare bin).
__aiv__ __attribute__((always_inline)) static int32_t
dhistv2ClampSentinel(int64_t num_bins) {
  return (num_bins > 0 && num_bins < kDhistLanes)
             ? static_cast<int32_t>(num_bins)
             : -1;
}

// Clamp-path counterpart of dhistv2ProcessChunkU16: one 256-element chunk,
// `active` gates the counted elements (tail count / packed mask); the value
// range is handled by the clamp, so no predicate merge chain is needed.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessChunkClampU16(DhistBins &acc, __ubuf__ uint16_t *src,
                            vector_bool active, int32_t sentinel) {
  vector_bool all = pset_b8(PAT_ALL);
  vector_u16 v0, v1;
  vlds(v0, src, 0, NORM);       // elements 0..127
  vlds(v1, src + 128, 0, NORM); // elements 128..255
  // Bias to signed order, then signed-min against the biased sentinel =
  // the unsigned clamp (see the section comment).
  vadds(v0, v0, static_cast<uint16_t>(0x8000), all, MODE_ZEROING);
  vadds(v1, v1, static_cast<uint16_t>(0x8000), all, MODE_ZEROING);
  int16_t biased =
      static_cast<int16_t>(0x8000 | static_cast<uint16_t>(sentinel));
  vmins((VectorReg<int16_t> &)v0, (VectorReg<int16_t> &)v0, biased, all,
        MODE_ZEROING);
  vmins((VectorReg<int16_t> &)v1, (VectorReg<int16_t> &)v1, biased, all,
        MODE_ZEROING);
  vector_u8 bytes;
  vpack(bytes, v0, LOWER, MODE_ZEROING);  // clamped bins of elements 0..127
  vpack(bytes, v1, HIGHER, MODE_MERGING); // clamped bins of elements 128..255
  INTRINSIC(dhistv2, acc.half[0], bytes, active, Bin_N0);
  INTRINSIC(dhistv2, acc.half[1], bytes, active, Bin_N1);
}

// Clamp-path accumulation of all full 256-element chunks (512-byte stride).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateClampU16(DhistBins &acc, __ubuf__ uint16_t *src,
                          int64_t fullChunks, int32_t sentinel) {
  vector_bool all = pset_b8(PAT_ALL);
  // The tile count stays far below 2^32 (UB caps the source at a few hundred
  // chunks), and the dav-c310 backend has no 64-bit division: compute the
  // tile bound in 32 bits.
  uint32_t tiles =
      static_cast<uint32_t>(fullChunks + kDhistFlushChunks - 1) / 255u;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkClampU16(acc, src + (done + c) * kDhistLanes, all,
                                  sentinel);
    dhistv2Flush(acc);
  }
}

// Clamp-path masked accumulation (packed mask bitstream, as the u8/u16 paths).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateClampMaskedU16(DhistBins &acc, __ubuf__ uint16_t *src,
                                int64_t fullChunks,
                                __ubuf__ uint8_t *maskBytes, int32_t sentinel) {
  // 32-bit tile bound: no 64-bit division on dav-c310 (see
  // dhistv2AccumulateClampU16).
  uint32_t tiles =
      static_cast<uint32_t>(fullChunks + kDhistFlushChunks - 1) / 255u;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkClampU16(
          acc, src + (done + c) * kDhistLanes,
          dhistv2LoadPackedMask(maskBytes, (done + c) * 32), sentinel);
    dhistv2Flush(acc);
  }
}

// Clamp-path u16 tail: count predicate keeps the first `lanes` elements.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkClampU16(DhistBins &acc, __ubuf__ uint16_t *src,
                                int64_t lanes, int32_t sentinel) {
  dhistv2ProcessChunkClampU16(acc, src,
                              dhistv2CountPredU16(static_cast<uint32_t>(lanes)),
                              sentinel);
}

// Clamp-path masked u16 tail: count predicate AND packed mask bits.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkClampMaskedU16(DhistBins &acc, __ubuf__ uint16_t *src,
                                      int64_t lanes,
                                      __ubuf__ uint8_t *maskBytes,
                                      int64_t chunkIdx, int32_t sentinel) {
  vector_bool active = dhistv2CountPredU16(static_cast<uint32_t>(lanes));
  vector_bool data = dhistv2LoadPackedMask(maskBytes, chunkIdx * 32);
  vector_bool all = pset_b8(PAT_ALL);
  vector_bool combined;
  pand(combined, active, data, all);
  dhistv2ProcessChunkClampU16(acc, src, combined, sentinel);
}

// u16 fast path (unmasked): narrow-bins clamp when num_bins < 256, otherwise
// segmented.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DU16(memref_t<__ubuf__ uint16_t, 1> *src,
                      memref_t<__ubuf__ int32_t, 1> *dst, int64_t num_bins) {
  __ubuf__ uint16_t *srcPtr = src->aligned + src->offset;
  __ubuf__ int32_t *binsPtr = dst->aligned + dst->offset;
  int64_t fullChunks = src->sizes[0] >> 8;
  int64_t tail = src->sizes[0] - (fullChunks << 8);
  // Vector-scope loops take a single uint16_t induction variable, so the
  // segment bound is hoisted (the eligibility gate caps it at
  // kDhistMaxSegmentsU16, well below 65536).
  uint16_t segs = static_cast<uint16_t>(dhistv2Segments(num_bins));
  int32_t sentinel = dhistv2ClampSentinel(num_bins);
  __VEC_SCOPE__ {
    DhistBins acc;
    if (sentinel >= 0) {
      dhistv2Init(acc);
      dhistv2AccumulateClampU16(acc, srcPtr, fullChunks, sentinel);
      if (tail > 0)
        dhistv2ProcessTailChunkClampU16(acc, srcPtr + fullChunks * kDhistLanes,
                                        tail, sentinel);
      dhistv2StoreBins(acc, binsPtr, num_bins);
    } else {
      for (uint16_t s = 0; s < segs; ++s)
        dhistv2Histogram1DU16Segment(acc, srcPtr, fullChunks, tail, binsPtr,
                                     num_bins, s);
    }
  }
}

// u16 fast path (masked): narrow-bins clamp when num_bins < 256, otherwise
// segmented.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DMaskedU16(memref_t<__ubuf__ uint16_t, 1> *src,
                            memref_t<__ubuf__ int32_t, 1> *dst,
                            memref_t<__ubuf__ bool, 1> *mask, int64_t num_bins) {
  __ubuf__ uint16_t *srcPtr = src->aligned + src->offset;
  __ubuf__ int32_t *binsPtr = dst->aligned + dst->offset;
  __ubuf__ uint8_t *maskBytes =
      reinterpret_cast<__ubuf__ uint8_t *>(mask->aligned + mask->offset);
  int64_t fullChunks = src->sizes[0] / kDhistLanes;
  int64_t tail = src->sizes[0] - fullChunks * kDhistLanes;
  uint16_t segs = static_cast<uint16_t>(dhistv2Segments(num_bins));
  int32_t sentinel = dhistv2ClampSentinel(num_bins);
  __VEC_SCOPE__ {
    DhistBins acc;
    if (sentinel >= 0) {
      dhistv2Init(acc);
      dhistv2AccumulateClampMaskedU16(acc, srcPtr, fullChunks, maskBytes,
                                      sentinel);
      if (tail > 0)
        dhistv2ProcessTailChunkClampMaskedU16(
            acc, srcPtr + fullChunks * kDhistLanes, tail, maskBytes,
            fullChunks, sentinel);
      dhistv2StoreBins(acc, binsPtr, num_bins);
    } else {
      for (uint16_t s = 0; s < segs; ++s)
        dhistv2Histogram1DMaskedU16Segment(acc, srcPtr, fullChunks, tail,
                                           maskBytes, binsPtr, num_bins, s);
    }
  }
}

//===-------------------------------------------------------------------===//
// dhistv2 SIMD fast path for u32/s32 histograms (segmented, bins-gated)
//===-------------------------------------------------------------------===//
// 32-bit inputs reuse the segment scheme of the u16 path: segment s covers
// values [s*256, (s+1)*256) and the counted bin is the low byte of
// (v - s*256). The eligibility gate is the same bins threshold
// (dhistv2SegmentEligible): values >= num_bins are never counted, so a small
// num_bins bounds the interesting value range regardless of the input width —
// u32/s32 inputs with few bins still profit from dhistv2.
//
// Per chunk the pipeline is (probe-validated on 950PR):
//   1. 4 x vlds of 64 u32 lanes (256 bytes each),
//   2. vadds the segment base down: w = v + (2^32 - s*256) mod 2^32, so
//      w < 256 (unsigned u32 compare) selects exactly the segment members —
//      values outside [0, 2^32) cannot exist and large values (e.g. values
//      with the sign bit set) wrap far above 256 and drop out,
//   3. one vcmps_lt per register; the four 64-lane predicates merge into the
//      256-bit byte predicate by a 3-level ppack/psel chain (por does not
//      exist on c310),
//   4. vpack chain u32 -> u16 -> u8 extracts the low bytes of the shifted
//      values (the segment bins),
//   5. two dhistv2 calls accumulate the 256 segment bins.
// s32 shares the u32 bit pattern (the SIMT path also casts through the
// unsigned type; negative values map to >= 2^31 and are never counted for
// the bins thresholds considered here).
//
// u64/s64 inputs stay on the SIMT path: ccec provides no 64-bit vector
// types (vpack tops out at u32 -> u16), and the interleaved lo/hi u32
// halves of in-memory u64 values cannot be de-interleaved or cross-lane
// checked with the available predicate operations.

// Count predicate keeping the first `lanes` u32 lanes of a chunk. The
// predicate applies to the packed 256-byte vector, so plt_b8 directly
// produces the dense per-byte mask (plt_b32 would yield a sparse stride-4
// bit pattern that does not match the dhistv2 byte lanes).
__aiv__ __attribute__((always_inline)) static vector_bool
dhistv2CountPredU32(uint32_t lanes) {
  vector_bool active;
  CREATE_MASK_BY_SIZE(active, uint8_t, lanes); // plt_b8: lane i < count
  return active;
}

// Add the frequencies of one 256-element u32 chunk of segment `negBase`
// (see the section comment) into the accumulators. `active` selects the
// counted elements. The four vlds always read a full 1024 bytes (the tail
// chunk may over-read up to 1020 bytes past the tensor end, gated by the
// predicate — the same contract as the u8/u16 paths).
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessChunkU32(DhistBins &acc, __ubuf__ uint32_t *src,
                       vector_bool active, uint32_t negBase) {
  vector_bool all = pset_b8(PAT_ALL);
  vector_u32 v0, v1, v2, v3;
  vlds(v0, src, 0, NORM);       // elements 0..63
  vlds(v1, src + 64, 0, NORM);  // elements 64..127
  vlds(v2, src + 128, 0, NORM); // elements 128..191
  vlds(v3, src + 192, 0, NORM); // elements 192..255
  vadds(v0, v0, negBase, all, MODE_ZEROING);
  vadds(v1, v1, negBase, all, MODE_ZEROING);
  vadds(v2, v2, negBase, all, MODE_ZEROING);
  vadds(v3, v3, negBase, all, MODE_ZEROING);
  // Countable iff the shifted value fits in a byte: segment member.
  vector_bool q0, q1, q2, q3, ml, mh, m01, m23, pl, ph, counted, pred;
  vcmps_lt(q0, v0, static_cast<uint32_t>(256), all);
  vcmps_lt(q1, v1, static_cast<uint32_t>(256), all);
  vcmps_lt(q2, v2, static_cast<uint32_t>(256), all);
  vcmps_lt(q3, v3, static_cast<uint32_t>(256), all);
  ppack(ml, q0, LOWER);
  ppack(mh, q1, HIGHER);
  psel(m01, ml, mh, pset_b8(PAT_H));
  ppack(ml, q2, LOWER);
  ppack(mh, q3, HIGHER);
  psel(m23, ml, mh, pset_b8(PAT_H));
  ppack(pl, m01, LOWER);
  ppack(ph, m23, HIGHER);
  psel(counted, pl, ph, pset_b8(PAT_H));
  pand(pred, counted, active, all);
  // Low bytes of the shifted values = segment bins of elements 0..255.
  vector_u16 w0, w1;
  vpack(w0, v0, LOWER, MODE_ZEROING);  // elements 0..127
  vpack(w0, v1, HIGHER, MODE_MERGING);
  vpack(w1, v2, LOWER, MODE_ZEROING);  // elements 128..255
  vpack(w1, v3, HIGHER, MODE_MERGING);
  vector_u8 bytes;
  vpack(bytes, w0, LOWER, MODE_ZEROING);
  vpack(bytes, w1, HIGHER, MODE_MERGING);
  INTRINSIC(dhistv2, acc.half[0], bytes, pred, Bin_N0);
  INTRINSIC(dhistv2, acc.half[1], bytes, pred, Bin_N1);
}

// Unmasked accumulation of all full 256-element chunks (1024-byte stride).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateU32(DhistBins &acc, __ubuf__ uint32_t *src, int64_t fullChunks,
                     uint32_t negBase) {
  vector_bool all = pset_b8(PAT_ALL);
  int64_t tiles = (fullChunks + kDhistFlushChunks - 1) / kDhistFlushChunks;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkU32(acc, src + (done + c) * kDhistLanes, all, negBase);
    dhistv2Flush(acc);
  }
}

// Masked accumulation: the packed mask bitstream gates each chunk (32 bytes
// per 256 elements, same layout as the u8/u16 paths).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateMaskedU32(DhistBins &acc, __ubuf__ uint32_t *src,
                           int64_t fullChunks, __ubuf__ uint8_t *maskBytes,
                           uint32_t negBase) {
  int64_t tiles = (fullChunks + kDhistFlushChunks - 1) / kDhistFlushChunks;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkU32(acc, src + (done + c) * kDhistLanes,
                             dhistv2LoadPackedMask(maskBytes, (done + c) * 32),
                             negBase);
    dhistv2Flush(acc);
  }
}

// u32 tail (< kDhistLanes trailing elements): one more chunk whose count
// predicate keeps only the first `lanes` elements (see dhistv2CountPredU32).
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkU32(DhistBins &acc, __ubuf__ uint32_t *src,
                           int64_t lanes, uint32_t negBase) {
  dhistv2ProcessChunkU32(acc, src,
                         dhistv2CountPredU32(static_cast<uint32_t>(lanes)),
                         negBase);
}

// Masked u32 tail: count predicate AND packed mask bits.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkMaskedU32(DhistBins &acc, __ubuf__ uint32_t *src,
                                 int64_t lanes, __ubuf__ uint8_t *maskBytes,
                                 int64_t chunkIdx, uint32_t negBase) {
  vector_bool active = dhistv2CountPredU32(static_cast<uint32_t>(lanes));
  vector_bool data = dhistv2LoadPackedMask(maskBytes, chunkIdx * 32);
  vector_bool all = pset_b8(PAT_ALL);
  vector_bool combined;
  pand(combined, active, data, all);
  dhistv2ProcessChunkU32(acc, src, combined, negBase);
}

// Run one segment pass (mirror of the u16 segment helpers).
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DU32Segment(DhistBins &acc, __ubuf__ uint32_t *srcPtr,
                             int64_t fullChunks, int64_t tail,
                             __ubuf__ int32_t *binsPtr, int64_t num_bins,
                             int64_t seg) {
  uint32_t negBase = static_cast<uint32_t>(0u - static_cast<uint32_t>(seg) * 256u);
  dhistv2Init(acc);
  dhistv2AccumulateU32(acc, srcPtr, fullChunks, negBase);
  if (tail > 0)
    dhistv2ProcessTailChunkU32(acc, srcPtr + fullChunks * kDhistLanes, tail,
                               negBase);
  dhistv2StoreBins(acc, binsPtr + seg * kDhistLanes, num_bins - seg * kDhistLanes);
}

__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DMaskedU32Segment(DhistBins &acc, __ubuf__ uint32_t *srcPtr,
                                   int64_t fullChunks, int64_t tail,
                                   __ubuf__ uint8_t *maskBytes,
                                   __ubuf__ int32_t *binsPtr, int64_t num_bins,
                                   int64_t seg) {
  uint32_t negBase = static_cast<uint32_t>(0u - static_cast<uint32_t>(seg) * 256u);
  dhistv2Init(acc);
  dhistv2AccumulateMaskedU32(acc, srcPtr, fullChunks, maskBytes, negBase);
  if (tail > 0)
    dhistv2ProcessTailChunkMaskedU32(acc, srcPtr + fullChunks * kDhistLanes,
                                     tail, maskBytes, fullChunks, negBase);
  dhistv2StoreBins(acc, binsPtr + seg * kDhistLanes, num_bins - seg * kDhistLanes);
}

// Clamp-path counterpart of dhistv2ProcessChunkU32: same vlds/vpack shape,
// bias + signed min instead of the segment shift and predicate chain (see
// the clamp section comment).
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessChunkClampU32(DhistBins &acc, __ubuf__ uint32_t *src,
                            vector_bool active, int32_t sentinel) {
  vector_bool all = pset_b8(PAT_ALL);
  vector_u32 v0, v1, v2, v3;
  vlds(v0, src, 0, NORM);       // elements 0..63
  vlds(v1, src + 64, 0, NORM);  // elements 64..127
  vlds(v2, src + 128, 0, NORM); // elements 128..191
  vlds(v3, src + 192, 0, NORM); // elements 192..255
  vadds(v0, v0, 0x80000000u, all, MODE_ZEROING);
  vadds(v1, v1, 0x80000000u, all, MODE_ZEROING);
  vadds(v2, v2, 0x80000000u, all, MODE_ZEROING);
  vadds(v3, v3, 0x80000000u, all, MODE_ZEROING);
  int32_t biased =
      static_cast<int32_t>(0x80000000u | static_cast<uint32_t>(sentinel));
  vmins((VectorReg<int32_t> &)v0, (VectorReg<int32_t> &)v0, biased, all,
        MODE_ZEROING);
  vmins((VectorReg<int32_t> &)v1, (VectorReg<int32_t> &)v1, biased, all,
        MODE_ZEROING);
  vmins((VectorReg<int32_t> &)v2, (VectorReg<int32_t> &)v2, biased, all,
        MODE_ZEROING);
  vmins((VectorReg<int32_t> &)v3, (VectorReg<int32_t> &)v3, biased, all,
        MODE_ZEROING);
  vector_u16 w0, w1;
  vpack(w0, v0, LOWER, MODE_ZEROING);  // clamped bins of elements 0..127
  vpack(w0, v1, HIGHER, MODE_MERGING);
  vpack(w1, v2, LOWER, MODE_ZEROING);  // clamped bins of elements 128..255
  vpack(w1, v3, HIGHER, MODE_MERGING);
  vector_u8 bytes;
  vpack(bytes, w0, LOWER, MODE_ZEROING);
  vpack(bytes, w1, HIGHER, MODE_MERGING);
  INTRINSIC(dhistv2, acc.half[0], bytes, active, Bin_N0);
  INTRINSIC(dhistv2, acc.half[1], bytes, active, Bin_N1);
}

// Clamp-path u32 accumulation of all full 256-element chunks (1024-byte
// stride).
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateClampU32(DhistBins &acc, __ubuf__ uint32_t *src,
                          int64_t fullChunks, int32_t sentinel) {
  vector_bool all = pset_b8(PAT_ALL);
  // 32-bit tile bound: no 64-bit division on dav-c310 (see
  // dhistv2AccumulateClampU16).
  uint32_t tiles =
      static_cast<uint32_t>(fullChunks + kDhistFlushChunks - 1) / 255u;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkClampU32(acc, src + (done + c) * kDhistLanes, all,
                                  sentinel);
    dhistv2Flush(acc);
  }
}

// Clamp-path masked u32 accumulation.
__aiv__ __attribute__((always_inline)) static void
dhistv2AccumulateClampMaskedU32(DhistBins &acc, __ubuf__ uint32_t *src,
                                int64_t fullChunks,
                                __ubuf__ uint8_t *maskBytes, int32_t sentinel) {
  // 32-bit tile bound: no 64-bit division on dav-c310 (see
  // dhistv2AccumulateClampU16).
  uint32_t tiles =
      static_cast<uint32_t>(fullChunks + kDhistFlushChunks - 1) / 255u;
  for (uint16_t t = 0; t < static_cast<uint16_t>(tiles); ++t) {
    int64_t done = static_cast<int64_t>(t) * kDhistFlushChunks;
    int64_t left = fullChunks - done;
    uint16_t inTile =
        static_cast<uint16_t>(left > kDhistFlushChunks ? kDhistFlushChunks
                                                       : left);
    for (uint16_t c = 0; c < inTile; ++c)
      dhistv2ProcessChunkClampU32(
          acc, src + (done + c) * kDhistLanes,
          dhistv2LoadPackedMask(maskBytes, (done + c) * 32), sentinel);
    dhistv2Flush(acc);
  }
}

// Clamp-path u32 tail: plt_b8 dense count predicate (see dhistv2CountPredU32).
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkClampU32(DhistBins &acc, __ubuf__ uint32_t *src,
                                int64_t lanes, int32_t sentinel) {
  dhistv2ProcessChunkClampU32(acc, src,
                              dhistv2CountPredU32(static_cast<uint32_t>(lanes)),
                              sentinel);
}

// Clamp-path masked u32 tail: count predicate AND packed mask bits.
__aiv__ __attribute__((always_inline)) static void
dhistv2ProcessTailChunkClampMaskedU32(DhistBins &acc, __ubuf__ uint32_t *src,
                                      int64_t lanes,
                                      __ubuf__ uint8_t *maskBytes,
                                      int64_t chunkIdx, int32_t sentinel) {
  vector_bool active = dhistv2CountPredU32(static_cast<uint32_t>(lanes));
  vector_bool data = dhistv2LoadPackedMask(maskBytes, chunkIdx * 32);
  vector_bool all = pset_b8(PAT_ALL);
  vector_bool combined;
  pand(combined, active, data, all);
  dhistv2ProcessChunkClampU32(acc, src, combined, sentinel);
}

// u32 fast path (unmasked): narrow-bins clamp when num_bins < 256, otherwise
// segmented.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DU32(memref_t<__ubuf__ uint32_t, 1> *src,
                      memref_t<__ubuf__ int32_t, 1> *dst, int64_t num_bins) {
  __ubuf__ uint32_t *srcPtr = src->aligned + src->offset;
  __ubuf__ int32_t *binsPtr = dst->aligned + dst->offset;
  int64_t fullChunks = src->sizes[0] / kDhistLanes;
  int64_t tail = src->sizes[0] - fullChunks * kDhistLanes;
  uint16_t segs = static_cast<uint16_t>(dhistv2Segments(num_bins));
  int32_t sentinel = dhistv2ClampSentinel(num_bins);
  __VEC_SCOPE__ {
    DhistBins acc;
    if (sentinel >= 0) {
      dhistv2Init(acc);
      dhistv2AccumulateClampU32(acc, srcPtr, fullChunks, sentinel);
      if (tail > 0)
        dhistv2ProcessTailChunkClampU32(acc, srcPtr + fullChunks * kDhistLanes,
                                        tail, sentinel);
      dhistv2StoreBins(acc, binsPtr, num_bins);
    } else {
      for (uint16_t s = 0; s < segs; ++s)
        dhistv2Histogram1DU32Segment(acc, srcPtr, fullChunks, tail, binsPtr,
                                     num_bins, s);
    }
  }
}

// u32 fast path (masked): narrow-bins clamp when num_bins < 256, otherwise
// segmented.
__aiv__ __attribute__((always_inline)) static void
dhistv2Histogram1DMaskedU32(memref_t<__ubuf__ uint32_t, 1> *src,
                            memref_t<__ubuf__ int32_t, 1> *dst,
                            memref_t<__ubuf__ bool, 1> *mask, int64_t num_bins) {
  __ubuf__ uint32_t *srcPtr = src->aligned + src->offset;
  __ubuf__ int32_t *binsPtr = dst->aligned + dst->offset;
  __ubuf__ uint8_t *maskBytes =
      reinterpret_cast<__ubuf__ uint8_t *>(mask->aligned + mask->offset);
  int64_t fullChunks = src->sizes[0] / kDhistLanes;
  int64_t tail = src->sizes[0] - fullChunks * kDhistLanes;
  uint16_t segs = static_cast<uint16_t>(dhistv2Segments(num_bins));
  int32_t sentinel = dhistv2ClampSentinel(num_bins);
  __VEC_SCOPE__ {
    DhistBins acc;
    if (sentinel >= 0) {
      dhistv2Init(acc);
      dhistv2AccumulateClampMaskedU32(acc, srcPtr, fullChunks, maskBytes,
                                      sentinel);
      if (tail > 0)
        dhistv2ProcessTailChunkClampMaskedU32(
            acc, srcPtr + fullChunks * kDhistLanes, tail, maskBytes,
            fullChunks, sentinel);
      dhistv2StoreBins(acc, binsPtr, num_bins);
    } else {
      for (uint16_t s = 0; s < segs; ++s)
        dhistv2Histogram1DMaskedU32Segment(acc, srcPtr, fullChunks, tail,
                                           maskBytes, binsPtr, num_bins, s);
    }
  }
}

// The dhistv2 path needs contiguous, 32-byte aligned vector accesses on src
// and dst (vlds/vsts NORM alignment requirement). The alignment is checked on
// the effective address (base + offset), not just the memref offset.
template <typename T>
__aiv__ __attribute__((always_inline)) static bool
dhistv2Aligned32(__ubuf__ T *p) {
  return (reinterpret_cast<uintptr_t>(p) % 32) == 0;
}

template <typename T>
__aiv__ __attribute__((always_inline)) static bool
dhistv2Eligible1D(memref_t<__ubuf__ T, 1> *src,
                  memref_t<__ubuf__ int32_t, 1> *dst) {
  return src->strides[0] == 1 && dst->strides[0] == 1 &&
         dhistv2Aligned32(src->aligned + src->offset) &&
         dhistv2Aligned32(dst->aligned + dst->offset);
}

// The packed mask bitstream is loaded 32 bytes per chunk with plds, which
// requires the mask base address to stay 32-byte aligned.
__aiv__ __attribute__((always_inline)) static bool
dhistv2MaskEligible1D(memref_t<__ubuf__ bool, 1> *mask) {
  return mask->strides[0] == 1 &&
         dhistv2Aligned32(mask->aligned + mask->offset);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *dst,
            int64_t num_bins) {
  // Byte-sized inputs (u8/s8 share one bit pattern and stay within the 256
  // dhistv2 bins for any num_bins) with a contiguous layout run on the
  // dhistv2 SIMD path; 16-bit (u16/s16) and 32-bit (u32/s32) inputs run the
  // dhistv2 path while dhistv2SegmentEligible holds — the bins count, not
  // the dtype, bounds the counted value range, so u32/s32 with few bins take
  // the same path. num_bins < 256 further selects the sentinel-clamp fast
  // path (no predicate merge chain, see the clamp section); wider histograms
  // run the segmented passes. Every other case (strided/misaligned layout,
  // 64-bit integers, very wide histograms) keeps the original SIMT template
  // flow.
  if constexpr (sizeof(T) == 1) {
    auto *srcBytes = reinterpret_cast<memref_t<__ubuf__ uint8_t, 1> *>(src);
    if (dhistv2Eligible1D(srcBytes, dst)) {
      dhistv2Histogram1DU8(srcBytes, dst, num_bins);
      return;
    }
  } else if constexpr (sizeof(T) == 2) {
    auto *srcU16 = reinterpret_cast<memref_t<__ubuf__ uint16_t, 1> *>(src);
    if (dhistv2Eligible1D(srcU16, dst) &&
        dhistv2SegmentEligible(num_bins, src->sizes[0],
                               kDhistMaxSegmentsU16)) {
      dhistv2Histogram1DU16(srcU16, dst, num_bins);
      return;
    }
  } else if constexpr (sizeof(T) == 4) {
    auto *srcU32 = reinterpret_cast<memref_t<__ubuf__ uint32_t, 1> *>(src);
    if (dhistv2Eligible1D(srcU32, dst) &&
        dhistv2SegmentEligible(num_bins, src->sizes[0],
                               kDhistMaxSegmentsU32)) {
      dhistv2Histogram1DU32(srcU32, dst, num_bins);
      return;
    }
  }
  cce::async_invoke<simt_histogram_1d<T>>(
    cce::dim3{MAX_THREAD_NUM},
    reinterpret_cast<__ubuf__ T *>(src->aligned + src->offset),
    reinterpret_cast<__ubuf__ int32_t *>(dst->aligned + dst->offset),
    src->sizes[0], src->strides[0], dst->strides[0], num_bins);
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
histogram_1d_masked(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ int32_t, 1> *dst,
                    memref_t<__ubuf__ bool, 1> *mask, int64_t num_bins) {
  if constexpr (sizeof(T) == 1) {
    auto *srcBytes = reinterpret_cast<memref_t<__ubuf__ uint8_t, 1> *>(src);
    if (dhistv2Eligible1D(srcBytes, dst) && dhistv2MaskEligible1D(mask)) {
      dhistv2Histogram1DMaskedU8(srcBytes, dst, mask, num_bins);
      return;
    }
  } else if constexpr (sizeof(T) == 2) {
    auto *srcU16 = reinterpret_cast<memref_t<__ubuf__ uint16_t, 1> *>(src);
    if (dhistv2Eligible1D(srcU16, dst) && dhistv2MaskEligible1D(mask) &&
        dhistv2SegmentEligible(num_bins, src->sizes[0],
                               kDhistMaxSegmentsU16)) {
      dhistv2Histogram1DMaskedU16(srcU16, dst, mask, num_bins);
      return;
    }
  } else if constexpr (sizeof(T) == 4) {
    auto *srcU32 = reinterpret_cast<memref_t<__ubuf__ uint32_t, 1> *>(src);
    if (dhistv2Eligible1D(srcU32, dst) && dhistv2MaskEligible1D(mask) &&
        dhistv2SegmentEligible(num_bins, src->sizes[0],
                               kDhistMaxSegmentsU32)) {
      dhistv2Histogram1DMaskedU32(srcU32, dst, mask, num_bins);
      return;
    }
  }
  cce::async_invoke<simt_histogram_1d_masked<T>>(
    cce::dim3{MAX_THREAD_NUM},
    reinterpret_cast<__ubuf__ T *>(src->aligned + src->offset),
    reinterpret_cast<__ubuf__ int32_t *>(dst->aligned + dst->offset),
    reinterpret_cast<__ubuf__ bool *>(mask->aligned + mask->offset),
    src->sizes[0], src->strides[0], dst->strides[0], mask->strides[0],
    num_bins);
}

extern "C" {
//===-------------------------------------------------------------------===//
// histogram, 1 dim
//===-------------------------------------------------------------------===//
REGISTE_HISTOGRAM(1, uint8_t)
REGISTE_HISTOGRAM(1, int8_t)
REGISTE_HISTOGRAM(1, uint16_t)
REGISTE_HISTOGRAM(1, int16_t)
REGISTE_HISTOGRAM(1, uint32_t)
REGISTE_HISTOGRAM(1, int32_t)
REGISTE_HISTOGRAM(1, uint64_t)
REGISTE_HISTOGRAM(1, int64_t)
REGISTE_HISTOGRAM_MASKED(1, uint8_t)
REGISTE_HISTOGRAM_MASKED(1, int8_t)
REGISTE_HISTOGRAM_MASKED(1, uint16_t)
REGISTE_HISTOGRAM_MASKED(1, int16_t)
REGISTE_HISTOGRAM_MASKED(1, uint32_t)
REGISTE_HISTOGRAM_MASKED(1, int32_t)
REGISTE_HISTOGRAM_MASKED(1, uint64_t)
REGISTE_HISTOGRAM_MASKED(1, int64_t)
}
#endif
