//===- ConvertDotInputToLinearLayout.cpp - FMA-optimized linear layout---===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt  for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This pass converts dot operation inputs to a LinearEncoding that is as
// FMA-friendly as possible while staying within the non-shared-memory
// conversion class.
//
// Strategy:
//   1. Start from the source layout's LinearLayout.
//   2. Search candidate destination layouts that preserve warp/block structure
//      and only rearrange register/lane bases.
//   3. Keep only candidates that are "at most warp-shuffle" (i.e. register
//      reorder or warp shuffle, but not shared memory).
//   4. Among those, choose the one that maximizes K-contiguity in registers.
//
// Marker attribute "fma.converted" prevents re-processing the same dot op.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>

#define DEBUG_TYPE "convert-dot-input-to-linear-layout"

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_CONVERTDOTINPUTTOLINEARLAYOUT
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Marker to prevent re-processing
static constexpr char kPropagatedAttr[] = "dot.propagated";
static constexpr char kFMAConvertedAttr[] = "fma.converted";

static bool isBlockedEncoding(Attribute enc) {
  return isa<BlockedEncodingAttr>(enc);
}

static bool isFMAConverted(Operation *op) {
  return op->hasAttr(kFMAConvertedAttr);
}

static void markFMAConverted(Operation *op, MLIRContext *ctx) {
  op->setAttr(kFMAConvertedAttr, UnitAttr::get(ctx));
}

static void markDotPropagated(Operation *op, MLIRContext *ctx) {
  op->setAttr(kPropagatedAttr, UnitAttr::get(ctx));
}

// ─── Conversion cost classification ──────────────────────────────────────────
//
// We use this only for debug logging. The actual search below filters for
// layouts that avoid shared memory.

enum class ConvertCost { RegisterReorder, WarpShuffle, SharedMemory };

static ConvertCost classifyConversionCost(RankedTensorType srcTy,
                                          RankedTensorType dstTy) {
  if (cvtReordersRegisters(srcTy, dstTy))
    return ConvertCost::RegisterReorder;
  if (cvtNeedsWarpShuffle(srcTy, dstTy))
    return ConvertCost::WarpShuffle;
  return ConvertCost::SharedMemory;
}

static StringRef convCostStr(ConvertCost c) {
  switch (c) {
  case ConvertCost::RegisterReorder:
    return "register-reorder (free)";
  case ConvertCost::WarpShuffle:
    return "warp-shuffle (fast)";
  case ConvertCost::SharedMemory:
    return "shared-memory (SLOW)";
  }
  llvm_unreachable("unknown ConvertCost");
}

static bool isAtMostWarpShuffle(RankedTensorType srcTy,
                                RankedTensorType dstTy) {
  return cvtReordersRegisters(srcTy, dstTy) ||
         cvtNeedsWarpShuffle(srcTy, dstTy);
}

// ─── Shuffle-compatible FMA layout construction ─────────────────────────────
//
// We treat "avoid shared memory" as the hard constraint, then search among
// several candidate layouts that only reshuffle register/lane basis vectors.
//
// Candidate families:
//   A. Global K-first ordering of register+lane bases.
//   B. Preserve register order, K-sort lane bases.
//   C. K-sort register bases, preserve lane order.
//   D. Preserve source order (baseline / fallback).
//
// We choose the best valid candidate by scoring K-contiguity in the register
// portion, with a small penalty for moving far away from the source order.
//
// This is conservative: if a layout cannot be proven to stay in the
// non-shared-memory class, it is rejected.

struct BasisVec {
  SmallVector<int32_t> vec;
  bool fromRegister = false;
  unsigned originalIndex = 0;
};

static int64_t getKContribution(ArrayRef<int32_t> v, unsigned kOutDimIdx) {
  if (kOutDimIdx >= v.size())
    return 0;
  return static_cast<int64_t>(v[kOutDimIdx]);
}

// Generic version: returns the contribution of any output dimension, not just K.
static int64_t getDimContribution(ArrayRef<int32_t> v, unsigned dimIdx) {
  if (dimIdx >= v.size())
    return 0;
  return static_cast<int64_t>(v[dimIdx]);
}

// Score a candidate basis ordering.
//
// kOutDimIdx:    tensor dimension index of K (reduction axis). Used only for
//                the lane-penalty term (penalize K in lanes).
// fastRegDimIdx: tensor dimension index that should vary fastest within the
//                register portion (innermost register dimension).
//                - For A [M, K]: fastRegDimIdx = 1 (K, same as kOutDimIdx).
//                - For B [K, N]: fastRegDimIdx = 1 (N, NOT K) so that
//                  consecutive N elements at a fixed k are sequential in the
//                  struct ordinal, enabling efficient N-tile extraction.
static int64_t scoreCandidate(ArrayRef<BasisVec> ordered, unsigned nRegBases,
                              unsigned kOutDimIdx, unsigned fastRegDimIdx) {
  // Higher is better.
  // Reward fast-dim-heavy bases in the register prefix; penalize K in lanes.
  int64_t score = 0;
  for (unsigned i = 0; i < ordered.size(); ++i) {
    const auto &b = ordered[i];
    int64_t fast = getDimContribution(b.vec, fastRegDimIdx);
    int64_t k    = getKContribution(b.vec, kOutDimIdx);

    if (i < nRegBases) {
      // Reward fast dimension in registers.
      score += 1000 * (fast > 0 ? 1 : 0);
      score += 10 * fast;
    } else {
      // Penalize K-heavy bases falling into lane (K must stay in registers
      // for FMA to work without cross-thread reduction).
      score -= 250 * (k > 0 ? 1 : 0);
      score -= 2 * k;
    }

    // Mild stability bias: prefer preserving original order.
    score -= static_cast<int64_t>(std::abs(
        static_cast<int64_t>(i) - static_cast<int64_t>(b.originalIndex)));
  }
  return score;
}

static SmallVector<BasisVec> collectBases(const LinearLayout &ll,
                                          StringAttr key, bool fromRegister) {
  SmallVector<BasisVec> out;
  const auto &bases = ll.getBases();
  if (auto it = bases.find(key); it != bases.end()) {
    unsigned idx = 0;
    for (const auto &bvec : it->second) {
      out.push_back(BasisVec{
          SmallVector<int32_t>(bvec.begin(), bvec.end()),
          fromRegister,
          idx++,
      });
    }
  }
  return out;
}

static std::vector<std::vector<int32_t>> toVecOfVec(ArrayRef<BasisVec> in) {
  std::vector<std::vector<int32_t>> out;
  out.reserve(in.size());
  for (const auto &b : in)
    out.emplace_back(b.vec.begin(), b.vec.end());
  return out;
}

static LinearLayout buildLayoutFromOrdering(MLIRContext *ctx,
                                            const LinearLayout &srcLL,
                                            ArrayRef<BasisVec> ordered,
                                            unsigned nRegBases) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");

  // Split the ordered pool at the original register count.
  SmallVector<BasisVec> regPart, lanePart;
  regPart.reserve(nRegBases);
  lanePart.reserve(ordered.size() - nRegBases);
  for (unsigned i = 0; i < ordered.size(); ++i) {
    if (i < nRegBases)
      regPart.push_back(ordered[i]);
    else
      lanePart.push_back(ordered[i]);
  }

  SmallVector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>
      newBases;
  newBases.push_back({kRegister, toVecOfVec(regPart)});
  newBases.push_back({kLane, toVecOfVec(lanePart)});

  // Preserve warp/block structure exactly. Even if absent in source, keep
  // empty vectors so downstream verifier paths remain happy.
  auto copyOrEmpty = [&](StringAttr dim) {
    if (auto it = srcLL.getBases().find(dim); it != srcLL.getBases().end())
      newBases.push_back({dim, it->second});
    else
      newBases.push_back({dim, {}});
  };
  copyOrEmpty(kWarp);
  copyOrEmpty(kBlock);

  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  auto outNames = llvm::to_vector(srcLL.getOutDimNames());
  auto outSizes = llvm::to_vector(srcLL.getOutDimSizes());
  outDims.reserve(outNames.size());
  for (size_t i = 0, e = outNames.size(); i < e; ++i)
    outDims.push_back({outNames[i], outSizes[i]});

  return LinearLayout(newBases, outDims, /*requireSurjective=*/true);
}

static LinearLayout buildShuffleCompatibleFMALayout(MLIRContext *ctx,
                                                    const LinearLayout &srcLL,
                                                    unsigned kOutDimIdx,
                                                    unsigned fastRegDimIdx,
                                                    RankedTensorType srcTy,
                                                    Attribute srcEnc) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");

  SmallVector<BasisVec> regBases =
      collectBases(srcLL, kRegister, /*fromRegister=*/true);
  SmallVector<BasisVec> laneBases =
      collectBases(srcLL, kLane, /*fromRegister=*/false);

  unsigned nRegBases = regBases.size();

  // Baseline ordered pool: source order.
  SmallVector<BasisVec> sourceOrder;
  sourceOrder.append(regBases.begin(), regBases.end());
  sourceOrder.append(laneBases.begin(), laneBases.end());

  // Sort bases by their contribution to fastRegDimIdx (descending).
  // For A: fastRegDimIdx == kOutDimIdx (K), so behaviour is unchanged.
  // For B: fastRegDimIdx == N dim, so we sort N-contribution high first,
  //        giving N-innermost register ordering that pairs with the k-outer
  //        N-tile loop in FMADotUtility.
  auto makeSorted = [&](ArrayRef<BasisVec> in) {
    SmallVector<BasisVec> v(in.begin(), in.end());
    std::stable_sort(v.begin(), v.end(),
                     [fastRegDimIdx](const BasisVec &a, const BasisVec &b) {
                       int64_t af = getDimContribution(a.vec, fastRegDimIdx);
                       int64_t bf = getDimContribution(b.vec, fastRegDimIdx);
                       if (af != bf)
                         return af > bf;
                       return a.originalIndex < b.originalIndex;
                     });
    return v;
  };

  // Candidate A: global K-first.
  SmallVector<BasisVec> candGlobal = makeSorted(sourceOrder);

  // Candidate B: preserve register order, K-sort lane bases.
  SmallVector<BasisVec> candRegStable = regBases;
  {
    SmallVector<BasisVec> laneSorted = makeSorted(laneBases);
    candRegStable.append(laneSorted.begin(), laneSorted.end());
  }

  // Candidate C: K-sort register bases, preserve lane order.
  SmallVector<BasisVec> candLaneStable = makeSorted(regBases);
  candLaneStable.append(laneBases.begin(), laneBases.end());

  // Candidate D: pure source order.
  SmallVector<BasisVec> candSource = sourceOrder;

  struct Candidate {
    LinearLayout layout;
    int64_t score = std::numeric_limits<int64_t>::min();
    bool valid = false;
    bool warpShuffle = false;
    StringRef name;
  };

  auto evalCandidate = [&](StringRef name, ArrayRef<BasisVec> ordered) {
    Candidate c;
    c.name = name;
    c.layout = buildLayoutFromOrdering(ctx, srcLL, ordered, nRegBases);

    auto dstTy = RankedTensorType::get(srcTy.getShape(), srcTy.getElementType(),
                                       LinearEncodingAttr::get(ctx, c.layout));

    c.valid = isAtMostWarpShuffle(srcTy, dstTy);
    c.warpShuffle = cvtNeedsWarpShuffle(srcTy, dstTy);
    c.score = scoreCandidate(ordered, nRegBases, kOutDimIdx, fastRegDimIdx);

    LLVM_DEBUG({
      llvm::dbgs() << "  [FMA] candidate " << name << ": "
                   << (c.valid ? "valid" : "reject(shared-memory)") << ", kind="
                   << (c.warpShuffle ? "warp-shuffle" : "register-reorder")
                   << ", score=" << c.score << "\n";
      llvm::dbgs() << "        srcEnc=" << srcEnc << "\n";
      llvm::dbgs() << "        dstTy=" << dstTy << "\n";
    });
    return c;
  };

  Candidate best;
  auto consider = [&](Candidate c) {
    if (!c.valid)
      return;
    // Tiebreak: prefer register-reorder (cheaper) over warp-shuffle.
    // Previously this condition was inverted (c.warpShuffle &&
    // !best.warpShuffle), which incorrectly replaced a cheaper register-reorder
    // best with a more expensive warp-shuffle candidate of equal score.
    if (!best.valid || c.score > best.score ||
        (c.score == best.score && !c.warpShuffle && best.warpShuffle)) {
      best = std::move(c);
    }
  };

  // Search from most aggressive to most conservative.
  consider(evalCandidate("global-k-first", candGlobal));
  consider(evalCandidate("register-stable", candRegStable));
  consider(evalCandidate("lane-stable", candLaneStable));
  consider(evalCandidate("source-order", candSource));

  if (!best.valid) {
    // This should be rare. Fall back to source order so we do not invent a
    // layout that forces shared-memory conversion.
    LLVM_DEBUG(
        llvm::dbgs()
        << "  [FMA] no non-shared candidate found; using source order\n");
    return buildLayoutFromOrdering(ctx, srcLL, sourceOrder, nRegBases);
  }

  LLVM_DEBUG(
      llvm::dbgs() << "  [FMA] selected candidate: " << best.name
                   << " (score=" << best.score << ", kind="
                   << (best.warpShuffle ? "warp-shuffle" : "register-reorder")
                   << ")\n");
  return best.layout;
}

// ─── FMA LinearEncoding creation ─────────────────────────────────────────────

/// Create a LinearEncodingAttr for FMA-optimized dot inputs.
///
/// kDimIdx:      tensor dimension index of the K (reduction) axis.
///   - A [M, K]: kDimIdx = 1
///   - B [K, N]: kDimIdx = 0
///
/// fastRegDimIdx: tensor dimension that should be innermost (fastest-varying)
///               in the register portion of the layout.
///   - A [M, K]: fastRegDimIdx = 1 (K — same as kDimIdx, no change).
///   - B [K, N]: fastRegDimIdx = 1 (N — NOT K) so that consecutive N elements
///               at the same k step are adjacent in the struct ordinal, giving
///               sequential extraction in the N-tile inner loop.
///
/// The returned layout preserves warp/block structure and searches for the best
/// register/lane ordering that stays out of shared memory.
static Attribute createFMALinearEncoding(MLIRContext *ctx,
                                         RankedTensorType inputType,
                                         unsigned kDimIdx,
                                         unsigned fastRegDimIdx) {
  auto inputLL = toLinearLayout(inputType.getShape(), inputType.getEncoding());
  auto fmaLL = buildShuffleCompatibleFMALayout(ctx, inputLL, kDimIdx,
                                               fastRegDimIdx, inputType,
                                               inputType.getEncoding());

  LLVM_DEBUG(
      llvm::dbgs() << "  [FMA] built candidate linear layout for kDimIdx="
                   << kDimIdx << ", fastRegDimIdx=" << fastRegDimIdx << "\n");
  return LinearEncodingAttr::get(ctx, fmaLL);
}

// ─── Rewrite pattern: Convert dot inputs to FMA-friendly LinearEncoding ──────

struct ConvertDotInputToLinearPattern : public OpRewritePattern<DotOp> {
  explicit ConvertDotInputToLinearPattern(MLIRContext *context)
      : OpRewritePattern<DotOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (isFMAConverted(dotOp)) {
      LLVM_DEBUG(llvm::dbgs() << "[FMA] skip already-converted dot\n");
      return failure();
    }

    auto resultType = cast<RankedTensorType>(dotOp.getResult().getType());
    Attribute resultEnc = resultType.getEncoding();

    // Only transform dots whose result uses blocked layout (D-layout is fixed).
    if (!resultEnc || !isBlockedEncoding(resultEnc))
      return failure();

    MLIRContext *ctx = dotOp.getContext();
    Location loc = dotOp.getLoc();

    Value a = dotOp.getA(), b = dotOp.getB(), c = dotOp.getC();
    auto aType = cast<RankedTensorType>(a.getType());
    auto bType = cast<RankedTensorType>(b.getType());

    // Build separate FMA LinearEncodings for A [M, K] and B [K, N].
    //
    // For A: K is both the reduction axis (kDimIdx=1) and the fast register dim
    //        (fastRegDimIdx=1) — K-innermost, unchanged behaviour.
    // For B: K is the reduction axis (kDimIdx=0) but N is the fast register dim
    //        (fastRegDimIdx=1) — N-innermost, so that consecutive B[*, ni]
    //        elements at a fixed k step are adjacent in the struct ordinal,
    //        matching the N-tile inner loop in parametricConvertFMADot.
    Attribute fmaEncA = createFMALinearEncoding(ctx, aType, /*kDimIdx=*/1,
                                                /*fastRegDimIdx=*/1);
    Attribute fmaEncB = createFMALinearEncoding(ctx, bType, /*kDimIdx=*/0,
                                                /*fastRegDimIdx=*/1);

    // Guard: FMADotUtility requires aElems.size() % K == 0 and
    // bElems.size() % K == 0, where elems = 2^nRegisterBits per thread.
    //
    // If the source encoding has too few register bits (e.g. K is spread
    // across lanes because sizePerThread was 1 in the K direction), the
    // produced layout cannot satisfy this invariant. In that case bail out and
    // let the blocked-encoding dot path handle lowering instead.
    auto elemsPerThreadOk = [&](Attribute enc, int64_t kSize,
                                StringRef name) -> bool {
      auto linEnc = enc.dyn_cast<LinearEncodingAttr>();
      if (!linEnc)
        return false;
      auto ll = linEnc.getLinearLayout();
      auto regKey = StringAttr::get(ctx, "register");
      auto it = ll.getBases().find(regKey);
      int64_t nRegBits =
          (it != ll.getBases().end()) ? static_cast<int64_t>(it->second.size()) : 0;
      int64_t elemsPerThread = 1LL << nRegBits;
      bool ok = (elemsPerThread % kSize == 0);
      LLVM_DEBUG(if (!ok) llvm::dbgs()
                 << "[FMA] guard reject: " << name << " elemsPerThread="
                 << elemsPerThread << " not divisible by K=" << kSize
                 << " (need more register bits; K is in lane dimension)\n");
      return ok;
    };

    int64_t kForA = aType.getShape()[1]; // A[M, K]
    int64_t kForB = bType.getShape()[0]; // B[K, N]
    if (!elemsPerThreadOk(fmaEncA, kForA, "A") ||
        !elemsPerThreadOk(fmaEncB, kForB, "B")) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[FMA] skip: produced layout incompatible with "
             "FMADotUtility K-alignment; leaving dot for blocked path\n");
      return failure();
    }

    // Convert one operand to its FMA linear encoding.
    //   - Logs the conversion cost tier at debug level.
    //   - Unwraps redundant ConvertLayoutOp chains (AtoXtoY becomes AtoY).
    //   - Returns the original value if already at the target encoding.
    auto convertOperand = [&](Value val, RankedTensorType type,
                              Attribute targetEnc, StringRef name) -> Value {
      if (type.getEncoding() == targetEnc)
        return val;

      // Unwrap an existing convert to avoid chaining two conversions.
      Value src = val;
      if (auto cvtOp = val.getDefiningOp<ConvertLayoutOp>())
        src = cvtOp.getSrc();

      auto srcType = cast<RankedTensorType>(src.getType());
      auto dstType = RankedTensorType::get(srcType.getShape(),
                                           srcType.getElementType(), targetEnc);

      LLVM_DEBUG({
        auto cost = classifyConversionCost(srcType, dstType);
        llvm::dbgs() << "  [FMA] convert " << name << ": " << convCostStr(cost)
                     << "\n"
                     << "    src: " << srcType << "\n"
                     << "    dst: " << dstType << "\n";
      });

      return rewriter.create<ConvertLayoutOp>(loc, dstType, src);
    };

    Value newA = convertOperand(a, aType, fmaEncA, "A[M,K]");
    Value newB = convertOperand(b, bType, fmaEncB, "B[K,N]");

    // C (accumulator) keeps its blocked encoding — DotOp requires
    // result type == C type, and FMA lowering reads the D-layout from it.

    if (newA == a && newB == b) {
      LLVM_DEBUG(llvm::dbgs() << "[FMA] operands already optimal, skipping\n");
      return failure();
    }

    // New dot: result type unchanged (blocked), so downstream uses need no
    // additional convert_layout.
    auto newDot = rewriter.create<DotOp>(
        loc, resultType, ValueRange{newA, newB, c}, dotOp->getAttrs());
    markFMAConverted(newDot, ctx);
    markDotPropagated(newDot, ctx);
    rewriter.replaceOp(dotOp, newDot.getResult());

    LLVM_DEBUG({
      llvm::dbgs() << "[FMA] rewrote dot with linear-layout inputs:\n"
                   << "  A: " << newA.getType() << "\n"
                   << "  B: " << newB.getType() << "\n"
                   << "  C/D (blocked, preserved): " << resultType << "\n";
    });

    return success();
  }
};

// ─── PushConvertThroughLoad: load A directly in linear layout ────────────────
//
// Matches:  ttg.convert_layout(tt.load(ptrs, mask, other), dst=#linear)
//           where the conversion would use shared memory.
//
// Rewrites: rematerialize the entire address computation in #linear, then
//           emit tt.load directly in that layout — no shared memory needed.
//
// Why this is safe: the pointer tensors are computed from deterministic
// thread-identity arithmetic (make_range, splat, addi, muli, addptr).
// Each thread can recompute its own pointers independently without any
// cross-thread data exchange.  The 32×32 A matrix (2 KB) fits in L1 cache,
// so the duplicated loads caused by the 64× replication in the linear layout
// land in cache and do not increase effective memory traffic.
//
// Benefit: eliminates the shared-memory write+read for A to #linear plus
// both gpu.barrier operations that guard that shared memory region.

// ---------------------------------------------------------------------------
// Helper: derive a LinearEncodingAttr for a broadcast SOURCE from the
// broadcast DESTINATION encoding.  For each basis vector, zero out all
// contributions to dimensions where srcShape[d] == 1 (the broadcast dims),
// and set those output dimension sizes to 1.
// ---------------------------------------------------------------------------
static Attribute computeBroadcastSrcEncoding(Attribute dstEnc,
                                             ArrayRef<int64_t> srcShape) {
  auto *ctx = dstEnc.getContext();
  auto linEnc = dyn_cast<LinearEncodingAttr>(dstEnc);
  if (!linEnc) {
    // Non-linear fallback: use SliceEncodingAttr on the first singleton dim.
    for (unsigned d = 0; d < srcShape.size(); ++d) {
      if (srcShape[d] == 1)
        return SliceEncodingAttr::get(ctx, d,
                                      cast<DistributedEncodingTrait>(dstEnc));
    }
    return dstEnc;
  }

  const auto &ll = linEnc.getLinearLayout();

  // Identify broadcast dimensions (singleton in source).
  SmallVector<unsigned> bcastDims;
  for (unsigned d = 0; d < srcShape.size(); ++d)
    if (srcShape[d] == 1)
      bcastDims.push_back(d);

  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");

  // Zero out contributions to broadcast dimensions in every basis vector.
  auto projectBases = [&](StringAttr key) -> std::vector<std::vector<int32_t>> {
    std::vector<std::vector<int32_t>> newBases;
    auto it = ll.getBases().find(key);
    if (it == ll.getBases().end())
      return newBases;
    for (const auto &bvec : it->second) {
      std::vector<int32_t> nb(bvec.begin(), bvec.end());
      for (unsigned d : bcastDims)
        if (d < nb.size())
          nb[d] = 0;
      newBases.push_back(std::move(nb));
    }
    return newBases;
  };

  // Build output dims: same names, but broadcast dims have size 1.
  auto outNames = llvm::to_vector(ll.getOutDimNames());
  auto outSizes = llvm::to_vector(ll.getOutDimSizes());
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.reserve(outNames.size());
  for (size_t i = 0; i < outNames.size(); ++i) {
    int32_t sz = (i < srcShape.size() && srcShape[i] == 1) ? 1 : outSizes[i];
    outDims.push_back({outNames[i], sz});
  }

  LinearLayout srcLL({{kRegister, projectBases(kRegister)},
                      {kLane, projectBases(kLane)},
                      {kWarp, projectBases(kWarp)},
                      {kBlock, projectBases(kBlock)}},
                     outDims,
                     /*requireSurjective=*/false);
  return LinearEncodingAttr::get(ctx, srcLL);
}

// ---------------------------------------------------------------------------
// Rematerialize `v` in encoding `enc`.
//
// Walks backward through the def-use chain, cloning each producer op with
// the target encoding.  Handles the specific op types found in Triton address
// computation:  make_range, splat, expand_dims, broadcast, constant,
// arith elementwise ops, and tt.addptr.
//
// `cache` avoids re-creating the same value.
// ---------------------------------------------------------------------------
static Value rematerInLayout(Value v, Attribute enc, PatternRewriter &rewriter,
                             DenseMap<Value, Value> &cache);

static Value rematerInLayout(Value v, Attribute enc, PatternRewriter &rewriter,
                             DenseMap<Value, Value> &cache) {
  if (auto it = cache.find(v); it != cache.end())
    return it->second;

  // Non-tensor values (scalars, pointers) are unchanged.
  auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorTy) {
    cache[v] = v;
    return v;
  }

  auto newTy = RankedTensorType::get(tensorTy.getShape(),
                                     tensorTy.getElementType(), enc);
  Operation *def = v.getDefiningOp();
  Value result;

  if (!def) {
    // Block argument: fall back to convert_layout (rare for address ops).
    result = rewriter.create<ConvertLayoutOp>(v.getLoc(), newTy, v);
  } else if (auto constOp = dyn_cast<arith::ConstantOp>(def)) {
    // Dense constant: re-emit with new type, same values.
    auto dense = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (dense)
      result = rewriter.create<arith::ConstantOp>(constOp.getLoc(), newTy,
                                                  dense.reshape(newTy));
    else
      result = rewriter.create<ConvertLayoutOp>(v.getLoc(), newTy, v);
  } else if (auto rangeOp = dyn_cast<MakeRangeOp>(def)) {
    // make_range: same integer values, just different layout.
    result = rewriter.create<MakeRangeOp>(rangeOp.getLoc(), newTy,
                                          rangeOp.getStart(), rangeOp.getEnd());
  } else if (auto splatOp = dyn_cast<SplatOp>(def)) {
    // splat(scalar): scalar is unchanged, result uses new encoding.
    result =
        rewriter.create<SplatOp>(splatOp.getLoc(), newTy, splatOp.getSrc());
  } else if (auto bcOp = dyn_cast<BroadcastOp>(def)) {
    // Broadcast from (M,1) or (1,K) to (M,K).
    // Source encoding: project out the broadcast dimension from dstEnc.
    auto srcShape = cast<RankedTensorType>(bcOp.getSrc().getType()).getShape();
    Attribute srcEnc = computeBroadcastSrcEncoding(enc, srcShape);
    Value newSrc = rematerInLayout(bcOp.getSrc(), srcEnc, rewriter, cache);
    result = rewriter.create<BroadcastOp>(bcOp.getLoc(), newTy, newSrc);
  } else if (auto expandOp = dyn_cast<ExpandDimsOp>(def)) {
    // expand_dims: source encoding = SliceEncodingAttr(axis, enc).
    Attribute srcEnc = inferSrcEncoding(expandOp.getOperation(), enc);
    Value newSrc = rematerInLayout(expandOp.getSrc(), srcEnc, rewriter, cache);
    result = rewriter.create<ExpandDimsOp>(expandOp.getLoc(), newTy, newSrc,
                                           expandOp.getAxis());
  } else if (auto addIOp = dyn_cast<arith::AddIOp>(def)) {
    auto lhs = rematerInLayout(addIOp.getLhs(), enc, rewriter, cache);
    auto rhs = rematerInLayout(addIOp.getRhs(), enc, rewriter, cache);
    result = rewriter.create<arith::AddIOp>(addIOp.getLoc(), lhs, rhs);
  } else if (auto mulIOp = dyn_cast<arith::MulIOp>(def)) {
    auto lhs = rematerInLayout(mulIOp.getLhs(), enc, rewriter, cache);
    auto rhs = rematerInLayout(mulIOp.getRhs(), enc, rewriter, cache);
    result = rewriter.create<arith::MulIOp>(mulIOp.getLoc(), lhs, rhs);
  } else if (auto cmpIOp = dyn_cast<arith::CmpIOp>(def)) {
    auto lhs = rematerInLayout(cmpIOp.getLhs(), enc, rewriter, cache);
    auto rhs = rematerInLayout(cmpIOp.getRhs(), enc, rewriter, cache);
    result = rewriter.create<arith::CmpIOp>(cmpIOp.getLoc(),
                                            cmpIOp.getPredicate(), lhs, rhs);
  } else if (auto andIOp = dyn_cast<arith::AndIOp>(def)) {
    auto lhs = rematerInLayout(andIOp.getLhs(), enc, rewriter, cache);
    auto rhs = rematerInLayout(andIOp.getRhs(), enc, rewriter, cache);
    result = rewriter.create<arith::AndIOp>(andIOp.getLoc(), lhs, rhs);
  } else if (auto addPtrOp = dyn_cast<AddPtrOp>(def)) {
    // Both ptr and offset use the same encoding.
    auto newPtr = rematerInLayout(addPtrOp.getPtr(), enc, rewriter, cache);
    auto newOff = rematerInLayout(addPtrOp.getOffset(), enc, rewriter, cache);
    result =
        rewriter.create<AddPtrOp>(addPtrOp.getLoc(), newTy, newPtr, newOff);
  } else {
    // Unknown op: conservative fallback — insert a convert_layout.
    // This may or may not use shared memory; the goal is correctness.
    LLVM_DEBUG(llvm::dbgs()
               << "[PushConvert] unknown op in backward slice: "
               << def->getName() << " — falling back to convert_layout\n");
    result = rewriter.create<ConvertLayoutOp>(v.getLoc(), newTy, v);
  }

  cache[v] = result;
  return result;
}

struct PushConvertThroughLoadPattern
    : public OpRewritePattern<ConvertLayoutOp> {
  explicit PushConvertThroughLoadPattern(MLIRContext *context)
      : OpRewritePattern<ConvertLayoutOp>(context, /*benefit=*/20) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    // 1. Destination must be a LinearEncodingAttr.
    auto dstTy = cast<RankedTensorType>(cvtOp.getType());
    if (!isa<LinearEncodingAttr>(dstTy.getEncoding()))
      return failure();

    // 2. Source must come from a tt.load.
    auto loadOp = cvtOp.getSrc().getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();

    // 3. If the load has multiple uses we still apply the pattern: we create a
    //    new load in the target encoding for THIS convert_layout, and the
    //    original load stays alive for its other users.  Since A is small
    //    (≤4KB) the second load hits L1 cache and costs nothing.
    //    To limit duplication for large tensors, only proceed when the source
    //    type fits within a configurable threshold.
    {
      auto srcTy2 = cast<RankedTensorType>(loadOp.getType());
      int64_t numElements = 1;
      for (int64_t d : srcTy2.getShape())
        numElements *= d;
      auto elBits = srcTy2.getElementTypeBitWidth();
      int64_t sizeBytes = static_cast<unsigned>(numElements) * elBits / 8;
      constexpr int64_t kMaxDuplicateBytes = 8192; // 8 KB threshold
      if (!loadOp.getResult().hasOneUse() && sizeBytes > kMaxDuplicateBytes)
        return failure();
    }

    // 4. Only act when the conversion would require shared memory.
    auto srcTy = cast<RankedTensorType>(cvtOp.getSrc().getType());
    if (isAtMostWarpShuffle(srcTy, dstTy))
      return failure();

    // 5. Verify the linear layout surjectively covers every element of the
    //    tensor shape.  Loading in a non-surjective layout leaves some
    //    element positions unloaded — those registers hold garbage values
    //    and produce wrong dot-product inputs.
    //
    //    Formally, the rank of the layout's linear map must equal
    //    log2(M × K) so its image spans every valid element position.
    //    Without this check, the pattern could fire even when some elements
    //    are permanently inaccessible in the chosen linear layout.
    {
      auto linearEnc = cast<LinearEncodingAttr>(dstTy.getEncoding());
      if (!linearEnc.getLinearLayout().isSurjective()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[PushConvert] skip: linear layout is not surjective "
                      "over tensor shape — some elements would never be "
                      "loaded\n");
        return failure();
      }
    }

    // 5b. UB staging alignment check (requires 8-byte aligned
    //     source pointers for vector loads). For each register basis vector v
    //     and each non-innermost dimension d, the stride |v[d]| × sizeof(elem)
    //     must be divisible by kUBAlignBytes.
    //
    //     If |v[d]| × sizeof(elem) % 8 != 0, then the address stride between
    //     consecutive register slots depends on the runtime memory row width
    //     (e.g. N_stride, typically unconstrained), and alignment cannot be
    //     guaranteed.
    //
    {
      auto linearEnc = cast<LinearEncodingAttr>(dstTy.getEncoding());
      const LinearLayout &ll = linearEnc.getLinearLayout();
      MLIRContext *ctx = cvtOp.getContext();
      auto regKey = StringAttr::get(ctx, "register");
      auto regIt = ll.getBases().find(regKey);
      if (regIt != ll.getBases().end()) {
        unsigned rank = static_cast<unsigned>(dstTy.getRank());
        unsigned elemBytes = dstTy.getElementTypeBitWidth() / 8;
        constexpr unsigned kUBAlignBytes = 8;

        // Innermost (fastest-varying) memory dimension: use source
        // encoding's order[0] if available, else default to rank-1.
        unsigned innermostDim = rank - 1;
        if (auto blkEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding()))
          innermostDim = blkEnc.getOrder()[0];

        for (const auto &basisVec : regIt->second) {
          for (unsigned d = 0; d < rank && d < basisVec.size(); ++d) {
            if (d == innermostDim || basisVec[d] == 0)
              continue;
            unsigned strideContrib = static_cast<unsigned>(std::abs(
                                         static_cast<int32_t>(basisVec[d]))) *
                                     elemBytes;
            if (strideContrib % kUBAlignBytes != 0) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[PushConvert] skip: register basis[" << d
                         << "]=" << basisVec[d] << " × " << elemBytes
                         << "B not divisible by " << kUBAlignBytes
                         << " — UB staging alignment not guaranteed\n");
              return failure();
            }
          }
        }
      }
    }

    // 6. Rematerialize pointer, mask, and 'other' in the target encoding.
    Attribute dstEnc = dstTy.getEncoding();
    Location loc = loadOp.getLoc();

    DenseMap<Value, Value> cache;

    // Pointer type: tensor<MxKx!tt.ptr<elem>, srcEnc> to
    // tensor<MxKx!tt.ptr<elem>, dstEnc> The result type of the new LoadOp is
    // inferred from the pointer type.
    Value newPtrs = rematerInLayout(loadOp.getPtr(), dstEnc, rewriter, cache);
    Value newMask = loadOp.getMask() ? rematerInLayout(loadOp.getMask(), dstEnc,
                                                       rewriter, cache)
                                     : Value{};
    Value newOther =
        loadOp.getOther()
            ? rematerInLayout(loadOp.getOther(), dstEnc, rewriter, cache)
            : Value{};

    // 7. Emit load directly in the target layout — no shared memory.
    // Use the (ptr, mask, other, cache, evict, isVolatile) builder which
    // infers the result tensor type from the pointer element type.
    Value newLoad = rewriter.create<LoadOp>(
        loc, newPtrs, newMask, newOther, loadOp.getCache(), loadOp.getEvict(),
        loadOp.getIsVolatile());

    rewriter.replaceOp(cvtOp, newLoad);
    // The original load becomes dead and will be cleaned up by DCE.

    LLVM_DEBUG(llvm::dbgs()
               << "[PushConvert] replaced convert_layout(load) with "
                  "direct load in linear layout\n");
    return success();
  }
};

// ─── Pass ────────────────────────────────────────────────────────────────────

class ConvertDotInputToLinearLayoutPass
    : public impl::ConvertDotInputToLinearLayoutBase<
          ConvertDotInputToLinearLayoutPass> {
public:
  void runOnOperation() override {
    FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    // PushConvertThroughLoadPattern (benefit=20): fires first — pushes
    // convert_layout(load, #linear) through the load, rematerializing
    // pointer arithmetic in the target encoding so no shared memory is needed.
    patterns.add<PushConvertThroughLoadPattern>(func.getContext());
    // ConvertDotInputToLinearPattern (benefit=2): rewrites dot inputs to
    // #linear layout, creating the convert_layout that the above pattern
    // will then push through.
    patterns.add<ConvertDotInputToLinearPattern>(func.getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertDotInputToLinearLayoutPass() {
  return std::make_unique<ConvertDotInputToLinearLayoutPass>();
}

} // namespace triton
} // namespace bishengir