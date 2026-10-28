#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::triton;

using mlir::triton::gpu::logBankConflictsLdSt;
using mlir::triton::gpu::optimalSwizzling;
using mlir::triton::gpu::optimalSwizzlingLdSt;
using mlir::triton::gpu::SharedMemoryBankInfo;

namespace {

SmallVector<int32_t> flatten(const LinearLayout &ll, StringAttr dim) {
  assert(ll.hasInDim(dim) && "in dim must exist");
  SmallVector<int32_t> vec;
  for (const auto &basis : ll.getBases().lookup(dim)) {
    assert(basis.size() == 1 && "basis must be a single int32_t");
    vec.push_back(basis[0]);
  }
  return vec;
}

// Enumerate emitted accesses independently of the rank-based cost model. Reads
// of different elements in the same aligned bank word are multicast.
int enumerateBankConflicts(const LinearLayout &layout, const LinearLayout &smem,
                           int32_t bitwidth,
                           const SharedMemoryBankInfo &bankInfo,
                           int vectorSize = 0) {
  auto transfer = layout.invertAndCompose(smem.flattenIns()).flattenOuts();
  auto *ctx = layout.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef name) { return StringAttr::get(ctx, name); };
  if (!vectorSize)
    vectorSize = std::min(transfer.getNumConsecutiveInOut(), 128 / bitwidth);
  int transactionLanes = std::min(
      32, static_cast<int>(bankInfo.numBanks * bankInfo.bankWidthBits) /
              (vectorSize * bitwidth));
  int maxConflict = 1;
  for (int warp = 0; warp < layout.getInDimSize(S("warp")); ++warp) {
    for (int reg = 0; reg < layout.getInDimSize(S("register"));
         reg += vectorSize) {
      for (int firstLane = 0; firstLane < layout.getInDimSize(S("lane"));
           firstLane += transactionLanes) {
        std::map<int, std::set<int>> bankWords;
        for (int lane = firstLane;
             lane < std::min(firstLane + transactionLanes,
                             layout.getInDimSize(S("lane")));
             ++lane) {
          for (int element = 0; element < vectorSize; ++element) {
            int offset = transfer
                             .apply({{S("register"), reg + element},
                                     {S("lane"), lane},
                                     {S("warp"), warp}})[0]
                             .second;
            int word = offset * bitwidth / bankInfo.bankWidthBits;
            bankWords[word % bankInfo.numBanks].insert(word);
          }
        }
        for (const auto &bank : bankWords)
          maxConflict =
              std::max(maxConflict, static_cast<int>(bank.second.size()));
      }
    }
  }
  return maxConflict;
}

class SwizzleTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

// ——— Tests ———

TEST_F(SwizzleTest, Test128x128Float8Transpose) {
  // 128x128 float8 matrix transpose
  LinearLayout matrix(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
       {S("lane"), {{0, 16}, {0, 32}, {0, 64}, {4, 0}, {8, 0}}},
       {S("warp"), {{16, 0}, {32, 0}, {64, 0}}}},
      {{S("dim0"), 128}, {S("dim1"), 128}}, /*requireSurjective=*/true);
  auto matrix_t = transposeLinearLayout(matrix, {1, 0});

  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/8);
  auto [r, w] = logBankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/8);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test16x16Bf16BlockedMma) {
  // 16×16 bf16 MMA
  LinearLayout blocked({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                        {S("lane"), {{0, 8}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                        {S("warp"), {}}},
                       {{S("dim0"), 16}, {S("dim1"), 16}},
                       /*requireSurjective=*/true);
  LinearLayout mma({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}}},
                   {{S("dim0"), 16}, {S("dim1"), 16}},
                   /*requireSurjective=*/true);

  auto smem = optimalSwizzlingLdSt(blocked, mma, /*bitwidth=*/16);
  auto [r, w] = logBankConflictsLdSt(blocked, mma, smem, /*bitwidth=*/16);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test16x256U4Mma) {
  // 16×256 u4 MMA
  LinearLayout blocked(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}, {8, 0}}},
       {S("lane"), {{0, 32}, {0, 64}, {0, 128}, {1, 0}, {2, 0}}},
       {S("warp"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);
  LinearLayout mma(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {0, 128}}},
       {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);

  auto smem = optimalSwizzlingLdSt(blocked, mma, /*bitwidth=*/4);
  auto [r, w] = logBankConflictsLdSt(blocked, mma, smem, /*bitwidth=*/4);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test32x16F32Transpose) {
  // 32×16 f32 transpose
  LinearLayout matrix({{S("register"), {{4, 0}, {8, 0}, {16, 0}}},
                       {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                       {S("warp"), {{2, 0}}}},
                      {{S("dim0"), 32}, {S("dim1"), 16}},
                      /*requireSurjective=*/true);
  LinearLayout matrix_t({{S("register"), {{0, 2}, {0, 4}, {0, 8}}},
                         {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                         {S("warp"), {{0, 1}}}},
                        {{S("dim0"), 32}, {S("dim1"), 16}},
                        /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/32);
  auto [r, w] = logBankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/32);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test128x128F16Transpose) {
  LinearLayout matrix(
      {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 32}, {0, 64}}},
       {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 1}}},
       {S("warp"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout matrix_t(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}}},
       {S("warp"), {{2, 0}, {4, 0}, {8, 0}, {16, 0}}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/16);
  auto [r, w] = logBankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/16);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, AscendAlignedWordMulticast) {
  const SharedMemoryBankInfo ascend{16, 64};
  const SharedMemoryBankInfo generic{32, 32};
  LinearLayout contiguous({{S("register"), {{32}}},
                           {S("lane"), {{1}, {2}, {4}, {8}, {16}}},
                           {S("warp"), {}}},
                          {{S("dim0"), 64}}, /*requireSurjective=*/true);
  auto smem = LinearLayout({{S("vector"), {}},
                            {S("bank"), {{1}, {2}, {4}, {8}, {16}}},
                            {S("segment"), {{32}}},
                            {S("reps"), {}}},
                           {{S("dim0"), 64}}, /*requireSurjective=*/true);

  // Thirty-two adjacent f32 accesses touch only sixteen aligned 8B words.
  EXPECT_EQ(enumerateBankConflicts(contiguous, smem, 32, ascend), 1);
  EXPECT_EQ(logBankConflictsLdSt(contiguous, contiguous, smem, 32, ascend),
            std::make_pair(0, 0));

  // Byte address = lane * 4 + (lane % 2) * 128. The two halves of each
  // bank now belong to distinct words, exposing the difference between models.
  LinearLayout bit7({{S("register"), {{32}}},
                     {S("lane"), {{33}, {2}, {4}, {8}, {16}}},
                     {S("warp"), {}}},
                    {{S("dim0"), 64}}, /*requireSurjective=*/true);
  EXPECT_EQ(enumerateBankConflicts(bit7, smem, 32, generic), 1);
  EXPECT_EQ(enumerateBankConflicts(bit7, smem, 32, ascend), 2);
  EXPECT_EQ(logBankConflictsLdSt(bit7, bit7, smem, 32, generic),
            std::make_pair(0, 0));
  EXPECT_EQ(logBankConflictsLdSt(bit7, bit7, smem, 32, ascend),
            std::make_pair(1, 1));
}

TEST_F(SwizzleTest, AscendBankRefinementImprovesExactReadConflicts) {
  const SharedMemoryBankInfo ascend{16, 64};
  const SharedMemoryBankInfo generic{32, 32};
  LinearLayout src({{S("register"), {{1}}},
                    {S("lane"), {{4}, {8}, {16}, {32}, {64}}},
                    {S("warp"), {{2}}}},
                   {{S("dim0"), 128}}, /*requireSurjective=*/true);
  LinearLayout dst({{S("register"), {{2}}},
                    {S("lane"), {{1}, {8}, {16}, {32}, {64}}},
                    {S("warp"), {{4}}}},
                   {{S("dim0"), 128}}, /*requireSurjective=*/true);
  auto baseline = optimalSwizzlingLdSt(src, dst, 32);
  auto refined = optimalSwizzlingLdSt(src, dst, 32, ascend);
  EXPECT_EQ(optimalSwizzlingLdSt(src, dst, 32, generic), baseline);
  EXPECT_EQ(enumerateBankConflicts(src, baseline, 32, ascend), 1);
  EXPECT_EQ(enumerateBankConflicts(dst, baseline, 32, ascend), 2);
  EXPECT_EQ(enumerateBankConflicts(src, refined, 32, ascend), 1);
  EXPECT_EQ(enumerateBankConflicts(dst, refined, 32, ascend), 1);
  EXPECT_EQ(logBankConflictsLdSt(src, dst, baseline, 32, ascend),
            std::make_pair(1, 0));
  EXPECT_EQ(logBankConflictsLdSt(src, dst, refined, 32, ascend),
            std::make_pair(0, 0));

  // Preserve allocation size, repetition structure, and emitted vector widths.
  for (auto dim : baseline.getInDimNames())
    EXPECT_EQ(refined.getInDimSize(dim), baseline.getInDimSize(dim));
  for (auto dim : {S("vector"), S("segment"), S("reps")})
    EXPECT_EQ(refined.getBases().lookup(dim), baseline.getBases().lookup(dim));
  for (const auto &layout : {src, dst}) {
    EXPECT_EQ(
        layout.invertAndCompose(refined.flattenIns()).getNumConsecutiveInOut(),
        layout.invertAndCompose(baseline.flattenIns())
            .getNumConsecutiveInOut());
  }

  // Every shared-memory offset still maps to exactly one logical element.
  auto flat = refined.flattenIns().flattenOuts();
  auto inverse = flat.invert();
  std::set<int> logicalElements;
  for (int offset = 0; offset < 128; ++offset) {
    int logical = flat.apply({{S("vector"), offset}})[0].second;
    EXPECT_EQ(inverse.apply({{S("dim0"), logical}})[0].second, offset);
    logicalElements.insert(logical);
  }
  EXPECT_EQ(logicalElements.size(), 128u);
}

TEST_F(SwizzleTest, AscendBankRefinementImprovesExactWriteConflicts) {
  const SharedMemoryBankInfo ascend{16, 64};
  LinearLayout src({{S("register"), {{32}}},
                    {S("lane"), {{1}, {2}, {4}, {8}, {16}}},
                    {S("warp"), {}}},
                   {{S("dim0"), 64}}, /*requireSurjective=*/true);
  LinearLayout dst({{S("register"), {{1}}},
                    {S("lane"), {{2}, {4}, {8}, {16}, {32}}},
                    {S("warp"), {}}},
                   {{S("dim0"), 64}}, /*requireSurjective=*/true);
  auto baseline = optimalSwizzlingLdSt(src, dst, 32);
  auto refined = optimalSwizzlingLdSt(src, dst, 32, ascend);
  // The segment basis mixes the two register assignments. A lane-dependent
  // low address bit keeps the baseline store scalar despite register stride 1.
  EXPECT_EQ(
      src.invertAndCompose(baseline.flattenIns()).getNumConsecutiveInOut(), 1);
  EXPECT_EQ(
      dst.invertAndCompose(baseline.flattenIns()).getNumConsecutiveInOut(), 1);
  EXPECT_EQ(enumerateBankConflicts(src, baseline, 32, ascend), 2);
  EXPECT_EQ(enumerateBankConflicts(dst, baseline, 32, ascend), 1);
  EXPECT_EQ(enumerateBankConflicts(src, refined, 32, ascend), 1);
  EXPECT_EQ(enumerateBankConflicts(dst, refined, 32, ascend), 1);
  for (const auto &layout : {src, dst})
    EXPECT_EQ(
        layout.invertAndCompose(refined.flattenIns()).getNumConsecutiveInOut(),
        layout.invertAndCompose(baseline.flattenIns())
            .getNumConsecutiveInOut());
  auto estimated = logBankConflictsLdSt(src, dst, refined, 32, ascend);
  EXPECT_EQ(1 << estimated.first,
            enumerateBankConflicts(dst, refined, 32, ascend));
  EXPECT_EQ(1 << estimated.second,
            enumerateBankConflicts(src, refined, 32, ascend));
}

TEST_F(SwizzleTest,
       AscendVectorTransactionsWithRegisterPermutationAndRepetitions) {
  const SharedMemoryBankInfo ascend{16, 64};
  LinearLayout layout({{S("register"), {{256}, {2}, {512}, {1}}},
                       {S("lane"), {{4}, {8}, {16}, {32}, {64}}},
                       {S("warp"), {{128}}}},
                      {{S("dim0"), 1024}}, /*requireSurjective=*/true);
  LinearLayout smem({{S("vector"), {{1}, {2}}},
                     {S("bank"), {{4}, {8}, {16}}},
                     {S("segment"), {{32}, {64}, {128}}},
                     {S("reps"), {{256}, {512}}}},
                    {{S("dim0"), 1024}}, /*requireSurjective=*/true);
  EXPECT_EQ(layout.invertAndCompose(smem.flattenIns()).getNumConsecutiveInOut(),
            1);

  // The emitted schedule moves the two repetition bits out of the scratch
  // address and permutes the remaining register bits to emit vector<4xf32>.
  // Each 128B transaction therefore covers eight lanes, rather than all 32.
  LinearLayout emitted({{S("register"), {{1}, {2}}},
                        {S("lane"), {{4}, {8}, {16}, {32}, {64}}},
                        {S("warp"), {{128}}}},
                       {{S("offset"), 256}}, /*requireSurjective=*/true);
  auto identity = LinearLayout::identity1D(256, S("offset"), S("offset"));
  EXPECT_EQ(enumerateBankConflicts(emitted, identity, 32, ascend, 4), 1);
  EXPECT_EQ(enumerateBankConflicts(emitted, identity, 32, ascend, 1), 4);
  EXPECT_EQ(logBankConflictsLdSt(layout, layout, smem, 32, ascend),
            std::make_pair(0, 0));
}

TEST_F(SwizzleTest, AscendBankModelMatchesEnumeratedBitPermutations) {
  const SharedMemoryBankInfo ascend{16, 64};
  auto makeLayout = [&](int regBit, int warpBit, bool reverseLanes) {
    std::vector<std::vector<int32_t>> lanes;
    for (int bit = 0; bit < 7; ++bit)
      if (bit != regBit && bit != warpBit)
        lanes.push_back({1 << bit});
    if (reverseLanes)
      std::reverse(lanes.begin(), lanes.end());
    return LinearLayout({{S("register"), {{1 << regBit}}},
                         {S("lane"), lanes},
                         {S("warp"), {{1 << warpBit}}}},
                        {{S("dim0"), 128}}, /*requireSurjective=*/true);
  };

  // One register bit limits emitted vectorization to scalar or two elements,
  // with no register permutations or scratch repetitions. This makes direct
  // enumeration independent of the lowering's schedule construction. Cover
  // every source register/warp assignment and both lane-bit orders.
  for (int sample = 0; sample < 42; ++sample) {
    SCOPED_TRACE(sample);
    int srcReg = sample % 7;
    int srcWarp = (srcReg + 1 + sample / 7) % 7;
    int dstReg = srcWarp;
    int dstWarp = (srcWarp + 3) % 7;
    auto src = makeLayout(srcReg, srcWarp, sample % 2);
    auto dst = makeLayout(dstReg, dstWarp, !(sample % 2));
    auto baseline = optimalSwizzlingLdSt(src, dst, 32);
    auto refined = optimalSwizzlingLdSt(src, dst, 32, ascend);
    ASSERT_EQ(baseline.getInDimSize(S("reps")), 1);
    ASSERT_EQ(refined.getInDimSize(S("reps")), 1);

    auto oldRead = enumerateBankConflicts(dst, baseline, 32, ascend);
    auto oldWrite = enumerateBankConflicts(src, baseline, 32, ascend);
    auto newRead = enumerateBankConflicts(dst, refined, 32, ascend);
    auto newWrite = enumerateBankConflicts(src, refined, 32, ascend);
    auto oldEstimate = logBankConflictsLdSt(src, dst, baseline, 32, ascend);
    auto newEstimate = logBankConflictsLdSt(src, dst, refined, 32, ascend);
    EXPECT_EQ(1 << oldEstimate.first, oldRead);
    EXPECT_EQ(1 << oldEstimate.second, oldWrite);
    EXPECT_EQ(1 << newEstimate.first, newRead);
    EXPECT_EQ(1 << newEstimate.second, newWrite);
    EXPECT_LE(newRead, oldRead);
    EXPECT_LE(newWrite, oldWrite);
    for (auto dim : baseline.getInDimNames())
      EXPECT_EQ(refined.getInDimSize(dim), baseline.getInDimSize(dim));
    for (const auto &layout : {src, dst})
      EXPECT_EQ(layout.invertAndCompose(refined.flattenIns())
                    .getNumConsecutiveInOut(),
                layout.invertAndCompose(baseline.flattenIns())
                    .getNumConsecutiveInOut());

    auto flat = refined.flattenIns().flattenOuts();
    auto inverse = flat.invert();
    std::set<int> logicalElements;
    for (int offset = 0; offset < 128; ++offset) {
      int logical = flat.apply({{S("vector"), offset}})[0].second;
      EXPECT_EQ(inverse.apply({{S("dim0"), logical}})[0].second, offset);
      logicalElements.insert(logical);
    }
    EXPECT_EQ(logicalElements.size(), 128u);
  }
}

} // namespace

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
