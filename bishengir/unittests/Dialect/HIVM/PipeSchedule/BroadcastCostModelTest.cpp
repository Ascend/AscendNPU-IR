//===- BroadcastCostModelTest.cpp - broadcast cost model unit tests
//--------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/BroadcastCostModel.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>

using namespace mlir;
using namespace mlir::hivm::costmodel;
namespace bc = mlir::hivm::costmodel::broadcast;

namespace {

class BroadcastCostModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    f16 = FloatType::getF16(context.get());
    bf16 = FloatType::getBF16(context.get());
    f32 = FloatType::getF32(context.get());
    f64 = FloatType::getF64(context.get());
    i1 = IntegerType::get(context.get(), 1);
    i8 = IntegerType::get(context.get(), 8);
    i16 = IntegerType::get(context.get(), 16);
    i32 = IntegerType::get(context.get(), 32);
    i64 = IntegerType::get(context.get(), 64);
  }

  /// `y` is the raw µs value of the model; the query reports ns.
  int64_t expectCost(double y) {
    double rounded = std::round(y * CostTimeUnitScale);
    return rounded < 1.0 ? 1 : static_cast<int64_t>(rounded);
  }

  /// Reference implementation of the bridging rule, written out from the
  /// two adjacent segments so the assertions document the rule itself.
  double bridgedSlope(const bc::Segment &prev, const bc::Segment &next,
                      int64_t v) {
    double t = static_cast<double>(v - prev.hi) /
               static_cast<double>(next.lo - prev.hi);
    return prev.slope + (next.slope - prev.slope) * t;
  }

  std::unique_ptr<MLIRContext> context;
  Type f16;
  Type bf16;
  Type f32;
  Type f64;
  Type i1;
  Type i8;
  Type i16;
  Type i32;
  Type i64;
};

// --- slopeAt: segment hits, bridging, range extension ------------------------

TEST_F(BroadcastCostModelTest, SlopeAtSegmentHit) {
  // Inside [2,6], the single-point [8,8], and the last [64,512].
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 4), 3.06e-3);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 6), 3.06e-3);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 8), 1.82e-3);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 300), 1.02e-3);
  // Segment boundaries belong to the segment that ends there.
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 512), 1.02e-3);
}

TEST_F(BroadcastCostModelTest, SlopeAtBridgesGaps) {
  // Gap (6, 8): midpoint 7 → the line from (6, 3.06e-3) to (8, 1.82e-3).
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 7),
                   bridgedSlope(bc::InnerSeg_I8[0], bc::InnerSeg_I8[1], 7));
  // Gap (48, 64): midpoint 56.
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 56),
                   bridgedSlope(bc::InnerSeg_I8[3], bc::InnerSeg_I8[4], 56));
  // Bridging is continuous at both gap endpoints (t = 0 and t = 1).
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 48), 1.68e-3);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 64), 1.02e-3);
  // A long gap (scalar1d i8, 288 → 1458) is bridged the same way.
  EXPECT_DOUBLE_EQ(
      bc::slopeAt(bc::Scalar1dSeg_I8, 800),
      bridgedSlope(bc::Scalar1dSeg_I8[1], bc::Scalar1dSeg_I8[2], 800));
}

TEST_F(BroadcastCostModelTest, SlopeAtExtendsRange) {
  // Below the first segment / above the last: the boundary piece extends.
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 1), 3.06e-3);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::InnerSeg_I8, 4096), 1.02e-3);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::Scalar1dSeg_F16, 1), 0.);
  EXPECT_DOUBLE_EQ(bc::slopeAt(bc::Scalar1dSeg_F16, 100000), 4.4e-6);
}

// --- inner / outer
// -------------------------------------------------------------

TEST_F(BroadcastCostModelTest, InnerEndToEnd) {
  // d1 = 4 → S = 3.06e-3; lat = Intercept_Inner + CV·S·d0 (the tables are
  // pre-CV calibrations, the slope term alone carries the CV factor).
  CostResult cost = bc::getInnerCost(i8, /*d0=*/100, /*d1=*/4);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Inner +
                                      VectorCVSlopeFactor * 3.06e-3 * 100));
  EXPECT_TRUE(cost.note.empty());

  // Bridged d1 = 7 (gap 6→8 of the i8 table).
  cost = bc::getInnerCost(i8, 100, 7);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Inner +
                                      VectorCVSlopeFactor *
                                          bridgedSlope(bc::InnerSeg_I8[0],
                                                       bc::InnerSeg_I8[1], 7) *
                                          100));
}

TEST_F(BroadcastCostModelTest, OuterEndToEnd) {
  // d1 = 128 → S = 6.06e-4 (last bf16 segment); lat = a + CV·S·d0.
  CostResult cost = bc::getOuterCost(bf16, /*d0=*/16, /*d1=*/128);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Outer +
                                      VectorCVSlopeFactor * 6.06e-4 * 16));
}

// --- middle
// ---------------------------------------------------------------------

TEST_F(BroadcastCostModelTest, MiddleEndToEnd) {
  // u = d0·d1 = 8·16 = 128 → S = 1.47e-3 (f16/bf16 table, [128,256]).
  CostResult cost = bc::getMiddleCost(f16, /*d0=*/8, /*d1=*/16);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Middle +
                                      VectorCVSlopeFactor * 1.47e-3 * 128));

  // u = 2·32 = 64 (i8 table, [32,64]).
  cost = bc::getMiddleCost(i8, 2, 32);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Middle +
                                      VectorCVSlopeFactor * 2.07e-3 * 64));

  // f16 and bf16 share the merged middle table.
  EXPECT_EQ(bc::getMiddleCost(f16, 8, 16).costTime,
            bc::getMiddleCost(bf16, 8, 16).costTime);
}

// --- scalar1d / scalar2d
// ---------------------------------------------------------

TEST_F(BroadcastCostModelTest, Scalar1dEndToEnd) {
  // i8, N = 100 → first segment slope 0 → the constant 0.0218 (CV on a
  // zero slope changes nothing).
  CostResult cost = bc::getScalar1dCost(i8, 100);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Scalar1d));

  // i32, N = 100 → slope 1.7e-6.
  cost = bc::getScalar1dCost(i32, 100);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Scalar1d +
                                      VectorCVSlopeFactor * 1.7e-6 * 100));

  // Single-point segment [7386, 7386] (f16 tail).
  cost = bc::getScalar1dCost(f16, 7386);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Scalar1d +
                                      VectorCVSlopeFactor * 4.4e-6 * 7386));
}

TEST_F(BroadcastCostModelTest, Scalar1dOutlierPeakIsBridged) {
  // The source doc's N≈170~190 isolated-peak noise segments are dropped:
  // i32 at N = 182 sits in the widened gap [138,170] → [288,7386] and is
  // bridged — no 2.9e-3 spike, just the gentle interpolation.
  CostResult cost = bc::getScalar1dCost(i32, 182);
  ASSERT_NE(cost.costTime, -1);
  double s = bridgedSlope(bc::Scalar1dSeg_I32[2], bc::Scalar1dSeg_I32[3], 182);
  EXPECT_LT(s, 1e-5); // vs. the dropped outlier's 2.9e-3
  EXPECT_EQ(cost.costTime,
            expectCost(bc::Intercept_Scalar1d + VectorCVSlopeFactor * s * 182));

  // bf16 at N = 174 (dropped [170,178] → 1.1e-2 outlier): same bridge.
  cost = bc::getScalar1dCost(bf16, 174);
  ASSERT_NE(cost.costTime, -1);
  s = bridgedSlope(bc::Scalar1dSeg_BF16[1], bc::Scalar1dSeg_BF16[2], 174);
  EXPECT_LT(s, 1e-5); // vs. the dropped outlier's 1.1e-2
  EXPECT_EQ(cost.costTime,
            expectCost(bc::Intercept_Scalar1d + VectorCVSlopeFactor * s * 174));
}

TEST_F(BroadcastCostModelTest, Scalar2dEndToEnd) {
  // f16, N = 8·8 = 64 → first segment slope 0 → constant 0.0215.
  CostResult cost = bc::getScalar2dCost(f16, 8, 8);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Scalar2d));

  // i32/f32, N = 32·32 = 1024 → last segment slope 9.5e-6.
  cost = bc::getScalar2dCost(i32, 32, 32);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Scalar2d +
                                      VectorCVSlopeFactor * 9.5e-6 * 1024));

  // i32 and f32 share every merged table except scalar1d.
  EXPECT_EQ(bc::getScalar2dCost(i32, 32, 32).costTime,
            bc::getScalar2dCost(f32, 32, 32).costTime);
  EXPECT_NE(bc::getScalar1dCost(i32, 100).costTime,
            bc::getScalar1dCost(f32, 100).costTime);
}

TEST_F(BroadcastCostModelTest, Scalar2dOutlierIsBridged) {
  // f16 N = 10·8 = 80 (the dropped [80,96] → -7.1e-3 outlier): bridged
  // between [4,64] (slope 0) and [128,192] (slope 9.6e-6) — positive and
  // small, never negative.
  CostResult cost = bc::getScalar2dCost(f16, 10, 8);
  ASSERT_NE(cost.costTime, -1);
  double s = bridgedSlope(bc::Scalar2dSeg_F16[0], bc::Scalar2dSeg_F16[1], 80);
  EXPECT_GT(s, 0.);
  EXPECT_LT(s, 9.6e-6);
  EXPECT_EQ(cost.costTime,
            expectCost(bc::Intercept_Scalar2d + VectorCVSlopeFactor * s * 80));
}

TEST_F(BroadcastCostModelTest, CVFactorHalvesSlopeOnly) {
  // The tables are pre-CV calibrations; the applied model is
  // a + CV·S·rate with the intercept untouched — the broadcast
  // counterpart of the generic chain's uniform CV rule.  A slope-free
  // shape sits exactly on the intercept...
  CostResult cost = bc::getScalar1dCost(i8, 100); // first i8 segment: S = 0
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Scalar1d));

  // ...while a slope-heavy shape bills half the raw slope term on top of
  // the full intercept (inner i8, d1 = 4 → S = 3.06e-3).
  cost = bc::getInnerCost(i8, /*d0=*/100000, /*d1=*/4);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(bc::Intercept_Inner +
                                      VectorCVSlopeFactor * 3.06e-3 * 100000));
}

// --- error sentinels --------------------------------------------------------

TEST_F(BroadcastCostModelTest, UncalibratedDtypeFails) {
  // i1 / i16 / i64 / f64 have no broadcast calibration.
  for (Type ty : {i1, i16, i64, f64}) {
    CostResult cost = bc::getInnerCost(ty, 100, 4);
    EXPECT_EQ(cost.costTime, -1);
    EXPECT_NE(cost.note.find("broadcast inner dtype not calibrated"),
              std::string::npos)
        << "dtype " << cost.note;
  }
  CostResult cost = bc::getScalar1dCost(f64, 100);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("broadcast scalar1d dtype not calibrated"),
            std::string::npos);
}

TEST_F(BroadcastCostModelTest, BadDimsFail) {
  CostResult cost = bc::getInnerCost(i8, 0, 4);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("broadcast inner d0 = 0, expect > 0"),
            std::string::npos);

  cost = bc::getScalar1dCost(i8, -3);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("expect > 0"), std::string::npos);
}

TEST_F(BroadcastCostModelTest, DimProductOverflowFails) {
  // d0·d1 overflows int64 → -1, not a wrapped latency.
  CostResult cost = bc::getScalar2dCost(i8, INT64_MAX / 2, 4);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("broadcast scalar2d d0*d1 overflows"),
            std::string::npos);

  cost = bc::getMiddleCost(i8, INT64_MAX / 2, 4);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("broadcast middle d0*d1 overflows"),
            std::string::npos);
}

} // namespace
