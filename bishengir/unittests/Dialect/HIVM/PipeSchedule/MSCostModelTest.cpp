//===- MSCostModelTest.cpp - MS cost model unit tests --------------------===//
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

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostModel.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/BroadcastCostModel.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostParams.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationState.h"
#include "gtest/gtest.h"

#include <cmath>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::hivm::costmodel;

namespace {

class MSCostModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    // Bare-named ops ("memref.alloc", "hivm.hir.vcast", ...) are created
    // via Operation::create without their dialects being registered — the
    // cost model only reads the op NAME, never the op interface.
    context->allowUnregisteredDialects();
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

  ShapedOperandDesc desc(std::initializer_list<int64_t> shape, Type elemTy) {
    return ShapedOperandDesc(SmallVector<int64_t, ShapeInlineCapacity>(shape),
                             elemTy);
  }

  /// `y` is the raw μs value of the formula; the model reports ns.
  int64_t expectCost(double y) {
    // Mirror the model's normalization: max(1, round(y · CostTimeUnitScale)).
    double rounded = std::round(y * CostTimeUnitScale);
    return rounded < 1.0 ? 1 : static_cast<int64_t>(rounded);
  }

  /// Bare op holding only a name — exactly what the cost model reads.
  Operation *makeOp(StringRef name) {
    OperationState state(UnknownLoc::get(context.get()), name);
    return Operation::create(state);
  }

  /// Slope multiplier of one VCastDirectPairList edge; mirrors the table
  /// constants so expected path sums are bit-identical to the model's
  /// accumulation (same constants, same left-to-right addition order).
  double vcastEdgeSlope(StringRef src, StringRef dst) {
    for (const auto &pair : VCastDirectPairList)
      if (pair.src == src && pair.dst == dst)
        return pair.slope;
    ADD_FAILURE() << "missing vcast edge " << src.str() << "->" << dst.str();
    return 0.;
  }

  /// Per-op y for a NON-vcast op, mirroring getVectorGroupCost's chain
  /// (same constants, same multiplication/association order — the
  /// comparisons below are then exact, not tolerance-based).
  double vectorOpY(std::optional<Type> dtype, size_t nInput, int64_t x,
                   CostMultiplier opTypeMult) {
    CostMultiplier dtypeMult{1., 1.};
    double inCountSlope = 1.;
    if (nInput > 0 && dtype) {
      dtypeMult = *vectorDtypeMultiplierFor(*dtype);
      inCountSlope = (static_cast<double>(nInput) - 1.) *
                         *vectorInputCountCoeffFor(*dtype) +
                     1.;
    }
    double slopeMult = dtypeMult.slope * inCountSlope *
                       VectorShapeMultiplier.slope * opTypeMult.slope;
    double interceptMult = dtypeMult.intercept *
                           VectorShapeMultiplier.intercept *
                           opTypeMult.intercept;
    double aEff = VectorBaseParam.a * slopeMult * VectorCVSlopeFactor;
    double bEff = VectorBaseParam.b * interceptMult;
    return aEff * static_cast<double>(x) + bEff;
  }

  /// Per-op y for a vcast op given its pair multiplier (CV factor
  /// applied here, exactly as in the model's step ⑥).
  double vcastOpY(int64_t x, CostMultiplier pair) {
    double aEff = VectorBaseParam.a * pair.slope * VectorCVSlopeFactor;
    double bEff = VectorBaseParam.b * pair.intercept;
    return aEff * static_cast<double>(x) + bEff;
  }

  /// Per-op y for a varange op with output element count n (CV factor on
  /// the slope, exactly as in the model's step ②).
  double varangeOpY(int64_t n) {
    double aEff = VarangeParam.a * VectorCVSlopeFactor;
    return aEff * static_cast<double>(n) + VarangeParam.b;
  }

  /// Group-level cost of a single-op vbrc group: the broadcast query
  /// reports ns, the group sums µs and applies the VF factor — this
  /// mirrors getVectorGroupCost's steps ③ + ⑧ in the same order, so the
  /// comparison is exact for any VF value.
  int64_t vbrcGroupCost(const CostResult &r) {
    return expectCost(static_cast<double>(r.costTime) / CostTimeUnitScale *
                      VectorVF_Factor);
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

// --- ShapedOperandDesc::getTotalElements -----------------------------------

TEST_F(MSCostModelTest, TotalElementsArbitraryRanks) {
  // 1-D.
  ASSERT_TRUE(succeeded(desc({128}, f16).getTotalElements()));
  EXPECT_EQ(*desc({128}, f16).getTotalElements(), 128);
  // 2-D.
  EXPECT_EQ(*desc({128, 128}, f16).getTotalElements(), 16384);
  // 4-D fractal.
  EXPECT_EQ(*desc({2, 5, 16, 16}, f16).getTotalElements(), 2560);
  // Rank beyond ShapeInlineCapacity (heap spill) stays correct.
  EXPECT_EQ(*desc({2, 3, 4, 5, 6}, f16).getTotalElements(), 720);
  // 0-D scalar tensor = 1 element.
  EXPECT_EQ(*desc({}, f16).getTotalElements(), 1);
}

TEST_F(MSCostModelTest, TotalElementsRejectsDynamicAndNonPositive) {
  EXPECT_TRUE(failed(desc({ShapedType::kDynamic, 4}, f16).getTotalElements()));
  EXPECT_TRUE(failed(desc({4, ShapedType::kDynamic}, f16).getTotalElements()));
  EXPECT_TRUE(failed(desc({0}, f16).getTotalElements()));
  EXPECT_TRUE(failed(desc({-5, 2}, f16).getTotalElements()));
}

// --- ShapedOperandDesc::getTotalBytes ----------------------------------------

TEST_F(MSCostModelTest, TotalBytesPerDtype) {
  EXPECT_EQ(*desc({64, 64}, f16).getTotalBytes(), 8192);
  EXPECT_EQ(*desc({64, 64}, bf16).getTotalBytes(), 8192);
  EXPECT_EQ(*desc({64, 64}, i8).getTotalBytes(), 4096);
  EXPECT_EQ(*desc({64, 64}, i1).getTotalBytes(), 4096);
  EXPECT_EQ(*desc({64, 64}, f32).getTotalBytes(), 16384);
  EXPECT_EQ(*desc({64, 64}, i32).getTotalBytes(), 16384);
  EXPECT_EQ(*desc({64, 64}, f64).getTotalBytes(), 32768);
  // 0-D scalar: one element's bytes.
  EXPECT_EQ(*desc({}, f16).getTotalBytes(), 2);
}

TEST_F(MSCostModelTest, TotalBytesRejectsUnknownDtype) {
  // A non-integer non-float type is not in the dtype table.
  EXPECT_TRUE(
      failed(desc({4, 4}, NoneType::get(context.get())).getTotalBytes()));
}

// --- dtype helpers -----------------------------------------------------------

TEST_F(MSCostModelTest, BytesOfDtypeAndIsAcceptedDtype) {
  EXPECT_EQ(*bytesOfDtype(f16), 2);
  EXPECT_EQ(*bytesOfDtype(bf16), 2);
  EXPECT_EQ(*bytesOfDtype(i1), 1);
  EXPECT_EQ(*bytesOfDtype(i8), 1);
  EXPECT_EQ(*bytesOfDtype(i32), 4);
  EXPECT_EQ(*bytesOfDtype(f32), 4);
  EXPECT_EQ(*bytesOfDtype(f64), 8);
  EXPECT_TRUE(failed(bytesOfDtype(NoneType::get(context.get()))));

  EXPECT_TRUE(isAcceptedDtype(f16));
  EXPECT_TRUE(isAcceptedDtype(bf16));
  EXPECT_TRUE(isAcceptedDtype(i8));
  EXPECT_TRUE(isAcceptedDtype(f32));
  EXPECT_FALSE(isAcceptedDtype(NoneType::get(context.get())));
}

// --- Cube --------------------------------------------------------------------

TEST_F(MSCostModelTest, CubeF16Formula) {
  MSCostModel model;
  CubeCostInput input;
  // lhs [0]=sk, [1]=sm; rhs [0]=sn, [1]=sk → x = sm·sn·sk = 5*3*2 = 30.
  input.lhs = desc({2, 5, 16, 16}, f16);
  input.rhs = desc({3, 2, 16, 16}, f16);
  CostResult cost = model.getCubeCost(input);
  ASSERT_NE(cost.costTime, -1);
  // Effective param = CubeBaseParam × CubeDtypeMultiplier_F16.
  EXPECT_EQ(cost.costTime,
            expectCost(CubeBaseParam.a * CubeDtypeMultiplier_F16.slope * 30 +
                       CubeBaseParam.b * CubeDtypeMultiplier_F16.intercept));
  EXPECT_TRUE(cost.note.empty());
}

TEST_F(MSCostModelTest, CubeBf16AndI8ShareF16Params) {
  MSCostModel model;
  CubeCostInput bf;
  bf.lhs = desc({4, 8, 16, 16}, bf16);
  bf.rhs = desc({2, 4, 16, 16}, bf16);
  CostResult bfCost = model.getCubeCost(bf);

  CubeCostInput i;
  i.lhs = desc({4, 8, 16, 32}, i8);
  i.rhs = desc({2, 4, 16, 32}, i8);
  CostResult iCost = model.getCubeCost(i);

  // Same (sk, sm, sn) → same x; the i8/f16/bf16 dtype multipliers are all
  // {1., 1.} today, so the costs agree despite different dtypes.
  ASSERT_NE(bfCost.costTime, -1);
  ASSERT_NE(iCost.costTime, -1);
  EXPECT_EQ(bfCost.costTime, iCost.costTime);
}

TEST_F(MSCostModelTest, CubeF32SeparateParams) {
  MSCostModel model;
  CubeCostInput input;
  input.lhs = desc({2, 5, 16, 8}, f32);
  input.rhs = desc({3, 2, 16, 8}, f32);
  CostResult cost = model.getCubeCost(input);
  ASSERT_NE(cost.costTime, -1);
  // Effective param = CubeBaseParam × CubeDtypeMultiplier_F32 (slope 8.).
  EXPECT_EQ(cost.costTime,
            expectCost(CubeBaseParam.a * CubeDtypeMultiplier_F32.slope * 30 +
                       CubeBaseParam.b * CubeDtypeMultiplier_F32.intercept));
}

TEST_F(MSCostModelTest, CubeErrorSentinels) {
  MSCostModel model;

  // Rank < 2.
  CubeCostInput rank;
  rank.lhs = desc({16}, f16);
  rank.rhs = desc({3, 2}, f16);
  CostResult cost = model.getCubeCost(rank);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("cube lhs shape rank < 2"), std::string::npos);

  // Dynamic dim among the first two.
  CubeCostInput dyn;
  dyn.lhs = desc({2, ShapedType::kDynamic, 16, 16}, f16);
  dyn.rhs = desc({3, 2, 16, 16}, f16);
  cost = model.getCubeCost(dyn);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("cube lhs dim 1 is dynamic"), std::string::npos);

  // Non-positive dim.
  CubeCostInput neg;
  neg.lhs = desc({-5, 5, 16, 16}, f16);
  neg.rhs = desc({3, 2, 16, 16}, f16);
  cost = model.getCubeCost(neg);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("expect > 0"), std::string::npos);

  // Unsupported dtype.
  CubeCostInput dtype;
  dtype.lhs = desc({2, 5, 16, 16}, f64);
  dtype.rhs = desc({3, 2, 16, 16}, f64);
  cost = model.getCubeCost(dtype);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("cube dtype not supported"), std::string::npos);
  EXPECT_NE(cost.note.find("f64"), std::string::npos);
}

// --- Vector
// -------------------------------------------------------------------

TEST_F(MSCostModelTest, VectorGroupSumAndVfFactor) {
  MSCostModel model;
  VectorGroupCostInput input;
  // Two ops, op = nullptr → "<unknown>" → fallback op-type multiplier
  // {1., 1.}; that also proves fallback ops land in the note.
  VectorOpDesc op0;
  op0.inputs.push_back(desc({128, 128}, f16));
  VectorOpDesc op1;
  op1.inputs.push_back(desc({128, 128}, f16));
  op1.inputs.push_back(desc({64, 128}, f16));
  input.ops.push_back(std::move(op0));
  input.ops.push_back(std::move(op1));

  // v2 semantics: x = max input element count (not the sum); the
  // intercept counts once per op (not once per input).
  double sum = vectorOpY(f16, 1, 128 * 128, VectorOpTypeMultiplier_Fallback) +
               vectorOpY(f16, 2, 128 * 128, VectorOpTypeMultiplier_Fallback);
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(sum * VectorVF_Factor));
  // The nullptr op resolves to "<unknown>" — a fallback op, noted.
  EXPECT_NE(cost.note.find("<unknown>"), std::string::npos);
}

TEST_F(MSCostModelTest, VectorZeroInputsOpCountsOnlyB) {
  MSCostModel model;
  // An op with no shaped inputs: n_input = 0 → in-count multiplier 1. and
  // x = 0, so only the per-op intercept term (b_eff) remains; a group of
  // only such ops still yields max(1, round(b_eff · n · VF)) ≥ 1.
  VectorGroupCostInput input;
  VectorOpDesc op0;
  input.ops.push_back(std::move(op0));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  // No inputs → dtype/shape multipliers default to {1., 1.}; op type
  // "<unknown>" is a fallback {1., 1.} — so b_eff = VectorBaseParam.b.
  EXPECT_EQ(cost.costTime,
            expectCost(
                vectorOpY(std::nullopt, 0, 0, VectorOpTypeMultiplier_Fallback) *
                VectorVF_Factor));
  EXPECT_GE(cost.costTime, 1);
}

TEST_F(MSCostModelTest, VectorEmptyGroupClampsToOne) {
  MSCostModel model;
  // An empty group (transfer-only WorkItems bill no vector cost) is legal:
  // y = 0 clamps to the minimum cost 1, not an error.
  VectorGroupCostInput input;
  CostResult cost = model.getVectorGroupCost(input);
  EXPECT_EQ(cost.costTime, 1);
  EXPECT_TRUE(cost.note.empty());
}

TEST_F(MSCostModelTest, VectorDynamicInputSkipsOp) {
  // An op with a dynamic input dimension is skipped (zero cost contribution)
  // instead of failing the group — stride_load and similar ops whose GM
  // source is naturally dynamic should not block the SMS scheduler.
  MSCostModel model;
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.inputs.push_back(desc({128, 128}, f16));
  op0.inputs.push_back(desc({ShapedType::kDynamic, 8}, f16));
  input.ops.push_back(std::move(op0));

  CostResult cost = model.getVectorGroupCost(input);
  // The dynamic op is skipped; an empty sum clamps to minimum cost 1.
  EXPECT_EQ(cost.costTime, 1);
}

TEST_F(MSCostModelTest, VectorDynamicInputSkipsOnlyThatOp) {
  // When a group has both a dynamic op and a static op, only the dynamic
  // one is skipped — the static op still contributes its cost.
  MSCostModel model;
  VectorGroupCostInput input;
  // Op 0: dynamic input — will be skipped.
  VectorOpDesc op0;
  op0.inputs.push_back(desc({ShapedType::kDynamic}, f16));
  input.ops.push_back(std::move(op0));
  // Op 1: static input — contributes normally.
  VectorOpDesc op1;
  op1.inputs.push_back(desc({128, 128}, f16));
  input.ops.push_back(std::move(op1));

  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  // The cost should equal a single-op group with the static op alone.
  VectorGroupCostInput singleOp;
  VectorOpDesc op1Copy;
  op1Copy.inputs.push_back(desc({128, 128}, f16));
  singleOp.ops.push_back(std::move(op1Copy));
  CostResult singleCost = model.getVectorGroupCost(singleOp);
  EXPECT_EQ(cost.costTime, singleCost.costTime);
}

TEST_F(MSCostModelTest, VectorNoteDoesNotAffectCostTime) {
  MSCostModel model;
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.inputs.push_back(desc({128, 128}, f16));
  input.ops.push_back(std::move(op0));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);

  // The note never participates in the computation: rewriting it must not
  // change the cost.  (Structural check — the model always recomputes.)
  CostResult again = model.getVectorGroupCost(input);
  EXPECT_EQ(again.costTime, cost.costTime);
  again.note = "rewritten";
  EXPECT_EQ(again.costTime, cost.costTime);
}

// --- Transfer
// ------------------------------------------------------------------

TEST_F(MSCostModelTest, TransferAllPipelines) {
  MSCostModel model;
  struct Row {
    StringRef pipeline;
    StringRef tag;
    const LinParam *base;
    CostMultiplier dirMult;
  };
  // Effective param = pipeline base × direction multiplier.
  const Row rows[] = {
      {"mte1", "L1_to_L0A", &TransferBaseParam_MTE1,
       TransferDirMult_MTE1_L1_to_L0A},
      {"mte1", "L1_to_L0B", &TransferBaseParam_MTE1,
       TransferDirMult_MTE1_L1_to_L0B},
      {"mte2", "GM_to_L1", &TransferBaseParam_MTE2,
       TransferDirMult_MTE2_GM_to_L1},
      {"mte2", "GM_to_UB", &TransferBaseParam_MTE2,
       TransferDirMult_MTE2_GM_to_UB},
      {"mte3", "UB_to_L1", &TransferBaseParam_MTE3,
       TransferDirMult_MTE3_UB_to_L1},
      {"mte3", "UB_to_GM", &TransferBaseParam_MTE3,
       TransferDirMult_MTE3_UB_to_GM},
      {"fixpipe", "L0C_to_L1", &TransferBaseParam_FIXPIPE,
       TransferDirMult_FIXPIPE_L0C_to_L1},
      {"fixpipe", "L0C_to_UB", &TransferBaseParam_FIXPIPE,
       TransferDirMult_FIXPIPE_L0C_to_UB},
      {"fixpipe", "L0C_to_GM", &TransferBaseParam_FIXPIPE,
       TransferDirMult_FIXPIPE_L0C_to_GM},
  };
  for (const Row &row : rows) {
    TransferCostInput input;
    input.data = desc({64, 64}, f16); // 8192 bytes.
    input.pipeline = row.pipeline;
    input.storageTag = row.tag;
    CostResult cost = model.getTransferCost(input);
    ASSERT_NE(cost.costTime, -1) << row.pipeline.str() << " " << row.tag.str();
    EXPECT_EQ(cost.costTime, expectCost(row.base->a * row.dirMult.slope * 8192 +
                                        row.base->b * row.dirMult.intercept))
        << row.pipeline.str() << " " << row.tag.str();
    EXPECT_TRUE(cost.note.empty());
  }
}

TEST_F(MSCostModelTest, TransferErrorSentinels) {
  MSCostModel model;

  // Unknown (pipeline, storageTag).
  TransferCostInput unknown;
  unknown.data = desc({64, 64}, f16);
  unknown.pipeline = "mte3";
  unknown.storageTag = "UB_to_UB";
  CostResult cost = model.getTransferCost(unknown);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("transfer (mte3, UB_to_UB) not in param table"),
            std::string::npos);

  // Dynamic shape — skips with fallback cost instead of failing.
  TransferCostInput dyn;
  dyn.data = desc({ShapedType::kDynamic, 64}, f16);
  dyn.pipeline = "mte2";
  dyn.storageTag = "GM_to_UB";
  cost = model.getTransferCost(dyn);
  EXPECT_GE(cost.costTime, 1);
  EXPECT_NE(cost.note.find("dynamic shape"), std::string::npos);

  // Dtype not in the table.
  TransferCostInput dtype;
  dtype.data = desc({64, 64}, NoneType::get(context.get()));
  dtype.pipeline = "mte2";
  dtype.storageTag = "GM_to_UB";
  cost = model.getTransferCost(dtype);
  EXPECT_EQ(cost.costTime, -1);
  EXPECT_NE(cost.note.find("transfer dtype not accepted"), std::string::npos);
}

// --- Vector v2: multiplier-chain lookups -------------------------------------

TEST_F(MSCostModelTest, VectorDtypeMultiplierLookup) {
  // 8 supported dtypes hit their own constants.
  EXPECT_EQ(vectorDtypeMultiplierFor(i1), (CostMultiplier{1., 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(i8), (CostMultiplier{1., 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(i16), (CostMultiplier{2., 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(f16), (CostMultiplier{2., 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(bf16), (CostMultiplier{2., 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(i32), (CostMultiplier{3.96, 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(f32), (CostMultiplier{3.96, 1.}));
  EXPECT_EQ(vectorDtypeMultiplierFor(i64), (CostMultiplier{13.96, 1.}));
  // Out-of-table dtype (Q8): the lookup misses — the group-level model
  // then falls back to {1., 1.} + note (non-fatal, unlike cube).
  EXPECT_EQ(vectorDtypeMultiplierFor(f64), std::nullopt);
}

TEST_F(MSCostModelTest, VectorInputCountMultiplier) {
  // Coefficients per dtype.
  EXPECT_EQ(vectorInputCountCoeffFor(i1), 0.546);
  EXPECT_EQ(vectorInputCountCoeffFor(i8), 0.546);
  EXPECT_EQ(vectorInputCountCoeffFor(i16), 0.722);
  EXPECT_EQ(vectorInputCountCoeffFor(f16), 0.722);
  EXPECT_EQ(vectorInputCountCoeffFor(bf16), 0.722);
  EXPECT_EQ(vectorInputCountCoeffFor(i32), 0.821);
  EXPECT_EQ(vectorInputCountCoeffFor(f32), 0.821);
  EXPECT_EQ(vectorInputCountCoeffFor(i64), 0.073);
  EXPECT_EQ(vectorInputCountCoeffFor(f64), std::nullopt);

  // Formula (n-1)·c+1. end-to-end: a 3-input f16 op vs a 1-input f16 op.
  MSCostModel model;
  VectorGroupCostInput single, triple;
  VectorOpDesc s0;
  s0.inputs.push_back(desc({64}, f16));
  single.ops.push_back(std::move(s0));
  VectorOpDesc t0;
  t0.inputs.push_back(desc({64}, f16));
  t0.inputs.push_back(desc({64}, f16));
  t0.inputs.push_back(desc({64}, f16));
  triple.ops.push_back(std::move(t0));
  CostResult sCost = model.getVectorGroupCost(single);
  CostResult tCost = model.getVectorGroupCost(triple);
  ASSERT_NE(sCost.costTime, -1);
  ASSERT_NE(tCost.costTime, -1);
  EXPECT_EQ(sCost.costTime,
            expectCost(vectorOpY(f16, 1, 64, VectorOpTypeMultiplier_Fallback) *
                       VectorVF_Factor));
  EXPECT_EQ(tCost.costTime,
            expectCost(vectorOpY(f16, 3, 64, VectorOpTypeMultiplier_Fallback) *
                       VectorVF_Factor));
}

TEST_F(MSCostModelTest, VectorCVFactorAppliedToAllOps) {
  MSCostModel model;
  // i8 single-input op: the CV factor scales the slope, not the
  // intercept (fallback op-type multiplier {1., 1.}).
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.inputs.push_back(desc({1000}, i8));
  input.ops.push_back(std::move(op0));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime,
            expectCost(vectorOpY(i8, 1, 1000, VectorOpTypeMultiplier_Fallback) *
                       VectorVF_Factor));
}

TEST_F(MSCostModelTest, VectorOpTypeMultiplierTable) {
  // All 13 calibrated op names hit the table with their own multipliers.
  for (const auto &[name, mult] : VectorOpTypeMultiplierList)
    EXPECT_EQ(lookupVectorOpTypeMultiplier(name), mult) << name.str();
  // Near-miss names miss (exact match only) → fallback.
  EXPECT_EQ(lookupVectorOpTypeMultiplier("hivm.hir.vaddx"), std::nullopt);
  EXPECT_EQ(lookupVectorOpTypeMultiplier("vadd"), std::nullopt);
  EXPECT_EQ(lookupVectorOpTypeMultiplier("hivm.hir.vadds"), std::nullopt);
}

TEST_F(MSCostModelTest, VectorIgnoreListPrefixMatch) {
  MSCostModel model;

  // Group 1: three dialect-prefixed ops, each with a sizable input — all
  // skipped (no latency, no note); y = 0 clamps to 1.
  VectorGroupCostInput ignored;
  Operation *ops[] = {makeOp("memref.alloc"), makeOp("tensor.empty"),
                      makeOp("bufferization.to_memref")};
  for (Operation *op : ops) {
    VectorOpDesc opDesc;
    opDesc.op = op;
    opDesc.inputs.push_back(desc({1024, 1024}, f16));
    ignored.ops.push_back(std::move(opDesc));
  }
  CostResult cost = model.getVectorGroupCost(ignored);
  EXPECT_EQ(cost.costTime, 1);
  EXPECT_TRUE(cost.note.empty());
  for (Operation *op : ops)
    op->destroy();

  // Group 2: same ignored op PLUS an "hivm.hir.*" op (never matches a
  // dialect prefix in the list) with a table hit — only it is billed.
  VectorGroupCostInput mixed;
  Operation *alloc = makeOp("memref.alloc");
  VectorOpDesc allocDesc;
  allocDesc.op = alloc;
  allocDesc.inputs.push_back(desc({1024, 1024}, f16));
  mixed.ops.push_back(std::move(allocDesc));
  Operation *vadd = makeOp("hivm.hir.vadd");
  VectorOpDesc vaddDesc;
  vaddDesc.op = vadd;
  vaddDesc.inputs.push_back(desc({128}, f16));
  mixed.ops.push_back(std::move(vaddDesc));
  cost = model.getVectorGroupCost(mixed);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(
      cost.costTime,
      expectCost(vectorOpY(f16, 1, 128,
                           *lookupVectorOpTypeMultiplier("hivm.hir.vadd")) *
                 VectorVF_Factor));
  EXPECT_TRUE(cost.note.empty()); // vadd is calibrated, no fallback.
  alloc->destroy();
  vadd->destroy();
}

// --- vcast: conversion-path multiplier ---------------------------------------

TEST_F(MSCostModelTest, VCastDirectPairMultiplier) {
  // i16→f16: single direct edge (slope 1.) → {2.×1., 1.}.
  FailureOr<CostMultiplier> m = vcastPairMultiplier(i16, f16);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, VCastSlopeBaseFactor * vcastEdgeSlope("i16", "f16"));
  EXPECT_EQ(m->intercept, 1.);

  // f32→i64: direct edge 14.99 — stays the 1-hop path even though the
  // 2-hop via-i32 sum (1.99 + 5.15 = 7.14) is numerically smaller (hop
  // count = hardware command count has priority, design §9.2).
  m = vcastPairMultiplier(f32, i64);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, VCastSlopeBaseFactor * vcastEdgeSlope("f32", "i64"));
  EXPECT_EQ(m->intercept, 1.);
}

TEST_F(MSCostModelTest, VCastTransitPathMinSum) {
  // i32→f16: three 2-hop paths — via i16 (2.65 + 1. = 3.65), via i8
  // (2.77 + 1.4 = 4.17), via f32 (1.99 + 2.65 = 4.64); via-i16 wins by
  // the smallest-slope-sum rule.
  FailureOr<CostMultiplier> m = vcastPairMultiplier(i32, f16);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, VCastSlopeBaseFactor * (vcastEdgeSlope("i32", "i16") +
                                              vcastEdgeSlope("i16", "f16")));
  EXPECT_EQ(m->intercept, 2.); // two hops

  // f16→i8: f16→f32→i32→i8 = 2.77 + 1.99 + 2.77 = 7.53, three hops.
  m = vcastPairMultiplier(f16, i8);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, VCastSlopeBaseFactor * (vcastEdgeSlope("f16", "f32") +
                                              vcastEdgeSlope("f32", "i32") +
                                              vcastEdgeSlope("i32", "i8")));
  EXPECT_EQ(m->intercept, 3.);

  // i8→f32: two equal-hop equal-sum (1.4 + 2.77 = 4.17) paths —
  // i8→f16→f32 beats i8→i16→f32 by VCastDirectPairList enumeration order
  // (tie-break rule 3); the slope value alone cannot distinguish them,
  // so this also guards against accidental reordering of the table.
  m = vcastPairMultiplier(i8, f32);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, VCastSlopeBaseFactor * (vcastEdgeSlope("i8", "f16") +
                                              vcastEdgeSlope("f16", "f32")));
  EXPECT_EQ(m->intercept, 2.);
}

TEST_F(MSCostModelTest, VCastSameDtypeAndUnknown) {
  // Same dtype: identity conversion {2.×1., 1.}.
  FailureOr<CostMultiplier> m = vcastPairMultiplier(f16, f16);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, VCastSlopeBaseFactor);
  EXPECT_EQ(m->intercept, 1.);

  // Out-of-7 dtypes → failure (the caller applies the fallback + note).
  EXPECT_TRUE(failed(vcastPairMultiplier(f64, i8)));
  EXPECT_TRUE(failed(vcastPairMultiplier(i8, f64)));
  EXPECT_TRUE(failed(vcastPairMultiplier(i1, f32)));
  EXPECT_TRUE(failed(vcastPairMultiplier(f32, i1)));
}

TEST_F(MSCostModelTest, VCastGroupLevelChain) {
  // End-to-end vcast branch inside getVectorGroupCost: a bare op NAMED
  // "hivm.hir.vcast" with i32 input / f16 output rides the i32→i16→f16
  // path (3.65); the CV 0.5 factor applies on top (Q3), and the op-type
  // table miss lands it in the note.
  MSCostModel model;
  Operation *vcast = makeOp("hivm.hir.vcast");
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.op = vcast;
  op0.inputs.push_back(desc({64}, i32));
  op0.outputs.push_back(desc({64}, f16));
  input.ops.push_back(std::move(op0));

  CostMultiplier pair{VCastSlopeBaseFactor * (vcastEdgeSlope("i32", "i16") +
                                              vcastEdgeSlope("i16", "f16")),
                      2.};
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(vcastOpY(64, pair) * VectorVF_Factor));
  EXPECT_NE(cost.note.find("hivm.hir.vcast"), std::string::npos);

  // Same-dtype vcast: identity pair {2., 1.}.
  VectorGroupCostInput identity;
  VectorOpDesc op1;
  op1.op = vcast;
  op1.inputs.push_back(desc({64}, f32));
  op1.outputs.push_back(desc({64}, f32));
  identity.ops.push_back(std::move(op1));
  cost = model.getVectorGroupCost(identity);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime,
            expectCost(vcastOpY(64, CostMultiplier{VCastSlopeBaseFactor, 1.}) *
                       VectorVF_Factor));

  // vcast with missing outputs (no dst dtype recorded) → fallback pair:
  // the 2. base factor still applies, only the pair multiplier falls
  // back to 1. (§5.8) → {2.×1., 1.} + note.
  VectorGroupCostInput noOut;
  VectorOpDesc op2;
  op2.op = vcast;
  op2.inputs.push_back(desc({64}, f32));
  noOut.ops.push_back(std::move(op2));
  cost = model.getVectorGroupCost(noOut);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(
      cost.costTime,
      expectCost(
          vcastOpY(64, CostMultiplier{VCastSlopeBaseFactor *
                                          VCastPairMultiplier_Fallback.slope,
                                      VCastPairMultiplier_Fallback.intercept}) *
          VectorVF_Factor));
  EXPECT_NE(cost.note.find("hivm.hir.vcast"), std::string::npos);

  vcast->destroy();
}

// --- vcast i1 source (1019) --------------------------------------------------

TEST_F(MSCostModelTest, VCastI1SourceRules) {
  // i1→i8: no vector time at all — both multipliers zero.
  FailureOr<CostMultiplier> m = vcastPairMultiplier(i1, i8);
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(*m, VCastPairMultiplier_I1_to_I8);
  EXPECT_EQ(m->slope, 0.);
  EXPECT_EQ(m->intercept, 0.);

  // i1→f16 / i1→i32: billed exactly like an i8 source (1-hop direct
  // edges i8→f16 / i8→i32).
  FailureOr<CostMultiplier> asI8 = vcastPairMultiplier(i8, f16);
  m = vcastPairMultiplier(i1, f16);
  ASSERT_TRUE(succeeded(asI8));
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, asI8->slope);
  EXPECT_EQ(m->intercept, asI8->intercept);

  asI8 = vcastPairMultiplier(i8, i32);
  m = vcastPairMultiplier(i1, i32);
  ASSERT_TRUE(succeeded(asI8));
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, asI8->slope);
  EXPECT_EQ(m->intercept, asI8->intercept);

  // i1→bf16 rides the same 3-hop path as i8→bf16
  // (i8→f16→f32→bf16 = 1.4 + 2.65 + 2.65).
  asI8 = vcastPairMultiplier(i8, bf16);
  m = vcastPairMultiplier(i1, bf16);
  ASSERT_TRUE(succeeded(asI8));
  ASSERT_TRUE(succeeded(m));
  EXPECT_EQ(m->slope, asI8->slope);
  EXPECT_EQ(m->intercept, asI8->intercept);

  // i1 as DESTINATION keeps the fallback semantics (only the source side
  // is covered by the 1019 calibration).
  EXPECT_TRUE(failed(vcastPairMultiplier(f16, i1)));
  EXPECT_TRUE(failed(vcastPairMultiplier(i1, i1)));
}

TEST_F(MSCostModelTest, VCastI1ToI8GroupBillsNothing) {
  // A group holding ONLY an i1→i8 vcast: y_op = 0 → the group clamps to
  // the minimum cost 1.  The op still lands in the note via the op-type
  // table miss (vcast is not in the 13-op table — pre-existing
  // semantics, unchanged by the 1019 fix).
  MSCostModel model;
  Operation *vcast = makeOp("hivm.hir.vcast");
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.op = vcast;
  op0.inputs.push_back(desc({64}, i1));
  op0.outputs.push_back(desc({64}, i8));
  input.ops.push_back(std::move(op0));

  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, 1);
  EXPECT_NE(cost.note.find("hivm.hir.vcast"), std::string::npos);

  vcast->destroy();
}

// --- varange: dedicated output-driven model (1019) ---------------------------

TEST_F(MSCostModelTest, VarangeOutputDrivenModel) {
  MSCostModel model;
  Operation *varange = makeOp("hivm.hir.varange");
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.op = varange;
  // varange has no real shaped inputs (dst is the DPS init, offset and
  // strides are Index scalars).  A recorded input — should it ever
  // appear — is IGNORED: x comes from the output, N = 32·4 = 128.
  op0.inputs.push_back(desc({1000}, i32));
  op0.outputs.push_back(desc({32, 4}, i32));
  input.ops.push_back(std::move(op0));

  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(varangeOpY(128) * VectorVF_Factor));
  // A dedicated calibrated model is not a table miss — no fallback note.
  EXPECT_TRUE(cost.note.empty());

  varange->destroy();
}

TEST_F(MSCostModelTest, VarangeMissingOutputsFallsBack) {
  // No recorded output (e.g. bufferized memref form) → x = 0, intercept
  // only, op in the note — non-fatal, vcast's missing-metadata
  // philosophy.
  MSCostModel model;
  Operation *varange = makeOp("hivm.hir.varange");
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.op = varange;
  op0.inputs.push_back(desc({128}, i32));
  input.ops.push_back(std::move(op0));

  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(varangeOpY(0) * VectorVF_Factor));
  EXPECT_NE(cost.note.find("hivm.hir.varange"), std::string::npos);

  varange->destroy();
}

TEST_F(MSCostModelTest, VarangeDynamicOutputSkipsOp) {
  // A dynamic output shape on varange is skipped (zero cost contribution)
  // instead of failing the group.
  MSCostModel model;
  Operation *varange = makeOp("hivm.hir.varange");
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.op = varange;
  op0.outputs.push_back(desc({ShapedType::kDynamic}, i32));
  input.ops.push_back(std::move(op0));

  CostResult cost = model.getVectorGroupCost(input);
  // The dynamic op is skipped; an empty sum clamps to minimum cost 1.
  EXPECT_EQ(cost.costTime, 1);

  varange->destroy();
}

// --- vbrc: dedicated broadcast model (1019) --------------------------------

TEST_F(MSCostModelTest, VbrcVectorPatternsEndToEnd) {
  MSCostModel model;
  Operation *vbrc = makeOp("hivm.hir.vbrc");

  // inner: src (d0,1), padded result (d0,pad(d1)), real out (d0,d1) — a
  // single-op group bills the broadcast query result scaled by the VF
  // factor (vbrcGroupCost mirrors the group-level ③ + ⑧ arithmetic).
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.inputs.push_back(desc({100, 1}, i8));
    op0.outputs.push_back(desc({100, 8}, i8));
    op0.vbrcRealOutShape = {100, 7};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              vbrcGroupCost(broadcast::getInnerCost(i8, 100, 7)));
    EXPECT_TRUE(cost.note.empty());
  }

  // outer: src (1,d1), real out (d0,d1).
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.inputs.push_back(desc({1, 128}, bf16));
    op0.outputs.push_back(desc({16, 128}, bf16));
    op0.vbrcRealOutShape = {10, 128};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              vbrcGroupCost(broadcast::getOuterCost(bf16, 10, 128)));
    EXPECT_TRUE(cost.note.empty());
  }

  // middle: src (d0,1,d2), real out (d0,d1,d2).
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.inputs.push_back(desc({8, 1, 16}, f16));
    op0.outputs.push_back(desc({8, 8, 16}, f16));
    op0.vbrcRealOutShape = {8, 7, 16};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              vbrcGroupCost(broadcast::getMiddleCost(f16, 8, 7)));
    EXPECT_TRUE(cost.note.empty());
  }

  vbrc->destroy();
}

TEST_F(MSCostModelTest, VbrcScalarPatternsEndToEnd) {
  MSCostModel model;
  Operation *vbrc = makeOp("hivm.hir.vbrc");

  // scalar1d: bare scalar source → no recorded inputs; real out (N).
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.outputs.push_back(desc({512}, i32));
    op0.vbrcRealOutShape = {300};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              vbrcGroupCost(broadcast::getScalar1dCost(i32, 300)));
    EXPECT_TRUE(cost.note.empty());
  }

  // scalar2d: a 0-D source tensor (empty shape) is a scalar source too.
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.inputs.push_back(desc({}, bf16));
    op0.outputs.push_back(desc({16, 2}, bf16));
    op0.vbrcRealOutShape = {10, 2};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              vbrcGroupCost(broadcast::getScalar2dCost(bf16, 10, 2)));
    EXPECT_TRUE(cost.note.empty());
  }

  vbrc->destroy();
}

TEST_F(MSCostModelTest, VbrcUncoveredFormsFallBackToGeneric) {
  MSCostModel model;
  Operation *vbrc = makeOp("hivm.hir.vbrc");

  // A rank-3 source with no 1-sized mid dim is not one of the five
  // patterns: generic billing (op-type fallback {1., 1.}), op in the note.
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.inputs.push_back(desc({2, 3, 4}, f16));
    op0.outputs.push_back(desc({2, 3, 4}, f16));
    op0.vbrcRealOutShape = {2, 3, 4};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(
        cost.costTime,
        expectCost(vectorOpY(f16, 1, 24, VectorOpTypeMultiplier_Fallback) *
                   VectorVF_Factor));
    EXPECT_NE(cost.note.find("hivm.hir.vbrc"), std::string::npos);
  }

  // Recognized inner form but an uncalibrated dtype (i64): the broadcast
  // model declines, the generic chain bills it (i64 has a b1 entry).
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.inputs.push_back(desc({4, 1}, i64));
    op0.outputs.push_back(desc({4, 8}, i64));
    op0.vbrcRealOutShape = {4, 6};
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              expectCost(vectorOpY(i64, 1, 4, VectorOpTypeMultiplier_Fallback) *
                         VectorVF_Factor));
    EXPECT_NE(cost.note.find("hivm.hir.vbrc"), std::string::npos);
  }

  // Missing real shape (e.g. bufferized memref form): generic billing;
  // with no inputs that is the intercept alone.
  {
    VectorGroupCostInput input;
    VectorOpDesc op0;
    op0.op = vbrc;
    op0.outputs.push_back(desc({64, 32}, f32));
    input.ops.push_back(std::move(op0));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1);
    EXPECT_EQ(cost.costTime,
              expectCost(vectorOpY(std::nullopt, 0, 0,
                                   VectorOpTypeMultiplier_Fallback) *
                         VectorVF_Factor));
    EXPECT_NE(cost.note.find("hivm.hir.vbrc"), std::string::npos);
  }

  vbrc->destroy();
}

TEST_F(MSCostModelTest, VectorPerOpInterceptSemantics) {
  MSCostModel model;
  // Two inputs on ONE op: the intercept counts ONCE (v2 semantics — v1
  // added b per input).
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.inputs.push_back(desc({32}, f32));
  op0.inputs.push_back(desc({32}, f32));
  input.ops.push_back(std::move(op0));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime,
            expectCost(vectorOpY(f32, 2, 32, VectorOpTypeMultiplier_Fallback) *
                       VectorVF_Factor));

  // Splitting the same inputs over TWO single-input ops doubles the
  // intercept term (one b_eff per op) — the per-op semantic, made visible.
  VectorGroupCostInput split;
  VectorOpDesc a, b;
  a.inputs.push_back(desc({32}, f32));
  b.inputs.push_back(desc({32}, f32));
  split.ops.push_back(std::move(a));
  split.ops.push_back(std::move(b));
  CostResult splitCost = model.getVectorGroupCost(split);
  ASSERT_NE(splitCost.costTime, -1);
  double yPerOp = vectorOpY(f32, 1, 32, VectorOpTypeMultiplier_Fallback);
  EXPECT_EQ(splitCost.costTime, expectCost(2 * yPerOp * VectorVF_Factor));
}

// --- Success invariants
// --------------------------------------------------------

TEST_F(MSCostModelTest, SuccessAlwaysAtLeastOne) {
  MSCostModel model;
  // Tiny shapes: y is far below 1 μs, but the ns scaling keeps it above the
  // minimum cost — a successful query never returns 0 or a negative value.
  CubeCostInput cube;
  cube.lhs = desc({1, 1, 16, 16}, f16);
  cube.rhs = desc({1, 1, 16, 16}, f16);
  CostResult cost = model.getCubeCost(cube);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_GE(cost.costTime, 1);

  TransferCostInput transfer;
  transfer.data = desc({1, 1}, f16);
  transfer.pipeline = "mte2";
  transfer.storageTag = "GM_to_UB";
  cost = model.getTransferCost(transfer);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_GE(cost.costTime, 1);
}

} // namespace
