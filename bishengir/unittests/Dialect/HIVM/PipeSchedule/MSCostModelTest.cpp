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
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostParams.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
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
    // Bare-named ops ("memref.alloc", "hivm.hir.vadd", ...) are created
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

  /// Per-op y, mirroring getVectorGroupCost's per-op formula (same
  /// constants, same multiplication/association order — the comparisons
  /// below are then exact, not tolerance-based).  Op kind is absent on
  /// purpose: it does not enter the billing.  `dtype` is std::nullopt for
  /// the two cases that keep the {1., 1.} fallback — an op with no shaped
  /// input, and an op whose dtype is outside the dtype table.
  double vectorOpY(std::optional<Type> dtype, int64_t x) {
    CostMultiplier dtypeMult{1., 1.};
    if (dtype)
      dtypeMult = *vectorDtypeMultiplierFor(*dtype);
    double aEff = VectorBaseParam.a * dtypeMult.slope * VectorCVSlopeFactor;
    double bEff = VectorBaseParam.b * dtypeMult.intercept;
    return aEff * static_cast<double>(x) + bEff;
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

TEST_F(MSCostModelTest, VectorGroupPlainSum) {
  MSCostModel model;
  VectorGroupCostInput input;
  // Two ops with op = nullptr: op kind no longer selects a multiplier, so
  // a nameless op bills exactly like a named one.
  VectorOpDesc op0;
  op0.inputs.push_back(desc({128, 128}, f16));
  VectorOpDesc op1;
  op1.inputs.push_back(desc({128, 128}, f16));
  op1.inputs.push_back(desc({64, 128}, f16));
  input.ops.push_back(std::move(op0));
  input.ops.push_back(std::move(op1));

  // x = max input element count (not the sum); the intercept counts once
  // per op (not once per input); the group cost is the plain per-op sum.
  // op1's extra, smaller input therefore raises neither term.
  double sum = vectorOpY(f16, 128 * 128) + vectorOpY(f16, 128 * 128);
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(sum));
  // f16 is in the dtype table and op kind is never consulted for a
  // multiplier, so there is nothing to report.
  EXPECT_TRUE(cost.note.empty());
}

TEST_F(MSCostModelTest, VectorZeroInputsOpCountsOnlyB) {
  MSCostModel model;
  // An op with no shaped inputs: the dtype multiplier stays {1., 1.} and
  // x = 0, so only the per-op intercept term (b_eff) remains; a group of
  // only such ops still yields max(1, round(b_eff · n)) ≥ 1.
  VectorGroupCostInput input;
  VectorOpDesc op0;
  input.ops.push_back(std::move(op0));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  // No inputs → the dtype multiplier defaults to {1., 1.}, so
  // b_eff = VectorBaseParam.b.
  EXPECT_EQ(cost.costTime, expectCost(vectorOpY(std::nullopt, 0)));
  EXPECT_GE(cost.costTime, 1);
  // No shaped input means no dtype lookup happened — nothing to note.
  EXPECT_TRUE(cost.note.empty());
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

// --- Vector: dtype ratio, the only multiplier table --------------------------

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
  // Out-of-table dtype (f64): the lookup misses — the group-level model
  // then falls back to {1., 1.} + note (non-fatal, unlike cube).
  EXPECT_EQ(vectorDtypeMultiplierFor(f64), std::nullopt);
}

TEST_F(MSCostModelTest, VectorInputCountDoesNotChangeCost) {
  // Operand count is not a cost dimension: a 3-input f16 op bills exactly
  // like a 1-input f16 op over the same element count, because x is the
  // MAX over the inputs and the intercept is charged once per op.  The
  // element count is large on purpose — at a small x the ns rounding
  // would hide a reintroduced operand-count term.
  MSCostModel model;
  VectorGroupCostInput single, triple;
  VectorOpDesc s0;
  s0.inputs.push_back(desc({1024, 1024}, f16));
  single.ops.push_back(std::move(s0));
  VectorOpDesc t0;
  t0.inputs.push_back(desc({1024, 1024}, f16));
  t0.inputs.push_back(desc({1024, 1024}, f16));
  t0.inputs.push_back(desc({1024, 1024}, f16));
  triple.ops.push_back(std::move(t0));
  CostResult sCost = model.getVectorGroupCost(single);
  CostResult tCost = model.getVectorGroupCost(triple);
  ASSERT_NE(sCost.costTime, -1);
  ASSERT_NE(tCost.costTime, -1);
  EXPECT_EQ(sCost.costTime, expectCost(vectorOpY(f16, 1024 * 1024)));
  EXPECT_EQ(tCost.costTime, sCost.costTime);
}

TEST_F(MSCostModelTest, VectorCVFactorAppliedToAllOps) {
  MSCostModel model;
  // Two ops of different dtypes: the CV factor scales every op's slope.
  // x is large so that halving the slope moves the reported ns by far
  // more than one rounding step.
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.inputs.push_back(desc({1000000}, i8));
  VectorOpDesc op1;
  op1.inputs.push_back(desc({1000000}, f32));
  input.ops.push_back(std::move(op0));
  input.ops.push_back(std::move(op1));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime,
            expectCost(vectorOpY(i8, 1000000) + vectorOpY(f32, 1000000)));

  // The i8 op alone, spelled out without the helper: the factor hits the
  // SLOPE term only and leaves the intercept whole.
  VectorGroupCostInput solo;
  VectorOpDesc soloOp;
  soloOp.inputs.push_back(desc({1000000}, i8));
  solo.ops.push_back(std::move(soloOp));
  CostResult soloCost = model.getVectorGroupCost(solo);
  ASSERT_NE(soloCost.costTime, -1);
  EXPECT_EQ(soloCost.costTime,
            expectCost(VectorBaseParam.a * VectorDtypeMult_I8.slope *
                           VectorCVSlopeFactor * 1000000 +
                       VectorBaseParam.b * VectorDtypeMult_I8.intercept));
  // Dropping the factor would roughly double the slope term, so the
  // uncorrected formula must land somewhere else entirely.
  EXPECT_LT(soloCost.costTime,
            expectCost(VectorBaseParam.a * VectorDtypeMult_I8.slope * 1000000 +
                       VectorBaseParam.b * VectorDtypeMult_I8.intercept));
}

TEST_F(MSCostModelTest, VectorOpKindDoesNotAffectCost) {
  // Op kind is not a cost dimension either: any billable name — one that
  // used to carry a calibrated multiplier, an unknown "hivm.hir.*" one, a
  // foreign-dialect one, or no op at all — bills the same formula for the
  // same dtype and element count.  vadd and vexp had DIFFERENT calibrated
  // multipliers, so at this element count a reintroduced op-type table
  // would separate them well beyond the rounding step.
  MSCostModel model;
  const StringRef names[] = {"hivm.hir.vadd", "hivm.hir.vexp",
                             "hivm.hir.vunknown", "scf.if"};
  const int64_t elements = 1024 * 1024;
  for (StringRef name : names) {
    Operation *op = makeOp(name);
    VectorGroupCostInput input;
    VectorOpDesc opDesc;
    opDesc.op = op;
    opDesc.inputs.push_back(desc({1024, 1024}, f16));
    input.ops.push_back(std::move(opDesc));
    CostResult cost = model.getVectorGroupCost(input);
    ASSERT_NE(cost.costTime, -1) << name.str();
    EXPECT_EQ(cost.costTime, expectCost(vectorOpY(f16, elements)))
        << name.str();
    EXPECT_TRUE(cost.note.empty()) << name.str();
    op->destroy();
  }

  // The nameless op (op == nullptr) lands on that same value.
  VectorGroupCostInput nameless;
  VectorOpDesc opDesc;
  opDesc.inputs.push_back(desc({1024, 1024}, f16));
  nameless.ops.push_back(std::move(opDesc));
  CostResult cost = model.getVectorGroupCost(nameless);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(vectorOpY(f16, elements)));
}

// --- Vector: dtype fallback, the one non-fatal anomaly -----------------------

TEST_F(MSCostModelTest, VectorOutOfTableDtypeFallsBackWithNote) {
  // f64 is outside the vector dtype table.  Unlike cube, that is NOT
  // fatal: the op bills with the {1., 1.} fallback and its name lands in
  // the note, which the caller turns into a warning.  This is the only
  // note the vector path produces.
  MSCostModel model;
  Operation *op = makeOp("hivm.hir.vadd");
  VectorGroupCostInput input;
  VectorOpDesc opDesc;
  opDesc.op = op;
  opDesc.inputs.push_back(desc({256}, f64));
  input.ops.push_back(std::move(opDesc));

  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(vectorOpY(std::nullopt, 256)));
  EXPECT_NE(cost.note.find("hivm.hir.vadd"), std::string::npos);

  op->destroy();
}

TEST_F(MSCostModelTest, VectorFallbackNoteNamesEachOpOnce) {
  // The same out-of-table op appearing twice is named once in the note; a
  // second distinct name is appended.  The cost still counts all three.
  MSCostModel model;
  Operation *vadd = makeOp("hivm.hir.vadd");
  Operation *vmul = makeOp("hivm.hir.vmul");
  VectorGroupCostInput input;
  for (Operation *op : {vadd, vadd, vmul}) {
    VectorOpDesc opDesc;
    opDesc.op = op;
    opDesc.inputs.push_back(desc({256}, f64));
    input.ops.push_back(std::move(opDesc));
  }

  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(vectorOpY(std::nullopt, 256) +
                                      vectorOpY(std::nullopt, 256) +
                                      vectorOpY(std::nullopt, 256)));
  EXPECT_EQ(cost.note, "fallback ops: hivm.hir.vadd, hivm.hir.vmul");

  vadd->destroy();
  vmul->destroy();
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
  // dialect prefix in the list) — only the latter is billed.
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
  EXPECT_EQ(cost.costTime, expectCost(vectorOpY(f16, 128)));
  EXPECT_TRUE(cost.note.empty()); // f16 is in the dtype table.
  alloc->destroy();
  vadd->destroy();
}

TEST_F(MSCostModelTest, VectorPerOpInterceptSemantics) {
  MSCostModel model;
  // Two inputs on ONE op: the intercept counts ONCE, not once per input.
  VectorGroupCostInput input;
  VectorOpDesc op0;
  op0.inputs.push_back(desc({32}, f32));
  op0.inputs.push_back(desc({32}, f32));
  input.ops.push_back(std::move(op0));
  CostResult cost = model.getVectorGroupCost(input);
  ASSERT_NE(cost.costTime, -1);
  EXPECT_EQ(cost.costTime, expectCost(vectorOpY(f32, 32)));

  // Splitting the same inputs over TWO single-input ops doubles the whole
  // per-op term, intercept included — the per-op semantic, made visible.
  VectorGroupCostInput split;
  VectorOpDesc a, b;
  a.inputs.push_back(desc({32}, f32));
  b.inputs.push_back(desc({32}, f32));
  split.ops.push_back(std::move(a));
  split.ops.push_back(std::move(b));
  CostResult splitCost = model.getVectorGroupCost(split);
  ASSERT_NE(splitCost.costTime, -1);
  double yPerOp = vectorOpY(f32, 32);
  EXPECT_EQ(splitCost.costTime, expectCost(2 * yPerOp));
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
