//===- ReportFatalErrorDeathTest.cpp - death tests for fatal helpers ------===//
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
//
// Death tests for llvm::report_fatal_error sites introduced by
// cb389a3252a6489bc12e8d29b0391b57410c469e that are reachable via public /
// header APIs. Sites buried in static helpers or pattern matchAndRewrite
// defaults cannot be covered without production-code changes.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionAnalyzer.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFFusionInterfaces.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFUnionFind.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HFusion/Transforms/regbase/RegBaseArchUtils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/HIVM/IR/HIVMVectorize.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/IndexBoundAnalyzer.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/regbase/Normalize/Utils/TrigTemplateHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace {

#ifdef GTEST_HAS_DEATH_TEST

struct DummyTraits {
  static Value createBinaryOp(PatternRewriter &, Location, Value, Value, Value,
                              BinaryKind) {
    llvm::report_fatal_error("DummyTraits::createBinaryOp should not run");
  }
};

struct DummyVFAnalyzer
    : public analysis::VFFusionAnalyzerBase<DummyVFAnalyzer> {
  DummyVFAnalyzer()
      : analysis::VFFusionAnalyzerBase<DummyVFAnalyzer>(
            analysis::VFFusionKindOption(/*enableOutlineCF=*/false,
                                         /*enableOutlineMemref=*/false,
                                         /*enableOutlineArith=*/false,
                                         /*enableOutlineCube=*/false)) {}
};

class ReportFatalErrorDeathTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect, LLVM::LLVMDialect,
                        hivm::HIVMDialect,
                        hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
    builder = std::make_unique<OpBuilder>(&context);
    rewriter = std::make_unique<PatternRewriter>(&context);
    module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module->getBody());
    rewriter->setInsertionPointToStart(module->getBody());
  }

  Location loc() { return builder->getUnknownLoc(); }

  Value constF32(float v = 1.0f) {
    return builder->create<arith::ConstantOp>(loc(),
                                              builder->getF32FloatAttr(v));
  }

  Value constIndex(int64_t v) {
    return builder->create<arith::ConstantIndexOp>(loc(), v);
  }

  Value i32(int32_t v = 0) {
    return rewriter->create<arith::ConstantOp>(loc(),
                                               rewriter->getI32IntegerAttr(v));
  }

  Value f32Vec(int64_t n = 64) {
    auto ty = VectorType::get({n}, rewriter->getF32Type());
    return rewriter->create<arith::ConstantOp>(
        loc(), DenseElementsAttr::get(ty, 0.0f));
  }

  Value i8Vec(int64_t n, int8_t v = 0) {
    auto ty = VectorType::get({n}, rewriter->getI8Type());
    return rewriter->create<arith::ConstantOp>(
        loc(), DenseElementsAttr::get(ty, v));
  }

  Value ptrLike() {
    // Stand-in pointer-like operand for builders that only branch on
    // elementType before creating ops.
    return i32(0);
  }

  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<PatternRewriter> rewriter;
  OwningOpRef<ModuleOp> module;
};

TEST_F(ReportFatalErrorDeathTest, UtilsSelectRoundModeUnsupportedType) {
  EXPECT_DEATH(
      (void)utils::selectRoundMode<hivm::RoundMode>(builder->getF16Type(),
                                                    builder->getBF16Type()),
      "unsupported type cast");
}

TEST_F(ReportFatalErrorDeathTest, HFusionSelectRoundModeUnsupportedType) {
  EXPECT_DEATH((void)hfusion::selectRoundMode<hfusion::RoundMode>(
                   builder->getIndexType(), builder->getF32Type()),
               "unsupported type cast");
}

TEST_F(ReportFatalErrorDeathTest, TaylorTermCountMustBePositive) {
  EXPECT_DEATH((void)getTaylorSeriesCoefficients(TaylerMode::SIN, 0),
               "Taylor expansion term count must be positive");
  EXPECT_DEATH((void)getTaylorSeriesCoefficients(TaylerMode::ATAN, -1),
               "Taylor expansion term count must be positive");
}

TEST_F(ReportFatalErrorDeathTest, TaylorUnsupportedMode) {
  EXPECT_DEATH(
      (void)getTaylorSeriesCoefficients(static_cast<TaylerMode>(99), 3),
      "unsupported TaylerMode");
}

TEST_F(ReportFatalErrorDeathTest, TaylorCoefficientsMustNotBeEmpty) {
  Value input = constF32();
  EXPECT_DEATH((void)buildTaylorApproximation<DummyTraits>(
                   *rewriter, loc(), input, /*coefficients=*/{}),
               "Taylor coefficients must not be empty");
}

TEST_F(ReportFatalErrorDeathTest, PolynomialCoefficientsMustNotBeEmpty) {
  Value input = constF32();
  EXPECT_DEATH((void)buildPolynomialFromSquaredInput<DummyTraits>(
                   *rewriter, loc(), input, input, /*coefficients=*/{}),
               "polynomial coefficients must not be empty");
}

TEST_F(ReportFatalErrorDeathTest, GetIdentityElementUnsupportedType) {
  EXPECT_DEATH((void)hivm::getIdentityElement(*builder, loc(),
                                              builder->getIndexType(),
                                              hivm::VectorArithKind::ADD),
               "unsupported element type for neutral element");
}

TEST_F(ReportFatalErrorDeathTest, CreateVectorArithOpUnsupportedType) {
  Value lhs = constIndex(1);
  Value rhs = constIndex(2);
  EXPECT_DEATH((void)hivm::createVectorArithOp(
                   *builder, loc(), hivm::VectorArithKind::ADD, lhs, rhs),
               "unsupported element type for vector arithmetic");
}

TEST_F(ReportFatalErrorDeathTest, IndexBoundUnknownPredicate) {
  Value lhs = constIndex(0);
  Value rhs = constIndex(1);
  utils::IndexBoundAnalyzer analyzer;
  EXPECT_DEATH(
      (void)analyzer.compare(
          lhs, static_cast<utils::BoundComparisonPredicate>(99), rhs),
      "unknown bound comparison predicate");
}

TEST_F(ReportFatalErrorDeathTest, BoundCompareResultUnknownKind) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  EXPECT_DEATH(os << utils::BoundCompareResult(
                   static_cast<utils::BoundCompareResult::Kind>(99)),
               "unknown bound comparison result");
}

TEST_F(ReportFatalErrorDeathTest, NoLibraryFunctionTrait) {
  EXPECT_DEATH(
      (void)OpTrait::NoLibraryFunctionTrait<Operation>::getOpLibraryMaxRankImpl(),
      "This op has no library function");
  EXPECT_DEATH((void)OpTrait::NoLibraryFunctionTrait<Operation>::
                   getOpLibraryCallName(std::nullopt),
               "This op has no library function");
}

TEST_F(ReportFatalErrorDeathTest, VFUnionFindAllocateMinimum) {
  analysis::VFUnionFind uf;
  EXPECT_DEATH(uf.allocateMinimum(0), "shouldn't allocate any new indices");
}

TEST_F(ReportFatalErrorDeathTest, FusionKindBaseAnalyzeBlockImpl) {
  analysis::FusionKindBase kind(analysis::VFFusionKindOption(
      /*enableOutlineCF=*/false, /*enableOutlineMemref=*/false,
      /*enableOutlineArith=*/false, /*enableOutlineCube=*/false));
  Block block;
  EXPECT_DEATH((void)kind.analyzeBlockImpl(block),
               "analyze block is not implemented");
}

TEST_F(ReportFatalErrorDeathTest, VFFusionAnalyzerFuseImpl) {
  DummyVFAnalyzer analyzer;
  Block block;
  EXPECT_DEATH((void)analyzer.fuseImpl(block),
               "missing implementation fuseImpl for the specified FusionKind");
}

TEST_F(ReportFatalErrorDeathTest, GetHWAlignBytesUnsupportedAddressSpace) {
  auto gm = hivm::AddressSpaceAttr::get(&context, hivm::AddressSpace::GM);
  EXPECT_DEATH((void)hivm::getHWAlignBytes(gm), "Unsupported address space");
}

TEST_F(ReportFatalErrorDeathTest, HivmaveGetIndicesUnsupportedOp) {
  Operation *op = constF32().getDefiningOp();
  EXPECT_DEATH((void)hivmave::getIndices(op), "unsupported op type");
}

TEST_F(ReportFatalErrorDeathTest, HivmCreateUnaryOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hivm::NormalizeTraitsBase::createUnaryOp(
                   *rewriter, loc(), v, v, static_cast<UnaryKind>(99)),
               "unsupported unary kind");
}

TEST_F(ReportFatalErrorDeathTest, HivmCreateBinaryOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hivm::NormalizeTraitsBase::createBinaryOp(
                   *rewriter, loc(), v, v, v, static_cast<BinaryKind>(99)),
               "unsupported binary kind");
}

TEST_F(ReportFatalErrorDeathTest, HivmCreateShiftOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hivm::NormalizeTraitsBase::createShiftOp(
                   *rewriter, loc(), v, v, v, static_cast<ShiftKind>(99)),
               "unsupported shift kind");
}

TEST_F(ReportFatalErrorDeathTest, HivmCreateTernaryOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hivm::NormalizeTraitsBase::createTernaryOp(
                   *rewriter, loc(), v, v, v, v, static_cast<TernaryKind>(99)),
               "unsupported ternary kind");
}

TEST_F(ReportFatalErrorDeathTest, HivmCreateCastOpPreserveUnsupported) {
  Value v = constF32();
  EXPECT_DEATH((void)hivm::NormalizeTraitsBase::createCastOp(
                   *rewriter, loc(), v, builder->getF16Type(),
                   CastRoundKind::Default, Value(), CastSignKind::Preserve),
               "createCastOp does not support CastSignKind::Preserve");
}

TEST_F(ReportFatalErrorDeathTest, HivmCreateFillOpNonScalar) {
  auto tensorTy = RankedTensorType::get({2}, builder->getF32Type());
  Value fillVal = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(tensorTy, 1.0f));
  Value out = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(tensorTy, 0.0f));
  EXPECT_DEATH((void)hivm::NormalizeTraitsBase::createFillOp(*rewriter, loc(),
                                                             fillVal, out),
               "NormalizeTraitsBase::createFillOp only supports scalar-to-tensor fills");
}

TEST_F(ReportFatalErrorDeathTest, HFusionCreateUnaryOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hfusion::NormalizeTraitsBase::createUnaryOp(
                   *rewriter, loc(), v, v, static_cast<UnaryKind>(99)),
               "unsupported unary kind");
}

TEST_F(ReportFatalErrorDeathTest, HFusionCreateBinaryOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hfusion::NormalizeTraitsBase::createBinaryOp(
                   *rewriter, loc(), v, v, v, static_cast<BinaryKind>(99)),
               "unsupported binary kind");
}

TEST_F(ReportFatalErrorDeathTest, HFusionCreateShiftOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hfusion::NormalizeTraitsBase::createShiftOp(
                   *rewriter, loc(), v, v, v, static_cast<ShiftKind>(99)),
               "unsupported shift kind");
}

TEST_F(ReportFatalErrorDeathTest, HFusionCreateTernaryOpUnsupportedKind) {
  Value v = constF32();
  EXPECT_DEATH((void)hfusion::NormalizeTraitsBase::createTernaryOp(
                   *rewriter, loc(), v, v, v, v, static_cast<TernaryKind>(99)),
               "unsupported ternary kind");
}

TEST_F(ReportFatalErrorDeathTest, HFusionCreateCastOpPreserveUnsupported) {
  Value v = constF32();
  EXPECT_DEATH((void)hfusion::NormalizeTraitsBase::createCastOp(
                   *rewriter, loc(), v, builder->getF16Type(),
                   CastRoundKind::Default, Value(), CastSignKind::Preserve),
               "createCastOp does not support CastSignKind::Preserve");
}

TEST_F(ReportFatalErrorDeathTest, BuildVBrInvalidElementType) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVBrOp(
                   i32(), rewriter->getI64Type(), *rewriter),
               "Invalid vbr element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildVldsInvalidElementType) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVldsOp(
                   ptrLike(), i32(), rewriter->getI64Type(), *rewriter),
               "Invalid vlds element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildVldusInvalidElementType) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVldusOp(
                   ptrLike(), i32(), rewriter->getIntegerType(128), *rewriter),
               "Invalid vldus element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildVldusPostInvalidElementType) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVldusPostOp(
                   ptrLike(), i32(), i32(), rewriter->getIntegerType(128),
                   *rewriter),
               "Invalid vldus post element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildVstsUnsupportedDatatypes) {
  auto src = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({64}, rewriter->getI64Type()), int64_t(0)));
  auto pred = i8Vec(256);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVstsOp(src, ptrLike(), i32(),
                                                      pred, *rewriter),
               "unsupported datatypes");
}

TEST_F(ReportFatalErrorDeathTest, BuildVstsInvalidElementType) {
  auto src = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({64}, rewriter->getI64Type()), int64_t(0)));
  auto pred = i8Vec(256);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVstsOp(
                   src, ptrLike(), i32(), pred, *rewriter, /*dist=*/1),
               "Invalid vsts element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildVstusPostInvalidElementType) {
  auto src = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({32}, rewriter->getIntegerType(128)),
                 APInt(128, 0)));
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVstusPostOp(
                   src, ptrLike(), i32(), i32(), *rewriter),
               "Invalid vstus element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildPsetInvalidBitWidth) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildPsetOp(
                   i32(), /*elementBitWidth=*/7, *rewriter),
               "Invalid element bit width for a predicate vector");
}

TEST_F(ReportFatalErrorDeathTest, BuildMovvpUnsupportedAlignment) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildMovvpOp(
                   loc(), rewriter->getI32Type(), i32(), *rewriter,
                   /*elementAlignment=*/8),
               "elementAlignment is not supported");
}

TEST_F(ReportFatalErrorDeathTest, BuildVpackInvalidElementType) {
  auto src = i8Vec(256);
  auto part = i32(0);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVpackOp(
                   loc(), part, src, src.getType(), *rewriter),
               "Invalid element type for VpackInstrOp");
}

TEST_F(ReportFatalErrorDeathTest, BuildVunpackInvalidElementType) {
  auto src = f32Vec(64);
  auto part = i32(0);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVunpackOp(
                   loc(), part, src, src.getType(), *rewriter),
               "Invalid element type for Vzunpack");
}

TEST_F(ReportFatalErrorDeathTest, BuildPgeInvalidAlignment) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildPgeOp(loc(), i32(),
                                                     /*elementAlignment=*/7,
                                                     *rewriter),
               "Invalid element bit width for a predicate vector");
}

TEST_F(ReportFatalErrorDeathTest, BuildPltInvalidAlignment) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildPltOp(loc(), i32(),
                                                     /*elementAlignment=*/7,
                                                     *rewriter),
               "Invalid element bit width for a predicate vector");
}

TEST_F(ReportFatalErrorDeathTest, BuildPltMInvalidAlignment) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildPltMOp(loc(), i32(), i32(),
                                                      /*elementAlignment=*/7,
                                                      *rewriter),
               "Invalid element bit width for a predicate vector");
}

TEST_F(ReportFatalErrorDeathTest, BuildPstuInvalidAlignment) {
  EXPECT_DEATH((void)hivm_regbaseintrins::buildPstuOp(
                   i8Vec(256), ptrLike(), *rewriter, /*elementAlignment=*/7,
                   i32()),
               "Invalid elementAlignment");
}

TEST_F(ReportFatalErrorDeathTest, BuildVdupInvalidElementType) {
  auto pred = i8Vec(256);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildVdupOp(
                   i32(), pred, rewriter->getI64Type(), *rewriter),
               "Invalid vsts element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildAddInvalidDavidElementType) {
  auto lhs = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({64}, rewriter->getI64Type()), int64_t(0)));
  auto pred = i8Vec(256);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildAddOp(lhs, lhs, pred, *rewriter),
               "Invalid david op element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildMaxInvalidDavidElementType) {
  auto lhs = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({64}, rewriter->getBF16Type()),
                 APFloat::getZero(APFloat::BFloat())));
  auto pred = i8Vec(256);
  EXPECT_DEATH((void)hivm_regbaseintrins::buildMaxOp(lhs, lhs, pred, *rewriter),
               "Invalid david op element type");
}

TEST_F(ReportFatalErrorDeathTest, BuildVselInvalidElementType) {
  auto src = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({64}, rewriter->getI64Type()), int64_t(0)));
  auto pred = i8Vec(256);
  EXPECT_DEATH(
      (void)hivm_regbaseintrins::buildVselOp(src, src, pred, *rewriter),
      "Invalid vsel element type");
}

TEST_F(ReportFatalErrorDeathTest, Remove1DVectorHighDimsExpectsLeadingOnes) {
  auto src = rewriter->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(
                 VectorType::get({2, 64}, rewriter->getF32Type()), 0.0f));
  EXPECT_DEATH((void)hivm_regbaseintrins::remove1DVectorHighDims(loc(), src,
                                                                 *rewriter),
               "Expecting 1D vector when reducing high dimensions");
}

TEST_F(ReportFatalErrorDeathTest, GetVldsBrcDistInvalidBitLength) {
  EXPECT_DEATH((void)hivm_regbaseintrins::getVldsBrcDist(7),
               "Invalid data bit length");
}

TEST_F(ReportFatalErrorDeathTest, CreateVLVectorTypeUnsupportedDatatype) {
  EXPECT_DEATH(
      (void)hivm_regbaseintrins::createVLVectorType(rewriter->getIntegerType(3)),
      "unsupported datatype");
}

#endif // GTEST_HAS_DEATH_TEST

} // namespace
