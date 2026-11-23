//===- FixpipeCapabilityTest.cpp - Production Fixpipe queries ----*- C++
//-*-===//
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

#include "bishengir/Target/NPU/DMA/Fixpipe/FixpipeCapability.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/Contracts/DMA/Fixpipe/FixpipePreQuantContract.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/NPUArchitecture.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace bishengir::npu;
using namespace mlir;

using Field = CapabilityRequestFieldDesc;
using Domain = CapabilityValueDomainKind;
using Parameter = CapabilityParameterKind;
using Property = CapabilitySlotPropertyKind;

namespace {

// ODS enum values from HIVMAttrs.td. The query stores these raw values.
constexpr int64_t kNz2nd = 0;
constexpr int64_t kNz2dn = 1;
constexpr int64_t kNz2nz = 2;
constexpr int64_t kNoQuant = 0;
constexpr int64_t kF322F16 = 1;
constexpr int64_t kS322I8 = 9;
constexpr int64_t kQF322F32Pre = 15;
constexpr int64_t kF322BF16 = 16;
constexpr int64_t kNoRelu = 0;
constexpr int64_t kNormalRelu = 1;
constexpr int64_t kLeakyRelu = 2;
// HIVM address-space enum values.
constexpr int64_t kGM = 1;
constexpr int64_t kL1 = 2;
constexpr int64_t kL0C = 5;
constexpr int64_t kUB = 6;

CapabilityRequestStorage
makeRequest(Type src, Type dst, int64_t dma, int64_t preQuant, int64_t preRelu,
            int64_t channelSplit, int64_t srcSpace = kL0C,
            int64_t dstSpace = kGM, int64_t quantScalePresent = 0) {
  CapabilityRequestStorage storage;
  storage.capability = CapabilityKind::Fixpipe;
  storage.values = {{Field::slot(0, Property::ElementType), src},
                    {Field::slot(0, Property::AddressSpace),
                     CapabilityValue{Domain::AddressSpace, srcSpace}},
                    {Field::slot(1, Property::ElementType), dst},
                    {Field::slot(1, Property::AddressSpace),
                     CapabilityValue{Domain::AddressSpace, dstSpace}},
                    {Field::parameter(Parameter::DmaMode),
                     CapabilityValue{Domain::FixpipeDMAMode, dma}},
                    {Field::parameter(Parameter::PreQuantMode),
                     CapabilityValue{Domain::FixpipePreQuantMode, preQuant}},
                    {Field::parameter(Parameter::PreRelu),
                     CapabilityValue{Domain::FixpipePreReluMode, preRelu}},
                    {Field::parameter(Parameter::ChannelSplit),
                     CapabilityValue{Domain::Bool, channelSplit}},
                    {Field::parameter(Parameter::QuantScalePresent),
                     CapabilityValue{Domain::Bool, quantScalePresent}}};
  return storage;
}

class FixpipeCapabilityTest : public ::testing::Test {
protected:
  MLIRContext context;
  Type f16 = Float16Type::get(&context);
  Type f32 = Float32Type::get(&context);
  Type f64 = Float64Type::get(&context);
  Type bf16 = BFloat16Type::get(&context);
  Type i8 = IntegerType::get(&context, 8);
  Type i32 = IntegerType::get(&context, 32);

  NativeSupport query(NPUArchitectureKind architecture, Type src, Type dst,
                      int64_t dma, int64_t preQuant, int64_t preRelu,
                      int64_t channelSplit, int64_t srcSpace = kL0C,
                      int64_t dstSpace = kGM, int64_t quantScalePresent = 0) {
    auto storage = makeRequest(src, dst, dma, preQuant, preRelu, channelSplit,
                               srcSpace, dstSpace, quantScalePresent);
    EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
    return queryNativeSupport(architecture, storage.getRequest());
  }
};

TEST_F(FixpipeCapabilityTest, RegisteredRowsAreNative) {
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, i32, i32, kNz2nd, kNoQuant,
                  kNoRelu, 0),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, i32, i8, kNz2nd, kS322I8,
                  kNoRelu, 0),
            NativeSupport::Native);
  for (int64_t dma : {kNz2nd, kNz2nz}) {
    EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f16, dma, kF322F16,
                    kNoRelu, 0),
              NativeSupport::Native);
    EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, bf16, dma, kF322BF16,
                    kNoRelu, 0),
              NativeSupport::Native);
  }
  for (int64_t dma : {kNz2nd, kNz2dn, kNz2nz}) {
    EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, dma, kNoQuant,
                    kNoRelu, 0),
              NativeSupport::Native);
    EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, dma, kQF322F32Pre,
                    kNoRelu, 0),
              NativeSupport::Native);
    EXPECT_EQ(query(NPUArchitectureKind::kDavC310, i32, i32, dma, kNoQuant,
                    kNoRelu, 0),
              NativeSupport::Native);
    EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f16, dma, kF322F16,
                    kNoRelu, 0),
              NativeSupport::Native);
    EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, bf16, dma, kF322BF16,
                    kNoRelu, 0),
              NativeSupport::Native);
  }
  for (int64_t dma : {kNz2nd, kNz2dn}) {
    EXPECT_EQ(
        query(NPUArchitectureKind::kDavC310, i32, i8, dma, kS322I8, kNoRelu, 0),
        NativeSupport::Native);
  }
}

TEST_F(FixpipeCapabilityTest, PartialMissIsUnknown) {
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2dn, kNoQuant,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nz, kNoQuant,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, i32, i8, kNz2nz, kS322I8,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f16, kNz2nd, kF322F16,
                  kNormalRelu, 0),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, i32, i8, kNz2nz, kS322I8,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  // channel_split=true is L0C→L1 only. The default destination here is GM.
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 1, kL0C, kL1),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 1, kL0C, kUB),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f16, kNz2nd, kF322F16,
                  kNoRelu, 1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kLeakyRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kQF322F32Pre,
                  kNoRelu, 0),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kQF322F32Pre,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  // Separately supported fields do not form a cross product.
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, i32, f16, kNz2nd, kF322F16,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, bf16, kNz2nd, kF322F16,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavM300, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDav920R1, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0),
            NativeSupport::Native);
  // C220 is L0C→GM only. C310 also allows L1 and UB.
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0, kL0C, kL1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0, kL0C, kUB),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0, kL0C, kL1),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0, kL0C, kUB),
            NativeSupport::Native);
  // quant_scale presence is C310 only.
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0, kL0C, kGM, 1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 0, kL0C, kGM, 1),
            NativeSupport::Native);
}

TEST_F(FixpipeCapabilityTest, MalformedRequestIsInvalid) {
  auto missing = makeRequest(f32, f32, kNz2nd, kNoQuant, kNoRelu, 0);
  missing.values.pop_back();
  EXPECT_FALSE(isCapabilityRequestWellFormed(missing.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, missing.getRequest()),
      NativeSupport::InvalidRequest);

  auto badDma = makeRequest(f32, f32, 99, kNoQuant, kNoRelu, 0);
  EXPECT_FALSE(isCapabilityRequestWellFormed(badDma.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, badDma.getRequest()),
      NativeSupport::InvalidRequest);

  auto nullType = makeRequest(Type(), f32, kNz2nd, kNoQuant, kNoRelu, 0);
  EXPECT_FALSE(isCapabilityRequestWellFormed(nullType.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC220, nullType.getRequest()),
      NativeSupport::InvalidRequest);
}

TEST_F(FixpipeCapabilityTest, RegisteredRowsSatisfyPreQuantContract) {
  // The counterexample below is expected to fail the contract. Keep its
  // diagnostic off the test log.
  context.getDiagEngine().registerHandler(
      [](Diagnostic &) { return success(); });

  Type i16 = IntegerType::get(&context, 16);
  Type ui8 = IntegerType::get(&context, 8, IntegerType::Unsigned);
  Type probes[] = {f16, f32, f64, bf16, i8, i16, i32, ui8};
  auto accept = [&](Type src, Type dst, int64_t preQuant) {
    return succeeded(verifyFixpipePreQuantSignature(
        static_cast<hivm::FixpipePreQuantMode>(preQuant), src, dst,
        [&] { return emitError(UnknownLoc::get(&context)); }));
  };
  EXPECT_FALSE(accept(f32, bf16, kF322F16));

  for (NPUArchitectureKind architecture :
       {NPUArchitectureKind::kDavC220, NPUArchitectureKind::kDavC310}) {
    const auto *profile = lookupArchitectureCapabilityProfile(architecture);
    ASSERT_NE(profile, nullptr);
    for (const auto &coverage : profile->capabilities) {
      EXPECT_EQ(coverage.completeness, ProfileCompleteness::Partial);
      for (const auto &supportCase : coverage.cases) {
        bool sawPair = false;
        SmallVector<int64_t> preQuantValues;
        for (const auto &constraint : supportCase.constraints) {
          if (constraint.field.kind == CapabilityRequestFieldKind::Parameter &&
              constraint.field.identity ==
                  static_cast<unsigned>(Parameter::PreQuantMode))
            preQuantValues.assign(constraint.allowedValues.begin(),
                                  constraint.allowedValues.end());
        }
        ASSERT_FALSE(preQuantValues.empty());
        for (Type src : probes) {
          for (Type dst : probes) {
            auto storage = makeRequest(src, dst, kNz2nd, preQuantValues.front(),
                                       kNoRelu, 0);
            bool matched = false;
            for (const auto &signature : supportCase.signatures)
              matched = matched || signature.matches(storage.getRequest());
            if (!matched)
              continue;
            sawPair = true;
            for (int64_t preQuant : preQuantValues)
              EXPECT_TRUE(accept(src, dst, preQuant))
                  << stringifyNPUArchitecture(architecture).str();
          }
        }
        EXPECT_TRUE(sawPair);
      }
    }
  }
}

std::optional<int64_t> rawParameter(const CapabilityRequestStorage &storage,
                                    Parameter parameter) {
  for (const CapabilityFieldValue &field : storage.values) {
    if (field.field.kind != CapabilityRequestFieldKind::Parameter ||
        field.field.identity != static_cast<unsigned>(parameter))
      continue;
    const auto *value = std::get_if<CapabilityValue>(&field.value);
    if (!value)
      return std::nullopt;
    return value->rawValue;
  }
  return std::nullopt;
}

Type slotType(const CapabilityRequestStorage &storage, unsigned position) {
  for (const CapabilityFieldValue &field : storage.values) {
    if (field.field.kind == CapabilityRequestFieldKind::Slot &&
        field.field.position == position)
      return std::get<Type>(field.value);
  }
  return Type();
}

bool sameRequest(const CapabilityRequestStorage &lhs,
                 const CapabilityRequestStorage &rhs) {
  if (lhs.capability != rhs.capability ||
      lhs.values.size() != rhs.values.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.values, rhs.values)) {
    if (left.field.kind != right.field.kind ||
        left.field.position != right.field.position ||
        left.field.identity != right.field.identity ||
        left.value.index() != right.value.index())
      return false;
    if (const auto *type = std::get_if<Type>(&left.value)) {
      if (*type != std::get<Type>(right.value))
        return false;
      continue;
    }
    const auto &leftValue = std::get<CapabilityValue>(left.value);
    const auto &rightValue = std::get<CapabilityValue>(right.value);
    if (leftValue.domain != rightValue.domain ||
        leftValue.rawValue != rightValue.rawValue)
      return false;
  }
  return true;
}

std::string printOp(Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  op->print(os, OpPrintingFlags().useLocalScope());
  return text;
}

class FixpipeBindingTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<hivm::HIVMDialect, tensor::TensorDialect,
                        memref::MemRefDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
    builder = OpBuilder::atBlockBegin(module->getBody());
    f16 = Float16Type::get(&context);
    f32 = Float32Type::get(&context);
    i8 = IntegerType::get(&context, 8);
    i32 = IntegerType::get(&context, 32);
  }

  Value tensorValue(Type elementType) {
    auto type = RankedTensorType::get({4, 4}, elementType);
    return builder.create<tensor::EmptyOp>(builder.getUnknownLoc(),
                                           type.getShape(), elementType);
  }

  Value memrefValue(Type elementType,
                    hivm::AddressSpace space = hivm::AddressSpace::GM) {
    auto type =
        MemRefType::get({4, 4}, elementType, MemRefLayoutAttrInterface(),
                        hivm::AddressSpaceAttr::get(&context, space));
    return builder.create<memref::AllocOp>(builder.getUnknownLoc(), type);
  }

  hivm::FixpipeOp makeFixpipe(Value src, Value dst,
                              hivm::FixpipeDMAModeAttr dma,
                              hivm::FixpipePreQuantModeAttr preQuant,
                              hivm::FixpipePreReluModeAttr preRelu,
                              BoolAttr channelSplit) {
    return builder.create<hivm::FixpipeOp>(
        builder.getUnknownLoc(), TypeRange{}, src, dst, dma,
        hivm::FixpipeDualDstModeAttr{}, hivm::FixpipeSubBlockAttr{}, preQuant,
        preRelu, channelSplit, BoolAttr{}, Value{});
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  OpBuilder builder{&context};
  Type f16;
  Type f32;
  Type i8;
  Type i32;
};

TEST_F(FixpipeBindingTest, TensorAndMemrefElementTypesMatchCreate) {
  Value src = memrefValue(f32, hivm::AddressSpace::L0C);
  Value dst = memrefValue(f16, hivm::AddressSpace::GM);
  auto op = makeFixpipe(
      src, dst,
      hivm::FixpipeDMAModeAttr::get(&context, hivm::FixpipeDMAMode::NZ2ND),
      hivm::FixpipePreQuantModeAttr::get(&context,
                                         hivm::FixpipePreQuantMode::F322F16),
      hivm::FixpipePreReluModeAttr::get(&context,
                                        hivm::FixpipePreReluMode::NO_RELU),
      builder.getBoolAttr(false));
  const std::string before = printOp(op);
  auto bound = FixpipeCapabilityRequestBuilder::from(op);
  ASSERT_TRUE(bound);
  auto fromOp = bound->build();
  auto created =
      FixpipeCapabilityRequestBuilder::create(
          f32, hivm::AddressSpace::L0C, f16, hivm::AddressSpace::GM,
          hivm::FixpipeDMAMode::NZ2ND, hivm::FixpipePreQuantMode::F322F16,
          hivm::FixpipePreReluMode::NO_RELU, false, false)
          .build();
  EXPECT_TRUE(sameRequest(fromOp, created));
  EXPECT_EQ(slotType(fromOp, 0), f32);
  EXPECT_EQ(slotType(fromOp, 1), f16);
  EXPECT_NE(slotType(fromOp, 0), src.getType());
  EXPECT_NE(slotType(fromOp, 1), dst.getType());
  EXPECT_TRUE(isCapabilityRequestWellFormed(fromOp.getRequest()));
  EXPECT_EQ(printOp(op), before);
}

TEST_F(FixpipeBindingTest, OverrideQueriesCandidateAndDoesNotModifyOp) {
  Value src = memrefValue(f32, hivm::AddressSpace::L0C);
  Value dst = memrefValue(f32, hivm::AddressSpace::GM);
  auto op = makeFixpipe(
      src, dst,
      hivm::FixpipeDMAModeAttr::get(&context, hivm::FixpipeDMAMode::NZ2ND),
      hivm::FixpipePreQuantModeAttr::get(&context,
                                         hivm::FixpipePreQuantMode::NO_QUANT),
      hivm::FixpipePreReluModeAttr::get(&context,
                                        hivm::FixpipePreReluMode::NO_RELU),
      builder.getBoolAttr(false));
  const std::string before = printOp(op);
  auto bound = FixpipeCapabilityRequestBuilder::from(op);
  ASSERT_TRUE(bound);
  auto candidate = bound->dstElementType(f16)
                       .dmaMode(hivm::FixpipeDMAMode::NZ2DN)
                       .preQuantMode(hivm::FixpipePreQuantMode::F322F16)
                       .build();
  auto created =
      FixpipeCapabilityRequestBuilder::create(
          f32, hivm::AddressSpace::L0C, f16, hivm::AddressSpace::GM,
          hivm::FixpipeDMAMode::NZ2DN, hivm::FixpipePreQuantMode::F322F16,
          hivm::FixpipePreReluMode::NO_RELU, false, false)
          .build();
  EXPECT_TRUE(sameRequest(candidate, created));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC220, candidate.getRequest()),
      NativeSupport::Unknown);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, candidate.getRequest()),
      NativeSupport::Native);
  EXPECT_EQ(op.getDst().getType(), dst.getType());
  EXPECT_EQ(op.getDmaMode(), hivm::FixpipeDMAMode::NZ2ND);
  EXPECT_EQ(printOp(op), before);
}

TEST_F(FixpipeBindingTest, OmittedDefaultsStayDistinctFromExplicitValues) {
  Value src = memrefValue(f32, hivm::AddressSpace::L0C);
  Value dst = memrefValue(f32, hivm::AddressSpace::GM);
  auto omitted = makeFixpipe(src, dst, {}, {}, {}, {});
  omitted->removeAttr("dma_mode");
  omitted->removeAttr("pre_quant");
  omitted->removeAttr("pre_relu");
  omitted->removeAttr("channel_split");
  EXPECT_FALSE(omitted->getAttr("dma_mode"));
  EXPECT_FALSE(omitted->getAttr("pre_relu"));
  EXPECT_FALSE(omitted->getAttr("channel_split"));
  auto omittedRequest = FixpipeCapabilityRequestBuilder::from(omitted);
  ASSERT_TRUE(omittedRequest);
  auto omittedStorage = omittedRequest->build();
  EXPECT_EQ(rawParameter(omittedStorage, Parameter::DmaMode), kNz2nz);
  EXPECT_EQ(rawParameter(omittedStorage, Parameter::PreQuantMode), kNoQuant);
  EXPECT_EQ(rawParameter(omittedStorage, Parameter::PreRelu), kNoRelu);
  EXPECT_EQ(rawParameter(omittedStorage, Parameter::ChannelSplit), 0);
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               omittedStorage.getRequest()),
            NativeSupport::Native);

  auto explicitOp = makeFixpipe(
      src, memrefValue(f32, hivm::AddressSpace::L1),
      hivm::FixpipeDMAModeAttr::get(&context, hivm::FixpipeDMAMode::NZ2ND),
      hivm::FixpipePreQuantModeAttr::get(&context,
                                         hivm::FixpipePreQuantMode::NO_QUANT),
      hivm::FixpipePreReluModeAttr::get(&context,
                                        hivm::FixpipePreReluMode::NORMAL_RELU),
      builder.getBoolAttr(true));
  auto explicitRequest = FixpipeCapabilityRequestBuilder::from(explicitOp);
  ASSERT_TRUE(explicitRequest);
  auto explicitStorage = explicitRequest->build();
  EXPECT_EQ(rawParameter(explicitStorage, Parameter::PreRelu), kNormalRelu);
  EXPECT_EQ(rawParameter(explicitStorage, Parameter::ChannelSplit), 1);
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               explicitStorage.getRequest()),
            NativeSupport::Native);
  EXPECT_FALSE(sameRequest(omittedStorage, explicitStorage));
}

TEST_F(FixpipeBindingTest, UnscopedTensorLeavesAddressSpaceUnset) {
  Value src = tensorValue(f32);
  Value dst = tensorValue(f32);
  auto op = makeFixpipe(
      src, dst,
      hivm::FixpipeDMAModeAttr::get(&context, hivm::FixpipeDMAMode::NZ2ND),
      hivm::FixpipePreQuantModeAttr::get(&context,
                                         hivm::FixpipePreQuantMode::NO_QUANT),
      hivm::FixpipePreReluModeAttr::get(&context,
                                        hivm::FixpipePreReluMode::NO_RELU),
      builder.getBoolAttr(false));
  auto bound = FixpipeCapabilityRequestBuilder::from(op);
  ASSERT_TRUE(bound);
  EXPECT_FALSE(isCapabilityRequestWellFormed(bound->build().getRequest()));
  auto set = bound->srcAddressSpace(hivm::AddressSpace::L0C)
                 .dstAddressSpace(hivm::AddressSpace::GM)
                 .build();
  EXPECT_TRUE(isCapabilityRequestWellFormed(set.getRequest()));
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC220, set.getRequest()),
            NativeSupport::Native);
}

TEST_F(FixpipeBindingTest, UnshapedValueIsNotARequest) {
  context.loadDialect<arith::ArithDialect>();
  Value scalar = builder.create<arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getI32IntegerAttr(1));
  Value dst = tensorValue(i8);
  auto op = makeFixpipe(
      scalar, dst,
      hivm::FixpipeDMAModeAttr::get(&context, hivm::FixpipeDMAMode::NZ2ND),
      hivm::FixpipePreQuantModeAttr::get(&context,
                                         hivm::FixpipePreQuantMode::S322I8),
      hivm::FixpipePreReluModeAttr::get(&context,
                                        hivm::FixpipePreReluMode::NO_RELU),
      builder.getBoolAttr(false));
  // A scalar has no element type. That is an illegal binding input, not a
  // hardware miss and not an ODS default. The shaped operand getter casts,
  // so compare the raw operand instead of printing the op.
  EXPECT_FALSE(FixpipeCapabilityRequestBuilder::from(op));
  EXPECT_EQ(op->getOperand(0), scalar);
  EXPECT_EQ(op->getOperand(1), dst);
}

TEST_F(FixpipeBindingTest, NullTypeStaysInvalidRequest) {
  auto storage =
      FixpipeCapabilityRequestBuilder::create(
          Type(), hivm::AddressSpace::L0C, f32, hivm::AddressSpace::GM,
          hivm::FixpipeDMAMode::NZ2NZ, hivm::FixpipePreQuantMode::NO_QUANT,
          hivm::FixpipePreReluMode::NO_RELU, false, false)
          .build();
  EXPECT_FALSE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, storage.getRequest()),
      NativeSupport::InvalidRequest);
}

} // namespace
