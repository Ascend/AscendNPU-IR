//===- FixpipeCapabilityTest.cpp - Production Fixpipe queries ----*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/Contracts/DMA/Fixpipe/FixpipePreQuantContract.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/DMA/Fixpipe/FixpipeCapability.h"
#include "bishengir/Target/NPU/NPUArchitecture.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
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

CapabilityRequestStorage makeRequest(Type src, Type dst, int64_t dma,
                                     int64_t preQuant, int64_t preRelu,
                                     int64_t channelSplit) {
  CapabilityRequestStorage storage;
  storage.capability = CapabilityKind::Fixpipe;
  storage.values = {
      {Field::slot(0, Property::ElementType), src},
      {Field::slot(1, Property::ElementType), dst},
      {Field::parameter(Parameter::DmaMode),
       CapabilityValue{Domain::FixpipeDMAMode, dma}},
      {Field::parameter(Parameter::PreQuantMode),
       CapabilityValue{Domain::FixpipePreQuantMode, preQuant}},
      {Field::parameter(Parameter::PreRelu),
       CapabilityValue{Domain::FixpipePreReluMode, preRelu}},
      {Field::parameter(Parameter::ChannelSplit),
       CapabilityValue{Domain::Bool, channelSplit}}};
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
                      int64_t channelSplit) {
    auto storage =
        makeRequest(src, dst, dma, preQuant, preRelu, channelSplit);
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
    EXPECT_EQ(query(NPUArchitectureKind::kDavC310, i32, i8, dma, kS322I8,
                    kNoRelu, 0),
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
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, i32, i8, kNz2nz, kS322I8,
                  kNoRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd, kNoQuant,
                  kNoRelu, 1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f16, kNz2nd, kF322F16,
                  kNoRelu, 1),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd, kNoQuant,
                  kLeakyRelu, 0),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, f32, f32, kNz2nd,
                  kQF322F32Pre, kNoRelu, 0),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, f32, f32, kNz2nd,
                  kQF322F32Pre, kNoRelu, 0),
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
            NativeSupport::Unknown);
}

TEST_F(FixpipeCapabilityTest, MalformedRequestIsInvalid) {
  auto missing = makeRequest(f32, f32, kNz2nd, kNoQuant, kNoRelu, 0);
  missing.values.pop_back();
  EXPECT_FALSE(isCapabilityRequestWellFormed(missing.getRequest()));
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               missing.getRequest()),
            NativeSupport::InvalidRequest);

  auto badDma = makeRequest(f32, f32, 99, kNoQuant, kNoRelu, 0);
  EXPECT_FALSE(isCapabilityRequestWellFormed(badDma.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, badDma.getRequest()),
      NativeSupport::InvalidRequest);

  auto nullType = makeRequest(Type(), f32, kNz2nd, kNoQuant, kNoRelu, 0);
  EXPECT_FALSE(isCapabilityRequestWellFormed(nullType.getRequest()));
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC220,
                               nullType.getRequest()),
            NativeSupport::InvalidRequest);
}

TEST_F(FixpipeCapabilityTest, RegisteredRowsSatisfyPreQuantContract) {
  // The counterexample below is expected to fail the contract. Keep its
  // diagnostic off the test log.
  context.getDiagEngine().registerHandler(
      [](Diagnostic &) { return success(); });

  Type probes[] = {f16, f32, f64, bf16, i8, i32};
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
            auto storage =
                makeRequest(src, dst, kNz2nd, preQuantValues.front(), kNoRelu, 0);
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

} // namespace
