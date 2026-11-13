//===- MmadCapabilityTest.cpp - Production Mmad queries ----------*- C++
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

#include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/NPUArchitecture.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace bishengir::npu;
using namespace bishengir::npu::mmad;
using namespace mlir;
using namespace mlir::hivm;

namespace {

class MmadCapabilityTest : public ::testing::Test {
protected:
  MLIRContext context;
  Type f16 = Float16Type::get(&context);
  Type f32 = Float32Type::get(&context);
  Type bf16 = BFloat16Type::get(&context);
  Type i8 = IntegerType::get(&context, 8);
  Type i32 = IntegerType::get(&context, 32);
  Type f8e4 = Float8E4M3FNType::get(&context);
  Type f8e5 = Float8E5M2Type::get(&context);

  NativeSupport queryMmad(NPUArchitectureKind arch, Type a, Type b, Type c,
                          bool ta, bool tb, bool hf32, bool i4, bool bias,
                          Type biasType) {
    auto storage =
        MmadCapabilityRequestBuilder::create(
            a, AddressSpace::L1, b, AddressSpace::L1, c, AddressSpace::L0C,
            biasType, AddressSpace::L1, ta, tb, hf32, i4, bias)
            .build();
    EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
    return queryNativeSupport(arch, storage.getRequest());
  }
};

TEST_F(MmadCapabilityTest, SharedFloatRowsAreNative) {
  for (NPUArchitectureKind arch :
       {NPUArchitectureKind::kDavC220, NPUArchitectureKind::kDavC310}) {
    EXPECT_EQ(
        queryMmad(arch, f16, f16, f32, false, false, false, false, false, f32),
        NativeSupport::Native);
    EXPECT_EQ(
        queryMmad(arch, f32, f32, f32, true, true, true, false, true, f32),
        NativeSupport::Native);
    EXPECT_EQ(
        queryMmad(arch, i8, i8, i32, false, true, false, false, true, i32),
        NativeSupport::Native);
  }
}

TEST_F(MmadCapabilityTest, PartialMissIsUnknown) {
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC220, f8e4, f8e4, f32, false,
                      false, false, false, false, f32),
            NativeSupport::Unknown);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC310, f8e4, f8e5, f32, false,
                      false, false, false, false, f32),
            NativeSupport::Unknown);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC220, i8, i8, i32, true, false,
                      false, true, false, f32),
            NativeSupport::Unknown);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC220, i8, i8, i32, false, false,
                      false, true, false, f32),
            NativeSupport::Native);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC310, i8, i8, i32, false, false,
                      false, true, false, f32),
            NativeSupport::Unknown);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC220, f32, f32, f32, false, true,
                      false, false, true, f16),
            NativeSupport::Unknown);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavC310, f32, f32, f32, false, true,
                      false, false, true, f16),
            NativeSupport::Native);
  EXPECT_EQ(queryMmad(NPUArchitectureKind::kDavM300, f16, f16, f32, false,
                      false, false, false, false, f32),
            NativeSupport::Unknown);
}

TEST_F(MmadCapabilityTest, BatchAndMmadMx) {
  auto batch =
      BatchMmadCapabilityRequestBuilder::create(
          f16, AddressSpace::L1, f16, AddressSpace::L1, f32, AddressSpace::L0C,
          f32, AddressSpace::L1, false, false, false, false, false)
          .build();
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, batch.getRequest()),
      NativeSupport::Native);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC220, batch.getRequest()),
      NativeSupport::Unknown);

  auto mx = MmadMxCapabilityRequestBuilder::create(
                f8e5, AddressSpace::L1, f8e4, AddressSpace::L1, i8,
                AddressSpace::L1, i8, AddressSpace::L1, f32, AddressSpace::L0C,
                f32, AddressSpace::L1, false, false, false,
                llvm::APInt(
                    64, static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
                llvm::APInt(
                    64, static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
                false, false, true)
                .build();
  EXPECT_TRUE(isCapabilityRequestWellFormed(mx.getRequest()));
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310, mx.getRequest()),
            NativeSupport::Native);
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC220, mx.getRequest()),
            NativeSupport::Unknown);

  auto fp4 =
      MmadMxCapabilityRequestBuilder::create(
          i8, AddressSpace::L1, i8, AddressSpace::L1, i8, AddressSpace::L1, i8,
          AddressSpace::L1, f32, AddressSpace::L0C, f32, AddressSpace::L1, true,
          true, true,
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP4E2M1_T)),
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP4E2M1_T)),
          true, true, true)
          .build();
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310, fp4.getRequest()),
            NativeSupport::Native);
  EXPECT_EQ(
      queryNativeSupport(
          NPUArchitectureKind::kDavC310,
          MmadMxCapabilityRequestBuilder::create(
              i8, AddressSpace::L1, i8, AddressSpace::L1, i8, AddressSpace::L1,
              i8, AddressSpace::L1, f32, AddressSpace::L0C, f32,
              AddressSpace::L1, false, false, false,
              llvm::APInt(
                  64, static_cast<uint64_t>(HIVMMatmulDataformat::FP4E2M1_T)),
              llvm::APInt(
                  64, static_cast<uint64_t>(HIVMMatmulDataformat::FP8E4M3_T)),
              true, true, true)
              .build()
              .getRequest()),
      NativeSupport::Unknown);

  auto unaligned =
      MmadMxCapabilityRequestBuilder::create(
          f8e5, AddressSpace::L1, f8e5, AddressSpace::L1, i8, AddressSpace::L1,
          i8, AddressSpace::L1, f32, AddressSpace::L0C, f32, AddressSpace::L1,
          false, false, false,
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          false, false, false)
          .build();
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, unaligned.getRequest()),
      NativeSupport::Unknown);
}

// Mirrors server63 dot_scaled IR:
// hfusion.matmul_mx {lhsFormat=2, rhsFormat=1}
// tensor<f8E4M3FN> x tensor<f8E5M2>, i8 scales, K=64, Ascend950/C310.
TEST_F(MmadCapabilityTest, DotScaledMixedFp8WithFormatAttrs) {
  auto mixedFp8FormatsPresent =
      MmadMxCapabilityRequestBuilder::create(
          f8e4, AddressSpace::L1, f8e5, AddressSpace::L1, i8, AddressSpace::L1,
          i8, AddressSpace::L1, f32, AddressSpace::L0C, f32, AddressSpace::L1,
          false, false, false,
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E4M3_T)),
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          true, true, true)
          .build();
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               mixedFp8FormatsPresent.getRequest()),
            NativeSupport::Native);

  auto mixedI8StorageFormatsPresent =
      MmadMxCapabilityRequestBuilder::create(
          i8, AddressSpace::L1, i8, AddressSpace::L1, i8, AddressSpace::L1, i8,
          AddressSpace::L1, f32, AddressSpace::L0C, f32, AddressSpace::L1,
          false, false, false,
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E4M3_T)),
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          true, true, true)
          .build();
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               mixedI8StorageFormatsPresent.getRequest()),
            NativeSupport::Native);

  auto mixedFp8NoFormatAttrs =
      MmadMxCapabilityRequestBuilder::create(
          f8e4, AddressSpace::L1, f8e5, AddressSpace::L1, i8, AddressSpace::L1,
          i8, AddressSpace::L1, f32, AddressSpace::L0C, f32, AddressSpace::L1,
          false, false, false,
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E4M3_T)),
          llvm::APInt(64,
                      static_cast<uint64_t>(HIVMMatmulDataformat::FP8E5M2_T)),
          false, false, true)
          .build();
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               mixedFp8NoFormatAttrs.getRequest()),
            NativeSupport::Native);
}

} // namespace
