//===- CubeLoadCapabilityTest.cpp - CubeLoad capability queries -*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/NPUArchitecture.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using bishengir::npu::NPUArchitectureKind;
using bishengir::npu::cubeload::isCapabilityRequestWellFormed;
using bishengir::npu::cubeload::NativeSupport;
using bishengir::npu::cubeload::queryNativeSupport;
using hivm::AddressSpace;

namespace {

class CubeLoadCapabilityTest : public ::testing::Test {
protected:
  MLIRContext context;
  Type f16 = Float16Type::get(&context);
  Type bf16 = BFloat16Type::get(&context);
  Type f32 = Float32Type::get(&context);
  Type f8 = Float8E4M3FNType::get(&context);
  Type i8 = IntegerType::get(&context, 8);
  Type i64 = IntegerType::get(&context, 64);

  template <typename Builder>
  NativeSupport query(NPUArchitectureKind architecture,
                      const Builder &builder) {
    auto storage = builder.build();
    EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
    return queryNativeSupport(architecture, storage.getRequest());
  }
};

TEST_F(CubeLoadCapabilityTest, Nd2NzRowsFollowTheLibrary) {
  auto row = [&](Type type, bool initOutBuffer = false) {
    return bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::create(
        type, AddressSpace::GM, type, AddressSpace::L1, initOutBuffer);
  };
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row(f16)),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row(f32, true)),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, row(f8)),
            NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row(f8)),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row(i64)),
            NativeSupport::Unknown);
  auto wrongDst =
      bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::create(
          f16, AddressSpace::GM, f16, AddressSpace::UB, false);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, wrongDst),
            NativeSupport::Unknown);
  auto mixed = bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::create(
      f32, AddressSpace::GM, f16, AddressSpace::L1, false);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, mixed),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavM300, row(f16)),
            NativeSupport::Unknown);
}

TEST_F(CubeLoadCapabilityTest, Nd2NzBatchIsC310And920) {
  auto row =
      bishengir::npu::cubeload::Nd2NzBatchCapabilityRequestBuilder::create(
          f32, AddressSpace::GM, f32, AddressSpace::L1);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, row), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDav920R1, row), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row), NativeSupport::Unknown);
  auto fp8 =
      bishengir::npu::cubeload::Nd2NzBatchCapabilityRequestBuilder::create(
          f8, AddressSpace::GM, f8, AddressSpace::L1);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, fp8), NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDav920R1, fp8), NativeSupport::Native);
}

TEST_F(CubeLoadCapabilityTest, Dav920R1AcceptsEveryWellFormedRequest) {
  EXPECT_EQ(
      query(NPUArchitectureKind::kDav920R1,
            bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::create(
                i64, AddressSpace::UB, f16, AddressSpace::GM, true)),
      NativeSupport::Native);
  EXPECT_EQ(
      query(NPUArchitectureKind::kDav920R1,
            bishengir::npu::cubeload::Nz2NdCapabilityRequestBuilder::create(
                f32, AddressSpace::L1, f32, AddressSpace::UB)),
      NativeSupport::Native);
  EXPECT_EQ(
      query(NPUArchitectureKind::kDav920R1,
            bishengir::npu::cubeload::L12UBCapabilityRequestBuilder::create(
                i8, AddressSpace::GM, i8, AddressSpace::GM)),
      NativeSupport::Native);
  EXPECT_EQ(
      query(NPUArchitectureKind::kDav920R1,
            bishengir::npu::cubeload::Nchw2Nc1hwc0CapabilityRequestBuilder::
                create(i8, AddressSpace::GM, i8, AddressSpace::L1)),
      NativeSupport::Native);
  EXPECT_EQ(
      query(NPUArchitectureKind::kDav920R1,
            bishengir::npu::cubeload::LoadScaleCapabilityRequestBuilder::create(
                f32, AddressSpace::GM, f32, AddressSpace::L1, true)),
      NativeSupport::Native);
}

TEST_F(CubeLoadCapabilityTest, Nz2NdIsC220L1ToGm) {
  auto row = bishengir::npu::cubeload::Nz2NdCapabilityRequestBuilder::create(
      i64, AddressSpace::L1, i64, AddressSpace::GM);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, row), NativeSupport::Unknown);
  auto ub = bishengir::npu::cubeload::Nz2NdCapabilityRequestBuilder::create(
      f32, AddressSpace::L1, f32, AddressSpace::UB);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, ub), NativeSupport::Unknown);
}

TEST_F(CubeLoadCapabilityTest, L12UbIsC310L1ToUb) {
  auto row = bishengir::npu::cubeload::L12UBCapabilityRequestBuilder::create(
      i64, AddressSpace::L1, i64, AddressSpace::UB);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, row), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, row), NativeSupport::Unknown);
}

TEST_F(CubeLoadCapabilityTest, GroupedNchwLoadsAreC310Float) {
  auto nc1 =
      bishengir::npu::cubeload::Nchw2Nc1hwc0CapabilityRequestBuilder::create(
          bf16, AddressSpace::GM, bf16, AddressSpace::L1);
  auto c1 =
      bishengir::npu::cubeload::Nchw2C1hwnc0CapabilityRequestBuilder::create(
          f32, AddressSpace::GM, f32, AddressSpace::L1);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, nc1), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, c1), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, nc1), NativeSupport::Unknown);
  auto i8Row =
      bishengir::npu::cubeload::Nchw2Nc1hwc0CapabilityRequestBuilder::create(
          i8, AddressSpace::GM, i8, AddressSpace::L1);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, i8Row),
            NativeSupport::Unknown);
}

TEST_F(CubeLoadCapabilityTest, LoadScaleIsUntransposedI8OnC310) {
  auto plain =
      bishengir::npu::cubeload::LoadScaleCapabilityRequestBuilder::create(
          i8, AddressSpace::GM, i8, AddressSpace::L1, false);
  auto transposed =
      bishengir::npu::cubeload::LoadScaleCapabilityRequestBuilder::create(
          i8, AddressSpace::GM, i8, AddressSpace::L1, true);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, plain), NativeSupport::Native);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC310, transposed),
            NativeSupport::Unknown);
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, plain),
            NativeSupport::Unknown);
}

TEST_F(CubeLoadCapabilityTest, MemrefBindingMatchesCreate) {
  context.loadDialect<hivm::HIVMDialect, memref::MemRefDialect,
                      tensor::TensorDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto srcType =
      MemRefType::get({16, 16}, f16, MemRefLayoutAttrInterface(),
                      hivm::AddressSpaceAttr::get(&context, AddressSpace::GM));
  auto dstType =
      MemRefType::get({2, 2, 16, 16}, f16, MemRefLayoutAttrInterface(),
                      hivm::AddressSpaceAttr::get(&context, AddressSpace::L1));
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());
  Value src = builder.create<memref::AllocOp>(loc, srcType);
  Value dst = builder.create<memref::AllocOp>(loc, dstType);
  auto op = builder.create<hivm::ND2NZOp>(loc, TypeRange{}, src, dst,
                                          builder.getUnitAttr());
  auto bound =
      bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::from(op);
  ASSERT_TRUE(bound.has_value());
  EXPECT_EQ(query(NPUArchitectureKind::kDavC220, *bound),
            NativeSupport::Native);

  auto tensorType = RankedTensorType::get({16, 16}, f16);
  Value tensorSrc =
      builder.create<tensor::EmptyOp>(loc, tensorType.getShape(), f16);
  Value tensorDst = builder.create<tensor::EmptyOp>(
      loc, RankedTensorType::get({2, 2, 16, 16}, f16).getShape(), f16);
  auto unscoped = builder.create<hivm::ND2NZOp>(
      loc, TypeRange{}, tensorSrc, tensorDst, builder.getUnitAttr());
  auto unbound =
      bishengir::npu::cubeload::Nd2NzCapabilityRequestBuilder::from(unscoped);
  ASSERT_TRUE(unbound.has_value());
  EXPECT_FALSE(isCapabilityRequestWellFormed(unbound->build().getRequest()));
}

} // namespace
