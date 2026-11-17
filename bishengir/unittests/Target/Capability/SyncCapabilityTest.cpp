//===- SyncCapabilityTest.cpp - Production sync queries --------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Target/NPU/NPUArchitecture.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;
using bishengir::npu::NPUArchitectureKind;
using bishengir::npu::sync::CapabilityKind;
using bishengir::npu::sync::CapabilityParameterKind;
using bishengir::npu::sync::CapabilityRequestFieldDesc;
using bishengir::npu::sync::CapabilityRequestStorage;
using bishengir::npu::sync::CapabilityValue;
using bishengir::npu::sync::CapabilityValueDomainKind;
using bishengir::npu::sync::GetBufCapabilityRequestBuilder;
using bishengir::npu::sync::isCapabilityRequestWellFormed;
using bishengir::npu::sync::NativeSupport;
using bishengir::npu::sync::queryNativeSupport;
using bishengir::npu::sync::RlsBufCapabilityRequestBuilder;
using bishengir::npu::sync::TryWaitCapabilityRequestBuilder;

namespace {

// ODS enum value from HIVMAttrs.td. The query stores this raw value.
constexpr int64_t kIntraBlock = 1;

CapabilityRequestStorage makeTryWaitRequest(int64_t syncMode) {
  CapabilityRequestStorage storage;
  storage.capability = CapabilityKind::TryWait;
  storage.values = {
      {CapabilityRequestFieldDesc::parameter(CapabilityParameterKind::SyncMode),
       CapabilityValue{CapabilityValueDomainKind::TryWaitSyncMode, syncMode}}};
  return storage;
}

CapabilityRequestStorage makeFieldFreeRequest(CapabilityKind capability) {
  CapabilityRequestStorage storage;
  storage.capability = capability;
  return storage;
}

TEST(SyncCapabilityTest, RegisteredRowsAreNative) {
  auto tryWait = makeTryWaitRequest(kIntraBlock);
  ASSERT_TRUE(isCapabilityRequestWellFormed(tryWait.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, tryWait.getRequest()),
      NativeSupport::Native);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDav920R1, tryWait.getRequest()),
      NativeSupport::Native);

  auto getBuf = makeFieldFreeRequest(CapabilityKind::GetBuf);
  ASSERT_TRUE(isCapabilityRequestWellFormed(getBuf.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, getBuf.getRequest()),
      NativeSupport::Native);

  auto rlsBuf = makeFieldFreeRequest(CapabilityKind::RlsBuf);
  ASSERT_TRUE(isCapabilityRequestWellFormed(rlsBuf.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, rlsBuf.getRequest()),
      NativeSupport::Native);
}

TEST(SyncCapabilityTest, ForwardCompatOpensRegBaseRows) {
  auto getBuf = makeFieldFreeRequest(CapabilityKind::GetBuf);
  auto rlsBuf = makeFieldFreeRequest(CapabilityKind::RlsBuf);

  // dav-920r1 is not surveyed for the sync intrinsics: every confirmed
  // regbase row stays open under the forward-compatibility policy.
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDav920R1, getBuf.getRequest()),
      NativeSupport::Native);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDav920R1, rlsBuf.getRequest()),
      NativeSupport::Native);
}

TEST(SyncCapabilityTest, PartialMissIsUnknown) {
  auto tryWait = makeTryWaitRequest(kIntraBlock);
  auto getBuf = makeFieldFreeRequest(CapabilityKind::GetBuf);
  auto rlsBuf = makeFieldFreeRequest(CapabilityKind::RlsBuf);

  // Memory-based architectures carry surveyed empty coverages: Unknown like
  // the old gates, and distinct from an unsurveyed missing profile.
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC220, tryWait.getRequest()),
      NativeSupport::Unknown);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC220, getBuf.getRequest()),
      NativeSupport::Unknown);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC220, rlsBuf.getRequest()),
      NativeSupport::Unknown);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavM300, tryWait.getRequest()),
      NativeSupport::Unknown);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavM300, getBuf.getRequest()),
      NativeSupport::Unknown);
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavM300, rlsBuf.getRequest()),
      NativeSupport::Unknown);
}

TEST(SyncCapabilityTest, MalformedRequestIsInvalid) {
  // A try_wait request without the sync_mode field is not the schema.
  auto missing = makeFieldFreeRequest(CapabilityKind::TryWait);
  EXPECT_FALSE(isCapabilityRequestWellFormed(missing.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, missing.getRequest()),
      NativeSupport::InvalidRequest);

  // A sync_mode outside the exposed enum is not the schema.
  auto badMode = makeTryWaitRequest(99);
  EXPECT_FALSE(isCapabilityRequestWellFormed(badMode.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, badMode.getRequest()),
      NativeSupport::InvalidRequest);
}

std::string printOp(Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  op->print(os, OpPrintingFlags().useLocalScope());
  return text;
}

class SyncBindingTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<hivm::HIVMDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
    builder = OpBuilder::atBlockBegin(module->getBody());
  }

  hivm::TryWaitOp makeTryWait(OpFoldResult flagId,
                              hivm::TryWaitSyncMode syncMode) {
    return builder.create<hivm::TryWaitOp>(builder.getUnknownLoc(), flagId,
                                           syncMode);
  }

  hivm::GetBufOp makeGetBuf(OpFoldResult bufId) {
    return builder.create<hivm::GetBufOp>(
        builder.getUnknownLoc(),
        hivm::PipeAttr::get(&context, hivm::PIPE::PIPE_V), bufId,
        builder.getBoolAttr(false));
  }

  hivm::RlsBufOp makeRlsBuf(OpFoldResult bufId) {
    return builder.create<hivm::RlsBufOp>(
        builder.getUnknownLoc(),
        hivm::PipeAttr::get(&context, hivm::PIPE::PIPE_V), bufId,
        builder.getBoolAttr(true));
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  OpBuilder builder{&context};
};

TEST_F(SyncBindingTest, TryWaitFromMatchesCreate) {
  auto op = makeTryWait(builder.getI64IntegerAttr(31),
                        hivm::TryWaitSyncMode::INTRA_BLOCK);
  const std::string before = printOp(op);
  auto bound = TryWaitCapabilityRequestBuilder::from(op);
  ASSERT_TRUE(bound);
  auto fromOp = bound->build();
  auto created = TryWaitCapabilityRequestBuilder::create(
                     hivm::TryWaitSyncMode::INTRA_BLOCK)
                     .build();
  ASSERT_EQ(fromOp.values.size(), 1u);
  ASSERT_EQ(created.values.size(), 1u);
  EXPECT_EQ(fromOp.capability, CapabilityKind::TryWait);
  const auto &fromValue = std::get<CapabilityValue>(fromOp.values[0].value);
  const auto &createdValue = std::get<CapabilityValue>(created.values[0].value);
  EXPECT_EQ(fromValue.domain, CapabilityValueDomainKind::TryWaitSyncMode);
  EXPECT_EQ(fromValue.rawValue, createdValue.rawValue);
  EXPECT_EQ(fromValue.rawValue, kIntraBlock);
  EXPECT_TRUE(isCapabilityRequestWellFormed(fromOp.getRequest()));
  EXPECT_EQ(
      queryNativeSupport(NPUArchitectureKind::kDavC310, fromOp.getRequest()),
      NativeSupport::Native);
  EXPECT_EQ(printOp(op), before);
}

TEST_F(SyncBindingTest, MissingOrMistypedSyncModeIsNotARequest) {
  auto op = makeTryWait(builder.getI64IntegerAttr(31),
                        hivm::TryWaitSyncMode::INTRA_BLOCK);
  op->removeAttr("sync_mode");
  EXPECT_FALSE(TryWaitCapabilityRequestBuilder::from(op));

  op->setAttr("sync_mode", builder.getI32IntegerAttr(1));
  EXPECT_FALSE(TryWaitCapabilityRequestBuilder::from(op));
}

TEST_F(SyncBindingTest, FieldFreeBuildersCarryNoValues) {
  auto getBuf = makeGetBuf(builder.getI64IntegerAttr(3));
  const std::string getBufBefore = printOp(getBuf);
  auto getBufBound = GetBufCapabilityRequestBuilder::from(getBuf);
  ASSERT_TRUE(getBufBound);
  auto getBufRequest = getBufBound->build();
  EXPECT_EQ(getBufRequest.capability, CapabilityKind::GetBuf);
  EXPECT_TRUE(getBufRequest.values.empty());
  EXPECT_TRUE(isCapabilityRequestWellFormed(getBufRequest.getRequest()));
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDavC310,
                               getBufRequest.getRequest()),
            NativeSupport::Native);
  EXPECT_EQ(printOp(getBuf), getBufBefore);

  auto rlsBuf = makeRlsBuf(builder.getI64IntegerAttr(3));
  auto rlsBufBound = RlsBufCapabilityRequestBuilder::from(rlsBuf);
  ASSERT_TRUE(rlsBufBound);
  auto rlsBufRequest = rlsBufBound->build();
  EXPECT_EQ(rlsBufRequest.capability, CapabilityKind::RlsBuf);
  EXPECT_TRUE(rlsBufRequest.values.empty());
  EXPECT_TRUE(isCapabilityRequestWellFormed(rlsBufRequest.getRequest()));
  // dav-920r1 inherits the confirmed buffer rows: open, not blocked.
  EXPECT_EQ(queryNativeSupport(NPUArchitectureKind::kDav920R1,
                               rlsBufRequest.getRequest()),
            NativeSupport::Native);

  // create() without an op builds the same request as from(op).
  auto created = GetBufCapabilityRequestBuilder::create().build();
  EXPECT_EQ(created.capability, getBufRequest.capability);
  EXPECT_EQ(created.values.size(), getBufRequest.values.size());
}

} // namespace
