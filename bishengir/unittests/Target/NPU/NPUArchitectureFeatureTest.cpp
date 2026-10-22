//===- NPUArchitectureFeatureTest.cpp - Architecture feature tests --------===//
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

#include "bishengir/Target/NPU/NPUArchitectureFeatures.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "gtest/gtest.h"

#include <optional>

using namespace mlir;
using namespace mlir::hacc;
using namespace bishengir::npu;

namespace {

// The old helpers enumerated these SKU families directly. Do not derive the
// expected result from getArchitecture() or the generated feature profile:
// either would allow a changed mapping to invalidate both sides of the test.
std::optional<ExecutionModel> getLegacyExecutionModel(StringRef device) {
  if (device.starts_with("Ascend910B") || device.starts_with("Ascend910_93"))
    return ExecutionModel::kMemBased;
  if (device.starts_with("Ascend310B") || device.starts_with("Ascend910_95") ||
      device.starts_with("Ascend950") || device.starts_with("Ascend960"))
    return ExecutionModel::kRegBased;
  return std::nullopt;
}

TEST(NPUArchitectureFeatureTest, EveryDeviceRetainsLegacyExecutionModel) {
  ASSERT_FALSE(targetDevices.empty());
  MLIRContext context;
  context.loadDialect<HACCDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());

  for (TargetDevice device : targetDevices) {
    StringRef spelling = stringifyTargetDeviceEnum(device);
    SCOPED_TRACE(spelling.str());
    auto expected = getLegacyExecutionModel(spelling);
    ASSERT_TRUE(expected.has_value()) << "unreviewed target family";
    EXPECT_EQ(getExecutionModel(getArchitecture(device)), *expected);
    bool memBased = *expected == ExecutionModel::kMemBased;
    bool regBased = *expected == ExecutionModel::kRegBased;
    EXPECT_EQ(hacc::utils::isMemBasedArch(device), memBased);
    EXPECT_EQ(hacc::utils::isRegBasedArch(device), regBased);
    EXPECT_EQ(hacc::utils::isRegBasedArch(spelling), regBased);

    hacc::utils::setTargetDevice(*module, device);
    EXPECT_EQ(hacc::utils::isMemBasedArch(*module), memBased);
    EXPECT_EQ(hacc::utils::isRegBasedArch(*module), regBased);
  }
}

TEST(NPUArchitectureFeatureTest, ProfilesAreKeyedByArchitecture) {
  struct ExpectedProfile {
    NPUArchitectureKind architecture;
    ExecutionModel executionModel;
    CubeVectorMode cubeVectorMode;
    CubeVectorDataExchange cubeVectorDataExchange;
  };
  const ExpectedProfile expectedProfiles[] = {
      {NPUArchitectureKind::kDavC220, ExecutionModel::kMemBased,
       CubeVectorMode::kSeparated, CubeVectorDataExchange::kGlobalMemoryOnly},
      {NPUArchitectureKind::kDavM300, ExecutionModel::kRegBased,
       CubeVectorMode::kCoupled, CubeVectorDataExchange::kOnChipSupported},
      {NPUArchitectureKind::kDavC310, ExecutionModel::kRegBased,
       CubeVectorMode::kSeparated, CubeVectorDataExchange::kOnChipSupported},
  };
  for (const ExpectedProfile &expected : expectedProfiles) {
    SCOPED_TRACE(stringifyNPUArchitecture(expected.architecture).str());
    const auto *profile =
        lookupArchitectureFeatureProfile(expected.architecture);
    ASSERT_NE(profile, nullptr);
    EXPECT_EQ(profile->architecture, expected.architecture);
    EXPECT_EQ(profile->executionModel, expected.executionModel);
    EXPECT_EQ(getExecutionModel(expected.architecture),
              expected.executionModel);
    EXPECT_EQ(profile->cubeVectorMode, expected.cubeVectorMode);
    EXPECT_EQ(getCubeVectorMode(expected.architecture),
              expected.cubeVectorMode);
    EXPECT_EQ(profile->cubeVectorDataExchange, expected.cubeVectorDataExchange);
    EXPECT_EQ(getCubeVectorDataExchange(expected.architecture),
              expected.cubeVectorDataExchange);
  }
}

TEST(NPUArchitectureFeatureTest, EveryDeviceHasReviewedCubeVectorFeatures) {
  for (TargetDevice device : targetDevices) {
    StringRef spelling = stringifyTargetDeviceEnum(device);
    SCOPED_TRACE(spelling.str());
    // Classify the reviewed SKU families independently of the generated
    // profiles, the architecture mapping, and ExecutionModel.
    bool isM300 = spelling.starts_with("Ascend310B");
    bool isC310 = spelling.starts_with("Ascend910_95") ||
                  spelling.starts_with("Ascend950") ||
                  spelling.starts_with("Ascend960");
    bool isC220 = spelling.starts_with("Ascend910B") ||
                  spelling.starts_with("Ascend910_93");
    ASSERT_TRUE(isM300 || isC310 || isC220) << "unreviewed target family";
    NPUArchitectureKind architecture = getArchitecture(device);
    EXPECT_EQ(getCubeVectorMode(architecture),
              isM300 ? CubeVectorMode::kCoupled : CubeVectorMode::kSeparated);
    EXPECT_EQ(getCubeVectorDataExchange(architecture),
              isC220 ? CubeVectorDataExchange::kGlobalMemoryOnly
                     : CubeVectorDataExchange::kOnChipSupported);
  }
}

TEST(NPUArchitectureFeatureTest, CubeVectorFeaturesRoundTrip) {
  EXPECT_EQ(stringifyCubeVectorMode(CubeVectorMode::kCoupled), "coupled");
  EXPECT_EQ(stringifyCubeVectorMode(CubeVectorMode::kSeparated), "separated");
  for (CubeVectorMode value :
       {CubeVectorMode::kCoupled, CubeVectorMode::kSeparated,
        CubeVectorMode::kUnknown})
    EXPECT_EQ(symbolizeCubeVectorMode(stringifyCubeVectorMode(value)), value);
  EXPECT_EQ(symbolizeCubeVectorMode(""), CubeVectorMode::kUnknown);
  EXPECT_EQ(symbolizeCubeVectorMode("invalid"), CubeVectorMode::kUnknown);

  EXPECT_EQ(stringifyCubeVectorDataExchange(
                CubeVectorDataExchange::kGlobalMemoryOnly),
            "global-memory-only");
  EXPECT_EQ(
      stringifyCubeVectorDataExchange(CubeVectorDataExchange::kOnChipSupported),
      "on-chip-supported");
  for (CubeVectorDataExchange value :
       {CubeVectorDataExchange::kGlobalMemoryOnly,
        CubeVectorDataExchange::kOnChipSupported,
        CubeVectorDataExchange::kUnknown})
    EXPECT_EQ(
        symbolizeCubeVectorDataExchange(stringifyCubeVectorDataExchange(value)),
        value);
  EXPECT_EQ(symbolizeCubeVectorDataExchange(""),
            CubeVectorDataExchange::kUnknown);
  EXPECT_EQ(symbolizeCubeVectorDataExchange("invalid"),
            CubeVectorDataExchange::kUnknown);
}

TEST(NPUArchitectureFeatureTest, ExecutionModelRoundTrips) {
  EXPECT_EQ(stringifyExecutionModel(ExecutionModel::kMemBased), "mem-based");
  EXPECT_EQ(stringifyExecutionModel(ExecutionModel::kRegBased), "reg-based");
  EXPECT_EQ(stringifyExecutionModel(ExecutionModel::kUnknown), "unknown");
  for (ExecutionModel value :
       {ExecutionModel::kMemBased, ExecutionModel::kRegBased,
        ExecutionModel::kUnknown})
    EXPECT_EQ(symbolizeExecutionModel(stringifyExecutionModel(value)), value);
  EXPECT_EQ(symbolizeExecutionModel("not-an-execution-model"),
            ExecutionModel::kUnknown);
  EXPECT_EQ(symbolizeExecutionModel(""), ExecutionModel::kUnknown);
}

TEST(NPUArchitectureFeatureTest, UnknownDoesNotSelectFeatures) {
  for (NPUArchitectureKind architecture :
       {NPUArchitectureKind::kUnknown, static_cast<NPUArchitectureKind>(-1)}) {
    EXPECT_EQ(lookupArchitectureFeatureProfile(architecture), nullptr);
    EXPECT_EQ(getExecutionModel(architecture), ExecutionModel::kUnknown);
    EXPECT_EQ(getCubeVectorMode(architecture), CubeVectorMode::kUnknown);
    EXPECT_EQ(getCubeVectorDataExchange(architecture),
              CubeVectorDataExchange::kUnknown);
  }
  EXPECT_FALSE(hacc::utils::isMemBasedArch(TargetDevice::Unknown));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(TargetDevice::Unknown));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(StringRef("unknown-target")));
  EXPECT_FALSE(hacc::utils::isMemBasedArch(static_cast<TargetDevice>(-1)));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(static_cast<TargetDevice>(-1)));
}

TEST(NPUArchitectureFeatureTest, MissingTargetRetainsLegacyModuleDefault) {
  MLIRContext context;
  context.loadDialect<HACCDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());

  EXPECT_FALSE(hacc::utils::getTargetDevice(*module).has_value());
  EXPECT_TRUE(hacc::utils::isMemBasedArch(*module));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(*module));
}

TEST(NPUArchitectureFeatureTest, PresentUnknownTargetDoesNotUseModuleDefault) {
  MLIRContext context;
  context.loadDialect<HACCDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());

  // A present but unknown target is not the same as an absent hacc.target.
  hacc::utils::setTargetDevice(*module, TargetDevice::Unknown);
  EXPECT_EQ(hacc::utils::getTargetDevice(*module), TargetDevice::Unknown);
  EXPECT_FALSE(hacc::utils::isMemBasedArch(*module));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(*module));

  module->getOperation()->setAttr(
      TargetAttr::name,
      TargetAttr::get(&context, builder.getStringAttr("unrecognized-device")));
  EXPECT_EQ(hacc::utils::getTargetDevice(*module), TargetDevice::Unknown);
  EXPECT_FALSE(hacc::utils::isMemBasedArch(*module));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(*module));
}

TEST(NPUArchitectureFeatureTest, Ascend960UsesC310Profile) {
  // Ascend960 (A6) belongs to the dav-c310 lineage: RegBased execution,
  // separated cube/vector cores, on-chip cube/vector exchange.
  EXPECT_EQ(getArchitecture(TargetDevice::Ascend960),
            NPUArchitectureKind::kDavC310);
  const auto *profile =
      lookupArchitectureFeatureProfile(NPUArchitectureKind::kDavC310);
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->architecture, NPUArchitectureKind::kDavC310);
  EXPECT_EQ(profile->executionModel, ExecutionModel::kRegBased);
  EXPECT_EQ(profile->cubeVectorMode, CubeVectorMode::kSeparated);
  EXPECT_EQ(profile->cubeVectorDataExchange,
            CubeVectorDataExchange::kOnChipSupported);

  // The module-level reg-based/mem-based classification for the device.
  MLIRContext context;
  context.loadDialect<HACCDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  hacc::utils::setTargetDevice(*module, TargetDevice::Ascend960);
  EXPECT_TRUE(hacc::utils::isRegBasedArch(*module));
  EXPECT_FALSE(hacc::utils::isMemBasedArch(*module));
}

} // namespace
