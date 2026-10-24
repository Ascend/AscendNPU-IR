//===- NPUArchitectureTest.cpp - Architecture tests ---------------===//
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

#include "bishengir/Target/NPU/NPUArchitecture.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "gtest/gtest.h"

#include <optional>

using namespace mlir;
using namespace mlir::hacc;
using namespace bishengir::npu;

namespace {

struct ExpectedArchitecture {
  NPUArchitectureKind kind;
  StringRef spelling;
};

// These are the pre-migration TargetSpec family assignments, independent of
// the generated architecture mapping. A new family must explicitly extend
// this oracle instead of silently passing a stringify/symbolize round trip.
std::optional<ExpectedArchitecture> getExpectedArchitecture(StringRef device) {
  if (device.starts_with("Ascend910B") || device.starts_with("Ascend910_93"))
    return ExpectedArchitecture{NPUArchitectureKind::kDavC220, "dav-c220"};
  if (device.starts_with("Ascend310B"))
    return ExpectedArchitecture{NPUArchitectureKind::kDavM300, "dav-m300"};
  if (device.starts_with("Ascend910_95") || device.starts_with("Ascend950"))
    return ExpectedArchitecture{NPUArchitectureKind::kDavC310, "dav-c310"};
  if (device.starts_with("Ascend960"))
    return ExpectedArchitecture{NPUArchitectureKind::kDav920R1, "dav-920r1"};
  return std::nullopt;
}

TEST(NPUArchitectureTest, EveryDeviceRetainsItsFamilyArchitecture) {
  ASSERT_FALSE(targetDevices.empty());
  for (TargetDevice device : targetDevices) {
    StringRef name = stringifyTargetDeviceEnum(device);
    SCOPED_TRACE(name.str());
    auto expected = getExpectedArchitecture(name);
    ASSERT_TRUE(expected.has_value()) << "unreviewed target family";
    EXPECT_EQ(getArchitecture(device), expected->kind);
    EXPECT_EQ(stringifyNPUArchitecture(getArchitecture(device)),
              expected->spelling);
    EXPECT_EQ(symbolizeTargetDeviceEnum(name), device);
  }
}

TEST(NPUArchitectureTest, TypedUnknownDoesNotSelectAnArchitecture) {
  EXPECT_EQ(getArchitecture(TargetDevice::Unknown),
            NPUArchitectureKind::kUnknown);
  EXPECT_EQ(symbolizeNPUArchitecture("not-an-architecture"),
            NPUArchitectureKind::kUnknown);
  EXPECT_EQ(stringifyNPUArchitecture(NPUArchitectureKind::kUnknown), "unknown");
}

#if GTEST_HAS_DEATH_TEST
TEST(NPUArchitectureDeathTest, InvalidDeviceValueRemainsFatal) {
  EXPECT_DEATH((void)getArchitecture(static_cast<TargetDevice>(-1)),
               "Unknown TargetDevice value");
}
#endif

TEST(NPUArchitectureTest, DeviceSpecArchStillSerializesAsStringAttr) {
  MLIRContext context;
  OpBuilder builder(&context);
  for (TargetDevice device : targetDevices) {
    StringRef name = stringifyTargetDeviceEnum(device);
    SCOPED_TRACE(name.str());
    auto expected = getExpectedArchitecture(name);
    ASSERT_TRUE(expected.has_value());

    TargetSpec spec{};
    spec.device = device;
    spec.Arch = getArchitecture(device);
    auto attr =
        dyn_cast<StringAttr>(spec.getSpecEntry(DeviceSpec::ARCH, builder));
    ASSERT_TRUE(attr);
    EXPECT_EQ(attr.getValue(), expected->spelling);
  }
}

struct ExpectedDeviceSpec {
  TargetDevice device;
  StringRef name;
  int coreCount;
  int vectorCoreCount;
  int ubSize;
  int l1Size;
  int l0cSize;
  bool hasRegBasedEntries;
  int minimalDCacheSize;
  int maximumDCacheSize;
  int customAIC = -1;
  int customAIV = -1;
};

TEST(NPUArchitectureTest, PublicHACCPassPreservesDeviceSpecSerialization) {
  // Literal pre-migration resources, in bits. Cover every TargetSpec family,
  // both legacy/current C310 names and distinct C310 core-count variants.
  const ExpectedDeviceSpec expectedDevices[] = {
      {TargetDevice::Ascend910B1, "Ascend910B1", 24, 48, 1572864, 4194304,
       1048576, false, 0, 0},
      {TargetDevice::Ascend910_9362, "Ascend910_9362", 20, 40, 1572864, 4194304,
       1048576, false, 0, 0},
      {TargetDevice::Ascend310B1, "Ascend310B1", 1, 1, 2097152, 8388608,
       1048576, true, 0, 0},
      {TargetDevice::Ascend950, "Ascend950", 8, 16, 2031616, 4194304, 2097152,
       true, 262144, 983040, 8, 16},
      {TargetDevice::Ascend910_9579, "Ascend910_9579", 28, 56, 2031616, 4194304,
       2097152, true, 262144, 983040},
      {TargetDevice::Ascend950PR_9589, "Ascend950PR_9589", 32, 64, 2031616,
       4194304, 2097152, true, 262144, 983040},
      {TargetDevice::Ascend950DT_950x, "Ascend950DT_950x", 8, 16, 2031616,
       4194304, 2097152, true, 262144, 983040},
      {TargetDevice::Ascend950DT_95A2, "Ascend950DT_95A2", 36, 72, 2031616,
       4194304, 2097152, true, 262144, 983040},
      {TargetDevice::Ascend960, "Ascend960", 32, 64, 2031616, 4194304, 2097152,
       true, 262144, 983040},
  };

  MLIRContext context;
  context.loadDialect<HACCDialect, DLTIDialect>();
  OpBuilder builder(&context);
  for (const ExpectedDeviceSpec &expected : expectedDevices) {
    SCOPED_TRACE(expected.name.str());
    EXPECT_EQ(stringifyTargetDeviceEnum(expected.device), expected.name);
    OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
    hacc::utils::setTargetDevice(*module, expected.device);
    PassManager pm(&context);
    pm.addPass(createAppendDeviceSpecPass(AppendTargetDeviceSpecOptions{
        TargetDevice::Unknown, expected.customAIC, expected.customAIV, ""}));
    ASSERT_TRUE(succeeded(pm.run(*module)));

    auto spec = hacc::utils::getNPUTargetSpec(*module);
    ASSERT_TRUE(spec.has_value());
    auto concreteSpec = cast<hacc::TargetDeviceSpecAttr>(*spec);
    EXPECT_EQ(concreteSpec.getEntries().size(),
              expected.hasRegBasedEntries ? 14u : 11u);

    auto expectInteger = [&](DeviceSpec field, int value) {
      SCOPED_TRACE(stringifyEnum(field).str());
      DataLayoutEntryInterface entry = spec->getSpecForIdentifierEnum(field);
      ASSERT_TRUE(entry);
      auto attr = dyn_cast<IntegerAttr>(entry.getValue());
      ASSERT_TRUE(attr);
      EXPECT_EQ(attr.getInt(), value);
      EXPECT_TRUE(attr.getType().isInteger(32));
    };
    expectInteger(DeviceSpec::AI_CORE_COUNT, expected.coreCount);
    expectInteger(DeviceSpec::CUBE_CORE_COUNT, expected.coreCount);
    expectInteger(DeviceSpec::VECTOR_CORE_COUNT, expected.vectorCoreCount);
    expectInteger(DeviceSpec::UB_SIZE, expected.ubSize);
    expectInteger(DeviceSpec::L1_SIZE, expected.l1Size);
    expectInteger(DeviceSpec::L0A_SIZE, 524288);
    expectInteger(DeviceSpec::L0B_SIZE, 524288);
    expectInteger(DeviceSpec::L0C_SIZE, expected.l0cSize);
    expectInteger(DeviceSpec::UB_ALIGN_SIZE, 256);
    expectInteger(DeviceSpec::L1_ALIGN_SIZE, 256);
    expectInteger(DeviceSpec::L0C_ALIGN_SIZE, 4096);

    if (!expected.hasRegBasedEntries) {
      EXPECT_FALSE(spec->getSpecForIdentifierEnum(DeviceSpec::ARCH));
      EXPECT_FALSE(
          spec->getSpecForIdentifierEnum(DeviceSpec::MINIMAL_D_CACHE_SIZE));
      EXPECT_FALSE(
          spec->getSpecForIdentifierEnum(DeviceSpec::MAXIMUM_D_CACHE_SIZE));
      continue;
    }
    expectInteger(DeviceSpec::MINIMAL_D_CACHE_SIZE, expected.minimalDCacheSize);
    expectInteger(DeviceSpec::MAXIMUM_D_CACHE_SIZE, expected.maximumDCacheSize);
    DataLayoutEntryInterface arch =
        spec->getSpecForIdentifierEnum(DeviceSpec::ARCH);
    ASSERT_TRUE(arch);
    auto attr = dyn_cast<StringAttr>(arch.getValue());
    ASSERT_TRUE(attr);
    auto expectedArchitecture = getExpectedArchitecture(expected.name);
    ASSERT_TRUE(expectedArchitecture.has_value());
    EXPECT_EQ(attr.getValue(), expectedArchitecture->spelling);
  }
}

TEST(NPUArchitectureTest, Ascend960MapsToDav920R1Architecture) {
  // Ascend960 (A6, dav-920r1) is a new architecture added alongside the
  // legacy C220/M300/C310 families and must round-trip through the stringify
  // / symbolize APIs and the generated architecture mapping.
  EXPECT_EQ(stringifyTargetDeviceEnum(TargetDevice::Ascend960), "Ascend960");
  EXPECT_EQ(symbolizeTargetDeviceEnum("Ascend960"), TargetDevice::Ascend960);
  EXPECT_EQ(getArchitecture(TargetDevice::Ascend960),
            NPUArchitectureKind::kDav920R1);
  EXPECT_EQ(stringifyNPUArchitecture(NPUArchitectureKind::kDav920R1),
            "dav-920r1");
  EXPECT_EQ(symbolizeNPUArchitecture("dav-920r1"),
            NPUArchitectureKind::kDav920R1);

  // The module-level TargetSpec serialization must still emit the spelling.
  MLIRContext context;
  OpBuilder builder(&context);
  TargetSpec spec{};
  spec.device = TargetDevice::Ascend960;
  spec.Arch = getArchitecture(TargetDevice::Ascend960);
  auto attr =
      dyn_cast<StringAttr>(spec.getSpecEntry(DeviceSpec::ARCH, builder));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), "dav-920r1");
}

TEST(NPUArchitectureTest, Ascend960IsFFTSSupported) {
  // isFFTSSupportedArch() was extended to include the Ascend960 family in
  // addition to Ascend910B/910_93/950.
  EXPECT_TRUE(hacc::utils::isAscend960(TargetDevice::Ascend960));
  EXPECT_TRUE(hacc::utils::isAscend960(StringRef("Ascend960")));
  EXPECT_FALSE(hacc::utils::isAscend960(TargetDevice::Ascend910B1));
  EXPECT_FALSE(hacc::utils::isAscend960(TargetDevice::Ascend950PR_9589));
  EXPECT_FALSE(hacc::utils::isAscend960(StringRef("Ascend950PR_9589")));
  EXPECT_FALSE(hacc::utils::isAscend960(StringRef("not-a-device")));

  EXPECT_TRUE(hacc::utils::isFFTSSupportedArch(TargetDevice::Ascend960));
  EXPECT_TRUE(hacc::utils::isFFTSSupportedArch(TargetDevice::Ascend910B1));
  EXPECT_TRUE(hacc::utils::isFFTSSupportedArch(TargetDevice::Ascend950PR_9589));
  EXPECT_FALSE(hacc::utils::isFFTSSupportedArch(TargetDevice::Unknown));
  EXPECT_FALSE(hacc::utils::isFFTSSupportedArch(TargetDevice::Ascend310B1));
}

TEST(NPUArchitectureTest, Ascend960ModuleLevelArchitecture) {
  MLIRContext context;
  context.loadDialect<HACCDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());

  // Without a target attribute the default is 910B (membase), so Ascend960
  // must not be selected.
  EXPECT_FALSE(hacc::utils::isAscend960(*module));
  EXPECT_FALSE(hacc::utils::isRegBasedArch(*module));
  EXPECT_TRUE(hacc::utils::isMemBasedArch(*module));

  hacc::utils::setTargetDevice(*module, TargetDevice::Ascend960);
  EXPECT_TRUE(hacc::utils::isAscend960(*module));
  EXPECT_TRUE(hacc::utils::isRegBasedArch(*module));
  EXPECT_FALSE(hacc::utils::isMemBasedArch(*module));
}

} // namespace
