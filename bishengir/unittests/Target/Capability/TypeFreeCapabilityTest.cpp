//===- TypeFreeCapabilityTest.cpp - Type-free runtime tests ---------------===//
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

#include "TypeFreeCapabilityTest.h.inc"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include <variant>

#include "TypeFreeCapabilityTest.cpp.inc"

using namespace bishengir::npu;

namespace {

constexpr auto architecture = static_cast<NPUArchitectureKind>(0);

TEST(TypeFreeCapabilityTest, EmptyRequestIsWellFormedAndUncovered) {
  mlir::MLIRContext context;
  CapabilityRequestStorage storage{CapabilityKind::Empty, {}};
  EXPECT_TRUE(getCapabilityRequestSchema(CapabilityKind::Empty).empty());
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::Unknown);
  storage.values.push_back({CapabilityRequestFieldDesc::slot(
                                0, static_cast<CapabilitySlotPropertyKind>(0)),
                            mlir::Float16Type::get(&context)});
  EXPECT_FALSE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::InvalidRequest);
}

TEST(TypeFreeCapabilityTest, ParameterOnlyValuesKeepValidation) {
  CapabilityRequestStorage storage{
      CapabilityKind::ParameterOnly,
      {{CapabilityRequestFieldDesc::parameter(CapabilityParameterKind::Enabled),
        CapabilityValue{CapabilityValueDomainKind::Bool, 1}}}};
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::Unknown);
  std::get<CapabilityValue>(storage.values[0].value).rawValue = 2;
  EXPECT_FALSE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::InvalidRequest);
}

TEST(TypeFreeCapabilityTest, ParameterOnlySchemaRejectsExtraTypeSlots) {
  mlir::MLIRContext context;
  CapabilityRequestStorage storage{
      CapabilityKind::ParameterOnly,
      {{CapabilityRequestFieldDesc::parameter(CapabilityParameterKind::Enabled),
        CapabilityValue{CapabilityValueDomainKind::Bool, 1}},
       {CapabilityRequestFieldDesc::slot(
            0, static_cast<CapabilitySlotPropertyKind>(0)),
        mlir::Float16Type::get(&context)}}};
  EXPECT_FALSE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::InvalidRequest);
  storage.values.pop_back();
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::Unknown);
}

} // namespace
