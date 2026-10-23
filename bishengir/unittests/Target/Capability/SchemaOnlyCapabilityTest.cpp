//===- SchemaOnlyCapabilityTest.cpp - Schema-only tests ----------===//
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

#include "SchemaOnlyCapabilityTest.h.inc"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include <variant>

#include "SchemaOnlyCapabilityTest.cpp.inc"

using namespace bishengir::npu;

namespace {

// A forward-declared architecture enum suffices for a profile-free catalogue.
constexpr auto architecture = static_cast<NPUArchitectureKind>(0);

void expectUncovered(const CapabilityRequestStorage &storage) {
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::Unknown);
}

void expectInvalid(const CapabilityRequestStorage &storage) {
  EXPECT_FALSE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(queryNativeSupport(architecture, storage.getRequest()),
            NativeSupport::InvalidRequest);
}

TEST(SchemaOnlyCapabilityTest, EmptySchemaIsNotAnInvalidCapability) {
  mlir::MLIRContext context;
  CapabilityRequestStorage empty{CapabilityKind::Empty, {}};
  EXPECT_TRUE(getCapabilityRequestSchema(CapabilityKind::Empty).empty());
  expectUncovered(empty);
  expectInvalid(CapabilityRequestStorage{});
  empty.values.push_back({CapabilityRequestFieldDesc::slot(
                              0, CapabilitySlotPropertyKind::ElementType),
                          mlir::Float32Type::get(&context)});
  expectInvalid(empty);
}

TEST(SchemaOnlyCapabilityTest, TypeOnlySchemaNeedsNoValueFields) {
  mlir::MLIRContext context;
  CapabilityRequestStorage storage{
      CapabilityKind::TypeOnly,
      {{CapabilityRequestFieldDesc::slot(
            0, CapabilitySlotPropertyKind::ElementType),
        mlir::Float32Type::get(&context)}}};
  ASSERT_EQ(getCapabilityRequestSchema(CapabilityKind::TypeOnly).size(), 1u);
  expectUncovered(storage);
  storage.values.clear();
  expectInvalid(storage);
}

TEST(SchemaOnlyCapabilityTest, TypeOnlySchemaAcceptsUnlistedTypesButNotNull) {
  mlir::MLIRContext context;
  CapabilityRequestStorage storage{
      CapabilityKind::TypeOnly,
      {{CapabilityRequestFieldDesc::slot(
            0, CapabilitySlotPropertyKind::ElementType),
        mlir::IndexType::get(&context)}}};
  expectUncovered(storage);
  storage.values[0].value = mlir::Float64Type::get(&context);
  expectUncovered(storage);
  storage.values[0].value = mlir::Type();
  expectInvalid(storage);
}

TEST(SchemaOnlyCapabilityTest, ParameterOnlySchemaNeedsNoTypeSlots) {
  CapabilityRequestStorage storage{
      CapabilityKind::ParameterOnly,
      {{CapabilityRequestFieldDesc::parameter(CapabilityParameterKind::Enabled),
        CapabilityValue{CapabilityValueDomainKind::Bool, 1}},
       {CapabilityRequestFieldDesc::parameter(
            CapabilityParameterKind::SecondaryEnabled),
        CapabilityValue{CapabilityValueDomainKind::Bool, 0}},
       {CapabilityRequestFieldDesc::parameter(
            CapabilityParameterKind::AliasEnabled),
        CapabilityValue{CapabilityValueDomainKind::ExplicitBoolDomain, 1}}}};
  ASSERT_EQ(getCapabilityRequestSchema(CapabilityKind::ParameterOnly).size(),
            3u);
  expectUncovered(storage);
  std::get<CapabilityValue>(storage.values[0].value).rawValue = 0;
  expectUncovered(storage);
  std::get<CapabilityValue>(storage.values[0].value).rawValue = 2;
  expectInvalid(storage);
  storage.values.clear();
  expectInvalid(storage);
}

} // namespace
