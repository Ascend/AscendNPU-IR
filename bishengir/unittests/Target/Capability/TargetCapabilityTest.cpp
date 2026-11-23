//===- TargetCapabilityTest.cpp - Capability tests ---------------===//
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

// Both files are generated from the test-only CapabilityTest.td. Including the
// declarations first also checks they need no production capability header.
#include "CapabilityTest.h.inc"

#include "bishengir/Target/NPU/NPUArchitecture.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <variant>

#include "CapabilityTest.cpp.inc"

using namespace bishengir::npu;

namespace {

using Field = CapabilityRequestFieldDesc;
using Domain = CapabilityValueDomainKind;
using Parameter = CapabilityParameterKind;
using Property = CapabilitySlotPropertyKind;

bool equalField(const Field &lhs, const Field &rhs) {
  return lhs.kind == rhs.kind && lhs.position == rhs.position &&
         lhs.identity == rhs.identity;
}

CapabilityFieldValue &findField(CapabilityRequestStorage &storage,
                                Field field) {
  auto found = llvm::find_if(storage.values, [&](const auto &value) {
    return equalField(value.field, field);
  });
  assert(found != storage.values.end());
  return *found;
}

void setType(CapabilityRequestStorage &storage, unsigned slot,
             mlir::Type type) {
  findField(storage, Field::slot(slot, Property::ElementType)).value = type;
}

void setTypes(CapabilityRequestStorage &storage,
              llvm::ArrayRef<mlir::Type> types) {
  for (auto [index, type] : llvm::enumerate(types))
    setType(storage, index, type);
}

CapabilityValue &domainValue(CapabilityRequestStorage &storage,
                             unsigned index) {
  return std::get<CapabilityValue>(storage.values[index].value);
}

class TargetCapabilityTest : public ::testing::Test {
protected:
  // The context outlives every request and borrowed field value in each test.
  mlir::MLIRContext context;
  mlir::Type f16 = mlir::Float16Type::get(&context);
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::Type i32 = mlir::IntegerType::get(&context, 32);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);
  mlir::Type si8 =
      mlir::IntegerType::get(&context, 8, mlir::IntegerType::Signed);
  mlir::Type ui8 =
      mlir::IntegerType::get(&context, 8, mlir::IntegerType::Unsigned);
  mlir::Type f8e4m3fn = mlir::Float8E4M3FNType::get(&context);
  mlir::Type f8e5m2 = mlir::Float8E5M2Type::get(&context);

  CapabilityRequestStorage
  makeRequest(CapabilityKind kind = CapabilityKind::Triad) {
    return {
        kind,
        {{Field::slot(0, Property::Layout),
          CapabilityValue{Domain::LayoutDomain, 1}},
         {Field::parameter(Parameter::Mode), CapabilityValue{Domain::Mode, 3}},
         {Field::parameter(Parameter::Enabled),
          CapabilityValue{Domain::Bool, 1}},
         {Field::slot(0, Property::ElementType), f16},
         {Field::slot(1, Property::ElementType), f16},
         {Field::slot(2, Property::ElementType), f32}}};
  }
};

NativeSupport
query(const CapabilityRequestStorage &storage,
      NPUArchitectureKind architecture = NPUArchitectureKind::kDavC220) {
  return queryNativeSupport(architecture, storage.getRequest());
}

void expectInvalid(const CapabilityRequestStorage &storage) {
  EXPECT_FALSE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(query(storage), NativeSupport::InvalidRequest);
  // Validation takes precedence over missing profile/coverage information.
  EXPECT_EQ(query(storage, NPUArchitectureKind::kDavC310),
            NativeSupport::InvalidRequest);
  EXPECT_EQ(query(storage, NPUArchitectureKind::kUnknown),
            NativeSupport::InvalidRequest);
}

// Deliberately bypass public request validation to check internal invariants.
// EXPECT_DEATH, unlike EXPECT_DEBUG_DEATH, also exercises release builds.
class TargetCapabilityDeathTest : public TargetCapabilityTest {
protected:
  const TypeSignatureConstraintDesc *getGeneratedTypeSignature() {
    const auto *profile =
        lookupArchitectureCapabilityProfile(NPUArchitectureKind::kDavC220);
    if (!profile)
      return nullptr;
    for (const auto &coverage : profile->capabilities) {
      if (coverage.capability != CapabilityKind::Triad)
        continue;
      for (const auto &supportCase : coverage.cases)
        if (!supportCase.signatures.empty())
          return &supportCase.signatures.front();
    }
    return nullptr;
  }
};

TEST_F(TargetCapabilityDeathTest, RejectsMismatchedCaseCapability) {
  auto storage = makeRequest();
  SupportCaseDesc supportCase{CapabilityKind::PartialTriad, {}, {}};
  EXPECT_DEATH(detail::matchSupportCase(supportCase, storage.getRequest()),
               "support case and request capability mismatch");
}

TEST_F(TargetCapabilityDeathTest, RejectsNullTypeSignatureMatcher) {
  auto storage = makeRequest();
  const TypeSignatureConstraintDesc signatures[] = {{nullptr}};
  SupportCaseDesc supportCase{CapabilityKind::Triad, signatures, {}};
  EXPECT_DEATH(detail::matchSupportCase(supportCase, storage.getRequest()),
               "support case has a null type-signature matcher");
}

TEST_F(TargetCapabilityDeathTest, RejectsValueConstraintForMissingField) {
  auto storage = makeRequest();
  const int64_t allowedValues[] = {2};
  const ValueConstraintDesc constraints[] = {
      {Field::slot(0, Property::AddressSpace), allowedValues}};
  SupportCaseDesc supportCase{CapabilityKind::Triad, {}, constraints};
  EXPECT_DEATH(detail::matchSupportCase(supportCase, storage.getRequest()),
               "value constraint references a missing request field");
}

TEST_F(TargetCapabilityDeathTest, RejectsValueConstraintForTypeField) {
  auto storage = makeRequest();
  const int64_t allowedValues[] = {1};
  const ValueConstraintDesc constraints[] = {
      {Field::slot(0, Property::ElementType), allowedValues}};
  SupportCaseDesc supportCase{CapabilityKind::Triad, {}, constraints};
  EXPECT_DEATH(detail::matchSupportCase(supportCase, storage.getRequest()),
               "value constraint requires an enum or boolean value");
}

TEST_F(TargetCapabilityDeathTest, RejectsEmptyAllowedValues) {
  auto storage = makeRequest();
  const ValueConstraintDesc constraints[] = {
      {Field::parameter(Parameter::Mode), {}}};
  SupportCaseDesc supportCase{CapabilityKind::Triad, {}, constraints};
  EXPECT_DEATH(detail::matchSupportCase(supportCase, storage.getRequest()),
               "value constraint has no allowed values");
}

TEST_F(TargetCapabilityDeathTest, GeneratedTypeMatcherRejectsMissingField) {
  const auto *signature = getGeneratedTypeSignature();
  ASSERT_NE(signature, nullptr);
  ASSERT_NE(signature->matches, nullptr);
  auto storage = makeRequest();
  storage.values.pop_back();
  EXPECT_DEATH(signature->matches(storage.getRequest()),
               "type signature references a missing request field");
}

TEST_F(TargetCapabilityDeathTest, GeneratedTypeMatcherRejectsNullType) {
  const auto *signature = getGeneratedTypeSignature();
  ASSERT_NE(signature, nullptr);
  ASSERT_NE(signature->matches, nullptr);
  auto storage = makeRequest();
  setType(storage, 0, mlir::Type());
  EXPECT_DEATH(signature->matches(storage.getRequest()),
               "type signature requires a non-null MLIR type");
}

TEST_F(TargetCapabilityDeathTest, GeneratedTypeMatcherRejectsWrongCarrier) {
  const auto *signature = getGeneratedTypeSignature();
  ASSERT_NE(signature, nullptr);
  ASSERT_NE(signature->matches, nullptr);
  auto storage = makeRequest();
  findField(storage, Field::slot(0, Property::ElementType)).value =
      CapabilityValue{Domain::Bool, 1};
  EXPECT_DEATH(signature->matches(storage.getRequest()),
               "type signature requires a non-null MLIR type");
}

TEST_F(TargetCapabilityTest, NamedSlotsGenerateTheCanonicalSchema) {
  auto schema = getCapabilityRequestSchema(CapabilityKind::Triad);
  const Field expectedFields[] = {
      Field::slot(0, Property::ElementType),
      Field::slot(1, Property::ElementType),
      Field::slot(2, Property::ElementType),
      Field::slot(0, Property::Layout),
      Field::parameter(Parameter::Mode),
      Field::parameter(Parameter::Enabled),
  };
  ASSERT_EQ(schema.size(), std::size(expectedFields));
  for (const Field &expected : expectedFields) {
    EXPECT_EQ(
        llvm::count_if(schema,
                       [&](const auto &actual) {
                         return actual.field.kind == expected.kind &&
                                actual.field.position == expected.position &&
                                actual.field.identity == expected.identity;
                       }),
        1);
  }
  for (const auto &entry : schema) {
    if (entry.field.kind == CapabilityRequestFieldKind::Slot &&
        entry.field.identity == static_cast<unsigned>(Property::ElementType)) {
      EXPECT_EQ(entry.domain, Domain::MLIRTypeDomain);
    } else {
      EXPECT_NE(entry.domain, static_cast<Domain>(-1));
    }
  }
  EXPECT_TRUE(isCapabilityRequestWellFormed(makeRequest().getRequest()));
}

TEST_F(TargetCapabilityTest, RequestsRetainActualMLIRTypes) {
  auto storage = makeRequest();
  auto request = storage.getRequest();
  ASSERT_EQ(request.values.size(), 6u);
  EXPECT_EQ(std::get<mlir::Type>(request.values[3].value), f16);
  EXPECT_EQ(std::get<mlir::Type>(request.values[4].value), f16);
  EXPECT_EQ(std::get<mlir::Type>(request.values[5].value), f32);
  EXPECT_EQ(request.values.data(), storage.values.data());
}

TEST_F(TargetCapabilityTest, MatchesTypesFromDifferentLiveContexts) {
  mlir::MLIRContext otherContext;
  auto storage = makeRequest();
  setTypes(storage, {mlir::Float16Type::get(&otherContext),
                     mlir::Float16Type::get(&otherContext),
                     mlir::Float32Type::get(&otherContext)});
  EXPECT_NE(std::get<mlir::Type>(storage.values[3].value), f16);
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(query(storage), NativeSupport::Native);
}

TEST_F(TargetCapabilityTest, ReusesConcreteODSTypePredicates) {
  auto storage = makeRequest(CapabilityKind::ConcreteTypes);
  setTypes(storage, {si8, ui8, i8});
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {i8, ui8, i8});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {si8, i8, i8});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {si8, ui8, si8});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {bf16, f16, f32});
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {f16, bf16, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {f8e4m3fn, f8e5m2, f32});
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {f8e5m2, f8e4m3fn, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
}

TEST_F(TargetCapabilityTest, BroadODSPredicatesAndAliasesCanOverlap) {
  auto storage = makeRequest(CapabilityKind::PredicateTypes);
  for (mlir::Type lhs : {f16, bf16, f32, f8e4m3fn, f8e5m2}) {
    setTypes(storage, {lhs, f16, f32});
    EXPECT_EQ(query(storage), NativeSupport::Native);
    setType(storage, 1, bf16);
    EXPECT_EQ(query(storage), NativeSupport::Native);
  }
  // f16 matches AnyFloat, the exact F16 alias, and a distinct equivalent
  // predicate. They are OR alternatives, not a global type-identity conflict.
  setTypes(storage, {f16, f16, f32});
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {i8, f16, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {f16, f32, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {f16, f16, f16});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
}

TEST_F(TargetCapabilityTest, UnlistedMLIRTypesRemainWellFormedRequests) {
  context.allowUnregisteredDialects();
  mlir::Type f64 = mlir::Float64Type::get(&context);
  mlir::Type index = mlir::IndexType::get(&context);
  mlir::Type opaque = mlir::OpaqueType::get(
      mlir::StringAttr::get(&context, "capability_test"), "unlisted");
  // Shaped types are not silently unwrapped to their supported element type.
  mlir::Type tensor = mlir::RankedTensorType::get({2, 2}, f16);
  mlir::Type vector = mlir::VectorType::get({4}, f16);
  mlir::Type memref = mlir::MemRefType::get({4}, f16);
  for (mlir::Type type : {bf16, si8, ui8, f8e4m3fn, f8e5m2, f64, index, opaque,
                          tensor, vector, memref}) {
    for (unsigned slot = 0; slot != 3; ++slot) {
      auto complete = makeRequest();
      setType(complete, slot, type);
      EXPECT_TRUE(isCapabilityRequestWellFormed(complete.getRequest()));
      EXPECT_EQ(query(complete), NativeSupport::Unsupported);
      complete.capability = CapabilityKind::PartialTriad;
      EXPECT_TRUE(isCapabilityRequestWellFormed(complete.getRequest()));
      EXPECT_EQ(query(complete), NativeSupport::Unknown);
    }
  }
}

TEST_F(TargetCapabilityTest, ValuesRetainTheirDomainAndSpelling) {
  EXPECT_EQ(stringifyCapabilityValue({Domain::LayoutDomain, 1}), "packed");
  EXPECT_EQ(stringifyCapabilityValue({Domain::LayoutDomain, 7}), "strided");
  EXPECT_EQ(stringifyCapabilityValue({Domain::LayoutDomain, 11}), "plain");
  EXPECT_EQ(stringifyCapabilityValue({Domain::Mode, 3}), "strict");
  EXPECT_EQ(stringifyCapabilityValue({Domain::Mode, 9}), "relaxed");
  EXPECT_EQ(stringifyCapabilityValue({Domain::Bool, 0}), "false");
  EXPECT_EQ(stringifyCapabilityValue({Domain::Bool, 1}), "true");
  EXPECT_EQ(stringifyCapabilityValue({Domain::LayoutDomain, 3}), std::nullopt);
  EXPECT_EQ(stringifyCapabilityValue({Domain::Bool, 2}), std::nullopt);
  EXPECT_EQ(stringifyCapabilityValue({static_cast<Domain>(-1), 0}),
            std::nullopt);
}

TEST_F(TargetCapabilityTest, NamesDeriveFromRecordsWithIndependentOverrides) {
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::ElementType),
            "element_type");
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::AddressSpace),
            "address_space");
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::HTTPElementType),
            "http_element_type");
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::FP8Type), "fp8_type");
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::ManualProperty),
            "cpp_property");
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::NamedProperty),
            "user.type");
  EXPECT_EQ(stringifyCapabilitySlotPropertyKind(Property::ChosenProperty),
            "user.property");
  EXPECT_EQ(stringifyCapabilityParameterKind(Parameter::HTTPParameter),
            "http_parameter");
  EXPECT_EQ(stringifyCapabilityParameterKind(Parameter::ManualParameter),
            "cpp_parameter");
  EXPECT_EQ(stringifyCapabilityParameterKind(Parameter::NamedParameter),
            "user.parameter");
  EXPECT_EQ(stringifyCapabilityValueDomainKind(Domain::MLIRTypeDomain),
            "mlir_type_domain");
  EXPECT_EQ(stringifyCapabilityValueDomainKind(Domain::HTTPDomain),
            "http_domain");
  EXPECT_EQ(stringifyCapabilityValueDomainKind(Domain::ManualDomain),
            "cpp_domain");
  EXPECT_EQ(stringifyCapabilityValueDomainKind(Domain::NamedDomain),
            "user.domain");
  EXPECT_EQ(stringifyCapabilityKind(CapabilityKind::HTTPKey), "http_key");
  EXPECT_EQ(stringifyCapabilityKind(CapabilityKind::ManualCapability),
            "cpp_key");
  EXPECT_EQ(stringifyCapabilityKind(CapabilityKind::NamedKey),
            "user.capability");
}

TEST_F(TargetCapabilityTest, ExactTupleAlternativesDoNotFormACartesianProduct) {
  auto storage = makeRequest();
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {i8, i8, i32});
  EXPECT_EQ(query(storage), NativeSupport::Native);

  // Each field occurs in a supported tuple, but these combinations do not.
  setTypes(storage, {f16, i8, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {i8, i8, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {f32, f16, f16});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
}

TEST_F(TargetCapabilityTest,
       ConstraintsAreConjoinedAndEnumValuesAreAlternatives) {
  auto storage = makeRequest();
  setTypes(storage, {f32, f16, f32});
  domainValue(storage, 1).rawValue = 9;
  domainValue(storage, 2).rawValue = 0;
  EXPECT_EQ(query(storage), NativeSupport::Native);
  domainValue(storage, 0).rawValue = 7;
  EXPECT_EQ(query(storage), NativeSupport::Native);
  domainValue(storage, 0).rawValue = 11;
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);

  domainValue(storage, 0).rawValue = 1;
  domainValue(storage, 1).rawValue = 3;
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  domainValue(storage, 1).rawValue = 9;
  domainValue(storage, 2).rawValue = 1;
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
}

TEST_F(TargetCapabilityTest, AnyValueStillRequiresAValidPresentValue) {
  auto storage = makeRequest();
  for (int64_t layout : {1, 7, 11}) {
    domainValue(storage, 0).rawValue = layout;
    EXPECT_EQ(query(storage), NativeSupport::Native);
  }
  domainValue(storage, 0).rawValue = 99;
  expectInvalid(storage);
  storage.values.erase(storage.values.begin());
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, CompletenessAndMissingInformationRemainDistinct) {
  auto partial = makeRequest(CapabilityKind::PartialTriad);
  EXPECT_EQ(query(partial), NativeSupport::Native);
  domainValue(partial, 2).rawValue = 0;
  EXPECT_EQ(query(partial), NativeSupport::Unknown);

  auto complete = makeRequest();
  domainValue(complete, 2).rawValue = 0;
  EXPECT_EQ(query(complete), NativeSupport::Unsupported);
  EXPECT_EQ(query(makeRequest(CapabilityKind::EmptyComplete)),
            NativeSupport::Unsupported);
  EXPECT_EQ(query(makeRequest(CapabilityKind::EmptyPartial)),
            NativeSupport::Unknown);
  EXPECT_EQ(query(makeRequest(CapabilityKind::Uncovered)),
            NativeSupport::Unknown);
  EXPECT_EQ(query(makeRequest(), NPUArchitectureKind::kDavC310),
            NativeSupport::Unknown);
  EXPECT_EQ(query(makeRequest(), NPUArchitectureKind::kUnknown),
            NativeSupport::Unknown);
  EXPECT_EQ(lookupArchitectureCapabilityProfile(NPUArchitectureKind::kDavC310),
            nullptr);
  EXPECT_EQ(lookupArchitectureCapabilityProfile(NPUArchitectureKind::kUnknown),
            nullptr);
}

TEST_F(TargetCapabilityTest, RowAndRequestValueOrderDoNotAffectSupport) {
  // Preserve the independent 768-combination oracle over the original four
  // types. Unlisted and broad-predicate types are exercised separately above.
  // C220 and M300 use reversed row/coverage order in the generated catalogue.
  const mlir::Type types[] = {f16, f32, i8, i32};
  auto storage = makeRequest();
  for (mlir::Type lhs : types)
    for (mlir::Type rhs : types)
      for (mlir::Type result : types)
        for (int64_t layout : {1, 7, 11})
          for (int64_t mode : {3, 9})
            for (int64_t enabled : {0, 1}) {
              SCOPED_TRACE(testing::Message()
                           << lhs.getAsOpaquePointer() << ','
                           << rhs.getAsOpaquePointer() << ','
                           << result.getAsOpaquePointer() << ':' << layout
                           << ',' << mode << ',' << enabled);
              setTypes(storage, {lhs, rhs, result});
              domainValue(storage, 0).rawValue = layout;
              domainValue(storage, 1).rawValue = mode;
              domainValue(storage, 2).rawValue = enabled;
              bool wildcardTuple =
                  (lhs == f16 && rhs == f16 && result == f32) ||
                  (lhs == i8 && rhs == i8 && result == i32);
              bool restrictedTuple = lhs == f32 && rhs == f16 && result == f32;
              bool native = (wildcardTuple && mode == 3 && enabled == 1) ||
                            (restrictedTuple && mode == 9 && enabled == 0 &&
                             (layout == 1 || layout == 7));
              NativeSupport expected =
                  native ? NativeSupport::Native : NativeSupport::Unsupported;
              EXPECT_EQ(query(storage), expected);
              EXPECT_EQ(query(storage, NPUArchitectureKind::kDavM300),
                        expected);
              std::reverse(storage.values.begin(), storage.values.end());
              EXPECT_EQ(query(storage), expected);
              EXPECT_EQ(query(storage, NPUArchitectureKind::kDavM300),
                        expected);
              std::reverse(storage.values.begin(), storage.values.end());
            }
}

TEST_F(TargetCapabilityTest, RejectsUnknownCapabilityAndNullType) {
  expectInvalid(CapabilityRequestStorage{});
  auto storage = makeRequest(static_cast<CapabilityKind>(-1));
  expectInvalid(storage);
  storage = makeRequest();
  for (unsigned slot = 0; slot != 3; ++slot) {
    auto invalid = storage;
    setType(invalid, slot, mlir::Type());
    expectInvalid(invalid);
  }
}

TEST_F(TargetCapabilityTest, RejectsMissingAndExtraTypeProperties) {
  auto storage = makeRequest();
  storage.values.pop_back();
  expectInvalid(storage);
  storage = makeRequest();
  storage.values.push_back({Field::slot(3, Property::ElementType), f32});
  expectInvalid(storage);
  storage = makeRequest();
  storage.values.resize(3);
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, RejectsMissingAndDuplicateRequiredValues) {
  auto storage = makeRequest();
  for (unsigned index = 0; index < storage.values.size(); ++index) {
    auto missing = storage;
    missing.values.erase(missing.values.begin() + index);
    expectInvalid(missing);
    auto duplicate = storage;
    duplicate.values.push_back(storage.values[index]);
    expectInvalid(duplicate);
  }
}

TEST_F(TargetCapabilityTest, RejectsExtraOrIncorrectlyIdentifiedFields) {
  const Field invalidFields[] = {
      Field::slot(1, Property::Layout),
      Field::slot(3, Property::Layout),
      Field::slot(0, Property::AddressSpace),
      Field::slot(0, static_cast<Property>(-1)),
      Field::parameter(static_cast<Parameter>(-1)),
      Field::slot(0, Property::ElementType),
      {static_cast<CapabilityRequestFieldKind>(-1), 0, 0},
  };
  for (const Field &field : invalidFields) {
    auto storage = makeRequest();
    storage.values.push_back({field, CapabilityValue{Domain::LayoutDomain, 1}});
    expectInvalid(storage);
  }
  auto storage = makeRequest();
  storage.values[1].field.position = 1;
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, RejectsWrongDomainsAndUnknownRawValues) {
  auto storage = makeRequest();
  for (unsigned index = 0; index != 3; ++index) {
    auto wrongDomain = storage;
    wrongDomain.values[index].value =
        index == 0 ? CapabilityValue{Domain::Mode, 3}
                   : CapabilityValue{Domain::LayoutDomain, 1};
    expectInvalid(wrongDomain);
    auto unknownDomain = storage;
    domainValue(unknownDomain, index).domain = static_cast<Domain>(-1);
    expectInvalid(unknownDomain);
    auto unknownValue = storage;
    domainValue(unknownValue, index).rawValue = 99;
    expectInvalid(unknownValue);
  }
  domainValue(storage, 2).rawValue = 2;
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, RejectsWrongValueCarriers) {
  auto storage = makeRequest();
  for (unsigned index = 0; index != storage.values.size(); ++index) {
    auto wrong = storage;
    if (std::holds_alternative<mlir::Type>(wrong.values[index].value))
      wrong.values[index].value = CapabilityValue{Domain::LayoutDomain, 1};
    else
      wrong.values[index].value = f16;
    expectInvalid(wrong);
  }
}

TEST_F(TargetCapabilityTest, PropertyFreeSlotsContributeNoFields) {
  CapabilityRequestStorage storage{CapabilityKind::PropertyFree, {}};
  EXPECT_TRUE(getCapabilityRequestSchema(CapabilityKind::PropertyFree).empty());
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(query(storage), NativeSupport::Native);
  storage.values.push_back({Field::slot(0, Property::ElementType), f16});
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, AddressOnlySlotDoesNotRequireAType) {
  CapabilityRequestStorage storage{
      CapabilityKind::AddressOnly,
      {{Field::slot(0, Property::AddressSpace),
        CapabilityValue{Domain::AddressSpaceDomain, 2}}}};
  auto schema = getCapabilityRequestSchema(CapabilityKind::AddressOnly);
  ASSERT_EQ(schema.size(), 1u);
  EXPECT_EQ(schema[0].domain, Domain::AddressSpaceDomain);
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(query(storage), NativeSupport::Native);
  domainValue(storage, 0).rawValue = 5;
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  storage.values.push_back({Field::slot(0, Property::ElementType), f16});
  expectInvalid(storage);
  storage.values.pop_back();
  storage.values[0].value = f16;
  expectInvalid(storage);
  storage.values.clear();
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, MultiplePropertiesKeepIndependentIdentities) {
  CapabilityRequestStorage storage{
      CapabilityKind::Rich,
      {{Field::slot(0, Property::ElementType), f16},
       {Field::slot(0, Property::StorageType), f32},
       {Field::slot(0, Property::Layout),
        CapabilityValue{Domain::LayoutDomain, 1}},
       {Field::slot(0, Property::AlternateLayout),
        CapabilityValue{Domain::LayoutDomain, 7}},
       {Field::slot(0, Property::Active),
        CapabilityValue{Domain::SlotBoolDomain, 1}}}};
  auto schema = getCapabilityRequestSchema(CapabilityKind::Rich);
  ASSERT_EQ(schema.size(), 5u);
  EXPECT_EQ(llvm::count_if(schema,
                           [](const auto &entry) {
                             return entry.domain == Domain::MLIRTypeDomain;
                           }),
            2);
  EXPECT_EQ(llvm::count_if(schema,
                           [](const auto &entry) {
                             return entry.domain == Domain::LayoutDomain;
                           }),
            2);
  EXPECT_EQ(query(storage), NativeSupport::Native);
  std::reverse(storage.values.begin(), storage.values.end());
  EXPECT_EQ(query(storage), NativeSupport::Native);
  // The second alternative lists StorageType before ElementType in TD.
  findField(storage, Field::slot(0, Property::ElementType)).value = i8;
  findField(storage, Field::slot(0, Property::StorageType)).value = i32;
  EXPECT_EQ(query(storage), NativeSupport::Native);
  findField(storage, Field::slot(0, Property::StorageType)).value = f32;
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  findField(storage, Field::slot(0, Property::ElementType)).value = f16;
  EXPECT_EQ(query(storage), NativeSupport::Native);
  // Both values use LayoutDomain, but exchanging their property identities
  // changes the meaning rather than preserving an unordered domain pair.
  auto &layout = findField(storage, Field::slot(0, Property::Layout));
  auto &alternate =
      findField(storage, Field::slot(0, Property::AlternateLayout));
  std::swap(layout.value, alternate.value);
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  std::swap(layout.value, alternate.value);
  auto &active = findField(storage, Field::slot(0, Property::Active));
  active.value = CapabilityValue{Domain::SlotBoolDomain, 0};
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  active.value = CapabilityValue{Domain::SlotBoolDomain, 2};
  expectInvalid(storage);
  active.value = CapabilityValue{Domain::SlotBoolDomain, 1};
  findField(storage, Field::slot(0, Property::StorageType)).value =
      CapabilityValue{Domain::LayoutDomain, 1};
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, AnyValueTypesLeaveExplicitTupleAlternatives) {
  auto storage = makeRequest(CapabilityKind::AnyLhs);
  setType(storage, 0, mlir::IndexType::get(&context));
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setType(storage, 1, i8);
  setType(storage, 2, i32);
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setType(storage, 2, f32);
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setType(storage, 0, mlir::Type());
  expectInvalid(storage);
  storage = makeRequest(CapabilityKind::AnyLhs);
  storage.values.erase(storage.values.begin() + 3);
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, AllTypeFieldsCanUseCaseLevelAnyValue) {
  auto storage = makeRequest(CapabilityKind::AllAnyTypes);
  setTypes(storage, {mlir::IndexType::get(&context),
                     mlir::MemRefType::get({4}, f16), bf16});
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setType(storage, 1, mlir::Type());
  expectInvalid(storage);
}

TEST_F(TargetCapabilityTest, AnyTypeIsAnAlternativeLocalWildcard) {
  auto storage = makeRequest(CapabilityKind::AnyTypeAlternatives);
  setType(storage, 0, mlir::IndexType::get(&context));
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {i8, i8, i32});
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {f16, i8, i32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
  setTypes(storage, {i8, f16, i32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
}

TEST_F(TargetCapabilityTest, SharedSignaturesRespectEachSchemasSlotOrder) {
  CapabilityRequestStorage storage{
      CapabilityKind::Reordered,
      {{Field::slot(0, Property::ElementType), f32},
       {Field::slot(1, Property::ElementType), f16},
       {Field::slot(2, Property::ElementType), f16},
       {Field::slot(2, Property::Layout),
        CapabilityValue{Domain::LayoutDomain, 1}},
       {Field::parameter(Parameter::Mode), CapabilityValue{Domain::Mode, 3}},
       {Field::parameter(Parameter::Enabled),
        CapabilityValue{Domain::Bool, 1}}}};
  EXPECT_TRUE(isCapabilityRequestWellFormed(storage.getRequest()));
  EXPECT_EQ(query(storage), NativeSupport::Native);
  std::reverse(storage.values.begin(), storage.values.end());
  EXPECT_EQ(query(storage), NativeSupport::Native);
  setTypes(storage, {f16, f16, f32});
  EXPECT_EQ(query(storage), NativeSupport::Unsupported);
}

} // namespace
