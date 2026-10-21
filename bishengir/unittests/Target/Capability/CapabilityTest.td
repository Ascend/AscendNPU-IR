//===- CapabilityTest.td - synthetic capability catalogue -*- tablegen -*-===//
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

include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

// No production dialect, Op, capability key, or device environment is used.
def TestDialect : Dialect {
  let name = "capability_test";
  let cppNamespace = "::capability_test";
}

def Packed : I32EnumAttrCase<"Packed", 1, "packed">;
def Strided : I32EnumAttrCase<"Strided", 7, "strided">;
def Plain : I32EnumAttrCase<"Plain", 11, "plain">;
def LayoutEnum
    : I32EnumAttr<"Layout", "synthetic slot layout", [Packed, Strided, Plain]> {
  let cppNamespace = "::capability_test";
}
def LayoutAttr : EnumAttr<TestDialect, LayoutEnum, "layout">;
def LayoutDomain : EnumAttrCapabilityDomain<LayoutAttr>;

def Strict : I32EnumAttrCase<"Strict", 3, "strict">;
def Relaxed : I32EnumAttrCase<"Relaxed", 9, "relaxed">;
def ModeEnum
    : I32EnumAttr<"Mode", "synthetic execution mode", [Strict, Relaxed]> {
  let cppNamespace = "::capability_test";
}
def ModeAttr : EnumAttr<TestDialect, ModeEnum, "mode">;
def Mode : EnumAttrCapabilityParameter<ModeAttr>;
def Enabled : BoolAttrCapabilityParameter<>;

// Name derivation follows the TD record independently for C++ and display
// names. Overrides of one spelling must not silently override the other.
def HTTPElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def FP8Type : CapabilitySlotProperty<MLIRTypeDomain>;
def CppProperty : CapabilitySlotProperty<MLIRTypeDomain, "ManualProperty">;
def NamedProperty : CapabilitySlotProperty<MLIRTypeDomain, "", "user.type">;
def BothProperty
    : CapabilitySlotProperty<MLIRTypeDomain, "ChosenProperty", "user.property">;
def HTTPParameter : BoolAttrCapabilityParameter<>;
def CppParameter : BoolAttrCapabilityParameter<"ManualParameter">;
def NamedParameter : BoolAttrCapabilityParameter<"", "user.parameter">;
def HTTPDomain : BoolAttrCapabilityDomain<BoolAttr>;
def CppDomain : BoolAttrCapabilityDomain<BoolAttr> {
  let CppName = "ManualDomain";
}
def NamedDomain : BoolAttrCapabilityDomain<BoolAttr> {
  let Name = "user.domain";
}
def HTTPKey : CapabilityKey<RequestSchema<[]>>;
def CppKey : CapabilityKey<RequestSchema<[]>, "ManualCapability">;
def NamedKey : CapabilityKey<RequestSchema<[]>, "", "user.capability">;

// Every property is explicit; a slot does not implicitly add a type field.
def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def Layout : CapabilitySlotProperty<LayoutDomain>;
def Lhs : CapabilitySlot<[ElementType, Layout]>;
def Rhs : CapabilitySlot<[ElementType]>;
def Result : CapabilitySlot<[ElementType]>;
def LhsType : SlotField<Lhs, ElementType>;
def RhsType : SlotField<Rhs, ElementType>;
def ResultType : SlotField<Result, ElementType>;
def TriadSchema : RequestSchema<[Lhs, Rhs, Result], [Mode, Enabled]>;

def Triad : CapabilityKey<TriadSchema>;
def PartialTriad : CapabilityKey<TriadSchema>;
def EmptyComplete : CapabilityKey<TriadSchema>;
def EmptyPartial : CapabilityKey<TriadSchema>;
def Uncovered : CapabilityKey<TriadSchema>;
def ConcreteTypes : CapabilityKey<TriadSchema>;
def PredicateTypes : CapabilityKey<TriadSchema>;

// These names intentionally collide with a naive record-name-plus-suffix
// emitter and contain punctuation accepted by TD but not C++ identifiers.
def RestrictedLayoutCaseSignatures
    : TypeSignature<[LhsType, RhsType, ResultType], [F16, F16, F32]>;
def "tuple.integer"
    : TypeSignature<[LhsType, RhsType, ResultType], [I8, I8, I32]>;
def MixedTuple : TypeSignature<[LhsType, RhsType, ResultType], [F32, F16, F32]>;

// OR between complete tuples; AND between the non-type field constraints.
// AnyValue permits every valid layout, but never a missing or invalid layout.
// Parameters are fields themselves; only slot properties need SlotField.
def WildcardLayoutCase
    : SupportCase<Triad, [RestrictedLayoutCaseSignatures,
                          !cast<TypeSignature>("tuple.integer"),
                          EnumEquals<Mode, Strict>,
                          BoolValueConstraint<Enabled, 1>,
                          AnyValue<SlotField<Lhs, Layout>>]>;
def RestrictedLayoutCase
    : SupportCase<
          Triad, [MixedTuple, EnumEquals<Mode, Relaxed>,
                  BoolValueConstraint<Enabled, 0>,
                  EnumAnyOf<SlotField<Lhs, Layout>, [Packed, Strided]>]>;
def PartialCase : SupportCase<PartialTriad, [RestrictedLayoutCaseSignatures,
                                             EnumEquals<Mode, Strict>,
                                             BoolValueConstraint<Enabled, 1>,
                                             AnyValue<SlotField<Lhs, Layout>>]>;

// Type constraints reuse ODS predicates directly. No registered type list or
// builderCall is required, including for broad and locally defined predicates.
def ConcreteTypeCase
    : SupportCase<
          ConcreteTypes,
          [TypeSignature<[LhsType, RhsType, ResultType], [SI8, UI8, I8]>,
           TypeSignature<[LhsType, RhsType, ResultType], [BF16, F16, F32]>,
           TypeSignature<[LhsType, RhsType, ResultType], [F8E4M3FN, F8E5M2,
                                                          F32]>,
           AnyValue<Mode>, AnyValue<Enabled>,
           AnyValue<SlotField<Lhs, Layout>>]>;
def F16Alias : TypeAlias<F16>;
// Punctuation is valid for an ODS record name because it is never emitted as a
// C++ identifier. The predicate deliberately has no builderCall.
def "equivalent.f16" : Type<CPred<"$_self.isF16() && true">, "equivalent f16">;
def PredicateTypeCase
    : SupportCase<
          PredicateTypes,
          [TypeSignature<[LhsType, RhsType, ResultType],
                         [AnyFloat, AnyTypeOf<[F16, BF16]>, F32]>,
           TypeSignature<[LhsType, RhsType, ResultType], [F16Alias, F16, F32]>,
           TypeSignature<[LhsType, RhsType, ResultType],
                         [!cast<Type>("equivalent.f16"), F16, F32]>,
           AnyValue<Mode>, AnyValue<Enabled>,
           AnyValue<SlotField<Lhs, Layout>>]>;

// A property-free slot contributes zero fields, even alongside real slots.
def Marker : CapabilitySlot<[]>;
def PropertyFree : CapabilityKey<RequestSchema<[Marker]>>;
def PropertyFreeCase : SupportCase<PropertyFree, []>;
def PropertyFreeCoverage
    : ArchitectureCapabilityCoverage<PropertyFree, [PropertyFreeCase],
                                     CompleteProfile>;

def Global : I32EnumAttrCase<"Global", 2, "global">;
def Local : I32EnumAttrCase<"Local", 5, "local">;
def AddressEnum
    : I32EnumAttr<"AddressSpace", "synthetic address space", [Global, Local]> {
  let cppNamespace = "::capability_test";
}
def AddressAttr : EnumAttr<TestDialect, AddressEnum, "address_space">;
def AddressSpaceDomain : EnumAttrCapabilityDomain<AddressAttr>;
def AddressSpace : CapabilitySlotProperty<AddressSpaceDomain>;
def AddressSlot : CapabilitySlot<[AddressSpace]>;
def AddressOnly : CapabilityKey<RequestSchema<[AddressSlot, Marker]>>;
def AddressCase
    : SupportCase<AddressOnly, [EnumEquals<SlotField<AddressSlot, AddressSpace>,
                                           Global>]>;
def AddressCoverage : ArchitectureCapabilityCoverage<AddressOnly, [AddressCase],
                                                     CompleteProfile>;

def StorageType : CapabilitySlotProperty<MLIRTypeDomain>;
def AlternateLayout : CapabilitySlotProperty<LayoutDomain>;
def SlotBoolDomain : BoolAttrCapabilityDomain<BoolAttr>;
def Active : CapabilitySlotProperty<SlotBoolDomain>;
def RichSlot : CapabilitySlot<[ElementType, StorageType, Layout,
                               AlternateLayout, Active]>;
def RichElementType : SlotField<RichSlot, ElementType>;
def RichStorageType : SlotField<RichSlot, StorageType>;
def Rich : CapabilityKey<RequestSchema<[RichSlot, Marker]>>;
def RichCase
    : SupportCase<
          Rich, [TypeSignature<[RichElementType, RichStorageType], [F16, F32]>,
                 TypeSignature<[RichStorageType, RichElementType], [I32, I8]>,
                 EnumEquals<SlotField<RichSlot, Layout>, Packed>,
                 EnumEquals<SlotField<RichSlot, AlternateLayout>, Strided>,
                 BoolValueConstraint<SlotField<RichSlot, Active>, 1>]>;
def RichCoverage
    : ArchitectureCapabilityCoverage<Rich, [RichCase], CompleteProfile>;

def AnyLhs : CapabilityKey<TriadSchema>;
def AnyLhsCase
    : SupportCase<AnyLhs, [AnyValue<LhsType>,
                           TypeSignature<[RhsType, ResultType], [F16, F32]>,
                           TypeSignature<[ResultType, RhsType], [I32, I8]>,
                           AnyValue<SlotField<Lhs, Layout>>, AnyValue<Mode>,
                           AnyValue<Enabled>]>;
def AnyLhsCoverage
    : ArchitectureCapabilityCoverage<AnyLhs, [AnyLhsCase], CompleteProfile>;

def AllAnyTypes : CapabilityKey<TriadSchema>;
def AllAnyCase : SupportCase<AllAnyTypes, [AnyValue<LhsType>, AnyValue<RhsType>,
                                           AnyValue<ResultType>,
                                           AnyValue<SlotField<Lhs, Layout>>,
                                           AnyValue<Mode>, AnyValue<Enabled>]>;
def AllAnyCoverage : ArchitectureCapabilityCoverage<AllAnyTypes, [AllAnyCase],
                                                    CompleteProfile>;

def AnyTypeAlternatives : CapabilityKey<TriadSchema>;
def AnyTypeCase
    : SupportCase<
          AnyTypeAlternatives,
          [TypeSignature<[LhsType, RhsType, ResultType], [AnyType, F16, F32]>,
           TypeSignature<[LhsType, RhsType, ResultType], [I8, I8, I32]>,
           AnyValue<SlotField<Lhs, Layout>>, AnyValue<Mode>,
           AnyValue<Enabled>]>;
def AnyTypeCoverage
    : ArchitectureCapabilityCoverage<AnyTypeAlternatives, [AnyTypeCase],
                                     CompleteProfile>;

// The shared signature binds named properties, not positions in TriadSchema.
def Reordered
    : CapabilityKey<RequestSchema<[Result, Rhs, Lhs], [Mode, Enabled]>>;
def ReorderedCase : SupportCase<Reordered, [RestrictedLayoutCaseSignatures,
                                            AnyValue<SlotField<Lhs, Layout>>,
                                            AnyValue<Mode>, AnyValue<Enabled>]>;
def ReorderedCoverage
    : ArchitectureCapabilityCoverage<Reordered, [ReorderedCase],
                                     CompleteProfile>;

def ForwardCoverage
    : ArchitectureCapabilityCoverage<
          Triad, [WildcardLayoutCase, RestrictedLayoutCase], CompleteProfile>;
def ReverseCoverage
    : ArchitectureCapabilityCoverage<
          Triad, [RestrictedLayoutCase, WildcardLayoutCase], CompleteProfile>;
def PartialCoverage
    : ArchitectureCapabilityCoverage<PartialTriad, [PartialCase],
                                     PartialProfile>;
def EmptyCompleteCoverage
    : ArchitectureCapabilityCoverage<EmptyComplete, [], CompleteProfile>;
def EmptyPartialCoverage
    : ArchitectureCapabilityCoverage<EmptyPartial, [], PartialProfile>;
def ConcreteTypeCoverage
    : ArchitectureCapabilityCoverage<ConcreteTypes, [ConcreteTypeCase],
                                     CompleteProfile>;
def PredicateTypeCoverage
    : ArchitectureCapabilityCoverage<PredicateTypes, [PredicateTypeCase],
                                     CompleteProfile>;

def ForwardProfile
    : ArchitectureCapabilityProfile<
          Dav_C220, [ForwardCoverage, PartialCoverage, EmptyCompleteCoverage,
                     EmptyPartialCoverage, ConcreteTypeCoverage,
                     PredicateTypeCoverage, PropertyFreeCoverage,
                     AddressCoverage, RichCoverage, AnyLhsCoverage,
                     AllAnyCoverage, AnyTypeCoverage, ReorderedCoverage]>;
def ReverseProfile : ArchitectureCapabilityProfile<
                         Dav_M300, [EmptyPartialCoverage, EmptyCompleteCoverage,
                                    PartialCoverage, ReverseCoverage]>;
// Dav_C310 intentionally has no profile, and Uncovered has no coverage.
