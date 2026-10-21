//===- SchemaOnlyCapabilityTest.td - Schema-only tests -----*- tablegen -*-===//
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

include "bishengir/Target/Capability/CapabilityBase.td"

// No architecture class, profile, enum attribute, or production dialect.
def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def Item : CapabilitySlot<[ElementType]>;
def Enabled : BoolAttrCapabilityParameter<>;
def SecondaryEnabled : BoolAttrCapabilityParameter<>;
// Anonymous boolean wrappers deduplicate by the ODS BoolAttr carrier. A named
// domain retains its own identity even when it uses that same carrier.
def ExplicitBoolDomain : BoolAttrCapabilityDomain<BoolAttr>;
def AliasEnabled : CapabilityParameter<ExplicitBoolDomain>;
def Empty : CapabilityKey<RequestSchema<[]>>;
def TypeOnly : CapabilityKey<RequestSchema<[Item]>>;
def ParameterOnly
    : CapabilityKey<
          RequestSchema<[], [Enabled, SecondaryEnabled, AliasEnabled]>>;

// These records exercise empty tuples and empty non-type constraint lists,
// although no architecture claims coverage for any of these capabilities.
def EmptyCase : SupportCase<Empty, [TypeSignature<[], []>]>;
def TypeCase
    : SupportCase<
          TypeOnly, [TypeSignature<[SlotField<Item, ElementType>], [F32]>]>;
def ParameterCase
    : SupportCase<ParameterOnly, [TypeSignature<[], []>,
                                  BoolValueConstraint<Enabled, 1>,
                                  BoolValueConstraint<SecondaryEnabled, 0>,
                                  AnyValue<AliasEnabled>]>;
