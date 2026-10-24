//===- CapabilityBase.td - Capability request schema -------*- tablegen -*-===//
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

#ifndef BISHENGIR_TARGET_CAPABILITY_CAPABILITYBASE_TD
#define BISHENGIR_TARGET_CAPABILITY_CAPABILITYBASE_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Capability Conceptual Model
//===----------------------------------------------------------------------===//
//
// A capability query asks whether an operation supports a particular
// combination of SSA value properties and operation attributes.
//
// A Slot names an SSA value's semantic role, such as src, dst, accumulator, or
// scale. Queries use information extracted from that value: element_type,
// address_space, rank, and so on, depending on the property being modeled.
// A Parameter describes an operation attribute, such as a conversion mode.
// For example, a query can organize its fields as follows:
//
//   Slot src        -> element_type, address_space, ...
//   Slot dst        -> element_type, ...
//   Parameter mode  -> the operation's mode attribute
//
//   CapabilityKey -- Schema --> RequestSchema
//        ^                         |-- Slots[]: SSA value roles
//        |                         |      `-- Properties[] -- Domain --> values
//        |                         `-- Parameters[]: operation attributes
//        |
//        `-- Capability -- SupportCase
//                              `-- Constraints[]: rules on request fields
//
// CapabilityKey identifies the semantic operation being queried. Its schema
// defines the fields; a runtime request supplies their actual values; each
// SupportCase describes a supported combination. The schema and cases do not
// depend on a concrete dialect Op or target identity.
//
// Neither a slot's name nor its position implies an ODS operand/result index,
// an input/output partition, or a memory read/write effect.
//
// Target-specific profiles group support cases and state their completeness;
// those records live outside this target-independent schema.

//===----------------------------------------------------------------------===//
// 1. Value Domains
//===----------------------------------------------------------------------===//

// A Domain describes the values a request field can hold. Both Slot properties
// and Parameters refer to a Domain. For example, a dma_mode parameter accepts
// DMA-mode enum cases, not address-space cases, even if their underlying
// integers happen to coincide.
//
// Named records default CppName to their TD record name and Name to its
// snake_case spelling: ElementType becomes ElementType / element_type.
// Empty naming arguments select these defaults; either can be overridden
// independently. Anonymous enum/bool domain wrappers instead reuse their ODS
// carrier's identity so generated names do not depend on anonymous record IDs.
class CapabilityValueDomain<string cppName = "", string name = ""> {
  string CppName = cppName;
  string Name = name;
}

//===----------------------------------------------------------------------===//
// 1.a Enum Domains
//===----------------------------------------------------------------------===//

// The EnumAttr supplies the backing enum and cases.
// Authors do not repeat the enum cases or maintain a second value whitelist.
class EnumAttrCapabilityDomain<Attr enumAttr> : CapabilityValueDomain<> {
  Attr EnumAttr = enumAttr;
}

//===----------------------------------------------------------------------===//
// 1.b Boolean Domains
//===----------------------------------------------------------------------===//

// BoolAttr defines the two values false/true.
class BoolAttrCapabilityDomain<Attr boolAttr> : CapabilityValueDomain<> {
  Attr BoolAttr = boolAttr;
}

//===----------------------------------------------------------------------===//
// 1.c MLIR Type Domains
//===----------------------------------------------------------------------===//

// Any non-null mlir::Type is a valid value. Support cases constrain it with
// ODS Type predicates; there is no type whitelist or generated type enum.
// This domain does not extract or unwrap types. An element_type property may
// hold f32, while an operand_type property may hold tensor<16xf32>.
class MLIRTypeCapabilityDomain : CapabilityValueDomain<>;
def MLIRTypeDomain : MLIRTypeCapabilityDomain;

//===----------------------------------------------------------------------===//
// 2. Parameters
//===----------------------------------------------------------------------===//

// A constraint targets either a Parameter directly or a SlotField identifying
// one property on a Slot. Both use this common field base.
class CapabilityRequestField<string fieldKind> { string FieldKind = fieldKind; }

// A named attribute-derived query parameter. Value properties, such as an
// address space, belong to a Slot rather than to a Parameter.
// Each parameter already identifies one request field, so constraints refer
// to it directly, for example BoolValueConstraint<Enabled, 1>.
class CapabilityParameter<CapabilityValueDomain domain, string cppName = "",
                          string name = "">
    : CapabilityRequestField<"parameter"> {
  string CppName = cppName;
  string Name = name;
  CapabilityValueDomain Domain = domain;
}

//===----------------------------------------------------------------------===//
// 2.a Enum Parameters
//===----------------------------------------------------------------------===//

// Reuse a single ODS EnumAttr as the parameter's complete value domain.
// Given an existing ODS enum attribute ModeAttr, the author only writes:
//
//   def Mode : EnumAttrCapabilityParameter<ModeAttr>;
//
// The inheritance below creates the association:
//   Mode.Domain -> EnumAttrCapabilityDomain<ModeAttr> -> ModeAttr's enum cases.
// No separate Domain declaration is required for this parameter.
class EnumAttrCapabilityParameter<Attr enumAttr>
    : CapabilityParameter<EnumAttrCapabilityDomain<enumAttr>>;

//===----------------------------------------------------------------------===//
// 2.b Boolean Parameters
//===----------------------------------------------------------------------===//

// Each boolean parameter derives its name from its own TD record while
// reusing the common boolean value domain.
class BoolAttrCapabilityParameter<string cppName = "", string name = "">
    : CapabilityParameter<BoolAttrCapabilityDomain<BoolAttr>, cppName, name>;

//===----------------------------------------------------------------------===//
// 3. Slot Properties and Slots
//===----------------------------------------------------------------------===//

// A named piece of information queried about an SSA value. The property
// identifies the information; its Domain defines the values it can hold.
// ElementType and AddressSpace are both properties, with different domains.
// Different properties may share a domain without becoming the same field.
class CapabilitySlotProperty<CapabilityValueDomain domain, string cppName = "",
                             string name = ""> {
  string CppName = cppName;
  string Name = name;
  CapabilityValueDomain Domain = domain;
}

// A semantic SSA value role with an explicit list of queried properties.
// The generator creates only these fields, all required. An empty Properties
// list contributes no fields: there is no automatically added type field.
// The schema declares what is queried, not how to extract it from IR.
// Given MLIRTypeDomain and AddressSpaceDomain, for example:
//
//   def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
//   def AddressSpace : CapabilitySlotProperty<AddressSpaceDomain>;
//   def Src : CapabilitySlot<[ElementType, AddressSpace]>;
//   def Dst : CapabilitySlot<[AddressSpace]>;
//
// Src contributes src.element_type and src.address_space. Dst contributes
// only dst.address_space, not an undeclared dst type.
class CapabilitySlot<list<CapabilitySlotProperty> properties = [],
                     string cppName = "", string name = ""> {
  string CppName = cppName;
  string Name = name;
  list<CapabilitySlotProperty> Properties = properties;
}

//===----------------------------------------------------------------------===//
// 4. Request Schemas
//===----------------------------------------------------------------------===//

// The named value roles and operation attributes that make up a request.
// A slot appears only once even if the SSA value has multiple semantic roles
// (for example, an accumulator that is also a destination).
class RequestSchema<list<CapabilitySlot> slots,
                    list<CapabilityParameter> parameters = []> {
  list<CapabilitySlot> Slots = slots;
  list<CapabilityParameter> Parameters = parameters;
}

//===----------------------------------------------------------------------===//
// 4.a Request Field References
//===----------------------------------------------------------------------===//

// A SlotField tells a support constraint which property of which Slot to
// check. It neither declares a new field nor extracts a value from an Op.
// Given AddressSpaceDomain and its enum case L0C, a minimal example is:
//
//   def AddressSpace : CapabilitySlotProperty<AddressSpaceDomain>;
//   def Src : CapabilitySlot<[AddressSpace]>;
//   def Enabled : BoolAttrCapabilityParameter<>;
//   def Schema : RequestSchema<[Src], [Enabled]>;
//   def SrcInL0C : EnumEquals<SlotField<Src, AddressSpace>, L0C>;
//   def MustBeEnabled : BoolValueConstraint<Enabled, 1>;
//
// AddressSpace names a kind of information: the address space of an SSA value.
// Src names the source role and lists the properties queried about that value.
// Enabled names a boolean attribute-derived parameter. Schema includes both
// Src and Enabled, declaring the fields src.address_space and enabled.
// SlotField<Src, AddressSpace> selects src.address_space; SrcInL0C requires its
// value to be L0C when used in a SupportCase.
// MustBeEnabled directly references Enabled and requires its value to be true.
//
// Src is a schema role, not an actual mlir::Value. A request builder supplies
// the field's concrete value. The same AddressSpace property can be reused on
// Dst; SlotField<Dst, AddressSpace> would refer to a different field.
// Parameters already identify individual fields and need no such reference:
// constraints use Enabled directly, without a field wrapper.

// Refers to a property explicitly listed on the Slot, regardless of its
// Domain. All properties use the same reference and request field list.
class SlotField<CapabilitySlot slot, CapabilitySlotProperty property>
    : CapabilityRequestField<"slot"> {
  CapabilitySlot Slot = slot;
  CapabilitySlotProperty Property = property;
}

//===----------------------------------------------------------------------===//
// 5. Capability Keys
//===----------------------------------------------------------------------===//

// Stable semantic operation identity, not a device name or intrinsic spelling.
class CapabilityKey<RequestSchema schema, string cppName = "",
                    string name = ""> {
  string CppName = cppName;
  string Name = name;
  RequestSchema Schema = schema;
}

//===----------------------------------------------------------------------===//
// 6. Support Constraints
//===----------------------------------------------------------------------===//

// Constraints select supported values from the fields declared by a schema.
// TypeSignature matches combinations of MLIRType-domain fields. Enum and
// boolean constraints restrict individual fields; AnyValue explicitly leaves
// a field unrestricted. SupportCase below combines these constraints.
class CapabilityConstraint;

//===----------------------------------------------------------------------===//
// 6.a Type Signature Constraints
//===----------------------------------------------------------------------===//

// One permitted combination of explicitly named type-backed fields. For
// example, with ElementType declared on both Src and Dst:
//
//   TypeSignature<[SlotField<Src, ElementType>, SlotField<Dst, ElementType>],
//                 [F32, F16]>
//
// This requires src.element_type=f32 AND dst.element_type=f16. Each ODS
// predicate applies to its corresponding field, not an implicit Slot index.
// Reordering fields together with their predicates does not change the rule.
// Multiple signatures are OR alternatives over complete combinations:
// [F32, F16] and [I32, I8] must not admit [F32, I8]. Every alternative covers
// exactly the declared type-backed fields not already covered by AnyValue.
// Use AnyType for a wildcard within one alternative, rather than omitting a
// field. No signature is required when there are no remaining type fields.
//
// The generator applies ODS predicates to the supplied mlir::Type values.
// AnyFloat and other broad predicates are allowed when that set is intended;
// predicates must depend only on $_self, not an existing Op. A type with no
// matching rule is a valid query, not an invalid request.
class TypeSignature<list<SlotField> fields, list<Type> types>
    : CapabilityConstraint {
  list<SlotField> Fields = fields;
  list<Type> Types = types;
}

//===----------------------------------------------------------------------===//
// 6.b Enum Constraints
//===----------------------------------------------------------------------===//

// Restrict one enum-valued field to a non-empty set of allowed cases. The field
// may be a Slot property or a Parameter. The generator checks that every case
// belongs to that field's ODS enum domain.
//
// EnumEquals and EnumAnyOf below are authoring shorthands for this same
// constraint; neither introduces a separate matching mechanism. For an enum
// parameter Mode with cases ModeA and ModeB:
//
//   EnumEquals<Mode, ModeA>          // mode == ModeA
//   EnumAnyOf<Mode, [ModeA, ModeB]>  // mode in {ModeA, ModeB}
//
// AnyOf means OR within this one field, not OR between SupportCases.
// Constraints on different fields in the same SupportCase are combined with
// AND. Use AnyValue instead when the field should not restrict support.
class EnumValueConstraint<CapabilityRequestField target,
                          list<EnumAttrCaseInfo> values>
    : CapabilityConstraint {
  CapabilityRequestField Field = target;
  list<EnumAttrCaseInfo> Values = values;
}

// Singleton shorthand: EnumEquals<Field, A> accepts only A.
class EnumEquals<CapabilityRequestField target, EnumAttrCaseInfo value>
    : EnumValueConstraint<target, [value]>;

// Set shorthand: EnumAnyOf<Field, [A, B]> accepts either A or B.
class EnumAnyOf<CapabilityRequestField target, list<EnumAttrCaseInfo> values>
    : EnumValueConstraint<target, values>;

//===----------------------------------------------------------------------===//
// 6.c Boolean Constraints
//===----------------------------------------------------------------------===//

// Restrict one BoolAttr-backed Slot property or Parameter to false (0) or
// true (1), for example BoolValueConstraint<Enabled, 1>.
class BoolValueConstraint<CapabilityRequestField target, bit value>
    : CapabilityConstraint {
  CapabilityRequestField Field = target;
  bit Value = value;
}

//===----------------------------------------------------------------------===//
// 6.d Unrestricted Fields: AnyValue
//===----------------------------------------------------------------------===//

// The field was considered and does not restrict this SupportCase. This also
// applies to type-backed properties: the field must still contain a non-null
// mlir::Type. AnyValue removes a matching predicate, not a required field.
class AnyValue<CapabilityRequestField target> : CapabilityConstraint {
  CapabilityRequestField Field = target;
}

//===----------------------------------------------------------------------===//
// 7. Support Cases
//===----------------------------------------------------------------------===//

// Match any complete type-signature alternative AND all value constraints.
// Every declared field must be covered, including fields intentionally ignored
// with AnyValue. Omitting a field is a generation error, so adding a property
// cannot silently widen support rules. A field-free schema needs no
// constraints.
class SupportCase<CapabilityKey capability,
                  list<CapabilityConstraint> constraints> {
  CapabilityKey Capability = capability;
  list<CapabilityConstraint> Constraints = constraints;
}

//===----------------------------------------------------------------------===//
// 8. Profile Completeness
//===----------------------------------------------------------------------===//

// Completeness describes the Native support database for one capability.
// Complete miss means non-Native, not that no legalization path can exist.
// Partial miss leaves the hardware support answer unknown.
class ProfileCompleteness<string cppName = "", string name = ""> {
  string CppName = cppName;
  string Name = name;
}

def CompleteProfile : ProfileCompleteness<"Complete", "complete">;
def PartialProfile : ProfileCompleteness<"Partial", "partial">;

#endif // BISHENGIR_TARGET_CAPABILITY_CAPABILITYBASE_TD
