// clang-format off
// RUN: not bishengir-target-spec-tblgen -DDUP_SLOT -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SLOT
// RUN: not bishengir-target-spec-tblgen -DDUP_SLOT_NAME -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SLOT
// RUN: not bishengir-target-spec-tblgen -DDUP_PROPERTY -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PROPERTY
// RUN: not bishengir-target-spec-tblgen -DDUP_PROPERTY_NAME -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PROPERTY
// RUN: not bishengir-target-spec-tblgen -DDUP_PARAMETER -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PARAMETER
// RUN: not bishengir-target-spec-tblgen -DSIGNATURE_ARITY -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=ARITY
// RUN: not bishengir-target-spec-tblgen -DMISSING_PARAMETER -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not bishengir-target-spec-tblgen -DMISSING_PROPERTY -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not bishengir-target-spec-tblgen -DUNKNOWN_SLOT_FIELD -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SLOTFIELD
// RUN: not bishengir-target-spec-tblgen -DUNKNOWN_PARAMETER_FIELD -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=FIELD
// RUN: not bishengir-target-spec-tblgen -DTYPE_FIELD_CONSTRAINT -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=ENUMBOOL
// RUN: not bishengir-target-spec-tblgen -DENUM_WRONG_DOMAIN -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=ENUMDOMAIN
// RUN: not bishengir-target-spec-tblgen -DBOOL_WRONG_DOMAIN -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=BOOLDOMAIN
// RUN: not bishengir-target-spec-tblgen -DENUM_FOR_BOOL -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=ENUMBOOL
// RUN: not bishengir-target-spec-tblgen -DEMPTY_ENUM -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=EMPTYENUM
// RUN: not bishengir-target-spec-tblgen -DDUP_CONSTRAINT -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=CONSTRAINT
// RUN: not bishengir-target-spec-tblgen -DDUP_SIGNATURE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SIGNATURE
// RUN: not bishengir-target-spec-tblgen -DDUP_ALIAS_SIGNATURE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SIGNATURE
// RUN: not bishengir-target-spec-tblgen -DDUP_PERMUTED_SIGNATURE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SIGNATURE
// RUN: not bishengir-target-spec-tblgen -DNO_SIGNATURE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=NOSIGNATURE
// RUN: not bishengir-target-spec-tblgen -DUNSUPPORTED_CONSTRAINT -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=UNKNOWNCONSTRAINT
// RUN: not bishengir-target-spec-tblgen -DINVALID_CPP_NAME -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=IDENTIFIER
// RUN: not bishengir-target-spec-tblgen -DINVALID_DEFAULT_CPP_NAME -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=IDENTIFIER
// RUN: not bishengir-target-spec-tblgen -DINVALID_TYPE_PREDICATE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PREDICATE
// RUN: not bishengir-target-spec-tblgen -DUNSUPPORTED_DOMAIN -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=UNSUPPORTEDDOMAIN
// RUN: not bishengir-target-spec-tblgen -DAMBIGUOUS_DOMAIN -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=AMBIGUOUSDOMAIN
// RUN: not bishengir-target-spec-tblgen -DTYPE_PARAMETER -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=TYPEPARAMETER
// RUN: not bishengir-target-spec-tblgen -DSIGNATURE_VALUE_FIELD -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=TYPEFIELD
// RUN: not bishengir-target-spec-tblgen -DDUP_SIGNATURE_FIELD -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=DUPFIELD
// RUN: not bishengir-target-spec-tblgen -DUNDECLARED_SIGNATURE_FIELD -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SIGNATUREFIELD
// RUN: not bishengir-target-spec-tblgen -DINCOMPLETE_SIGNATURE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: not bishengir-target-spec-tblgen -DINCONSISTENT_ALTERNATIVES -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: not bishengir-target-spec-tblgen -DOVERLAPPING_TYPE_COVERAGE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not bishengir-target-spec-tblgen -DDERIVED_KEY_NAME_COLLISION -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=KEYNAME
// RUN: not bishengir-target-spec-tblgen -DDERIVED_DOMAIN_NAME_COLLISION -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=DOMAINNAME
// RUN: not bishengir-target-spec-tblgen -DMISSING_PROPERTY_DOMAIN -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PROPERTYDOMAIN
// RUN: not bishengir-target-spec-tblgen -DUNKNOWN_REQUEST_FIELD -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=FIELDKIND
// RUN: not bishengir-target-spec-tblgen -DMUTATED_PARAMETER_KIND -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=FIELDKIND
// RUN: not bishengir-target-spec-tblgen -DMUTATED_SLOT_KIND -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=FIELDKIND
// clang-format on

include "bishengir/Target/Capability/CapabilityBase.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}
def ModeA : I32EnumAttrCase<"A", 0, "a">;
def ModeB : I32EnumAttrCase<"B", 1, "b">;
def ModeEnum : I32EnumAttr<"Mode", "test modes", [ModeA, ModeB]>;
def ModeAttr : EnumAttr<TestDialect, ModeEnum, "mode">;
def Mode : EnumAttrCapabilityParameter<ModeAttr>;
def Enabled : BoolAttrCapabilityParameter<>;
def Local : I32EnumAttrCase<"Local", 0, "local">;
def SpaceEnum : I32EnumAttr<"Space", "test spaces", [Local]>;
def SpaceAttr : EnumAttr<TestDialect, SpaceEnum, "space">;
def SpaceDomain : EnumAttrCapabilityDomain<SpaceAttr>;
def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def AddressSpace : CapabilitySlotProperty<SpaceDomain>;
def Lhs : CapabilitySlot<[ElementType, AddressSpace]>;
def Rhs : CapabilitySlot<[ElementType]>;
def Accumulator : CapabilitySlot<[ElementType]>;
def LhsType : SlotField<Lhs, ElementType>;
def RhsType : SlotField<Rhs, ElementType>;
def AccType : SlotField<Accumulator, ElementType>;
def Schema : RequestSchema<[Lhs, Rhs, Accumulator], [Mode, Enabled]>;
def Key : CapabilityKey<Schema>;
def Signature : TypeSignature<[LhsType, RhsType, AccType], [F16, F16, F32]>;
def AnySpace : AnyValue<SlotField<Lhs, AddressSpace>>;
def AnyMode : AnyValue<Mode>;
def AnyEnabled : AnyValue<Enabled>;

#ifdef DUP_SLOT
def BadSchema : RequestSchema<[Lhs, Lhs]>;
def BadKey : CapabilityKey<BadSchema>;
#endif
#ifdef DUP_SLOT_NAME
def OtherLhs : CapabilitySlot<[], "", "lhs">;
def BadSchema : RequestSchema<[Lhs, OtherLhs]>;
def BadKey : CapabilityKey<BadSchema>;
#endif
#ifdef DUP_PROPERTY
def BadSlot : CapabilitySlot<[AddressSpace, AddressSpace]>;
#endif
#ifdef DUP_PROPERTY_NAME
// Distinct valid C++ names derive the same snake_case field name.
def HTTPValue : CapabilitySlotProperty<SpaceDomain>;
def HttpValue : CapabilitySlotProperty<SpaceDomain>;
def BadSlot : CapabilitySlot<[HTTPValue, HttpValue]>;
#endif
#ifdef DUP_PARAMETER
def BadSchema : RequestSchema<[], [Mode, Mode]>;
def BadKey : CapabilityKey<BadSchema>;
#endif
#ifdef SIGNATURE_ARITY
def Bad : SupportCase<Key, [TypeSignature<[LhsType, RhsType, AccType], [F16]>,
                            AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef MISSING_PARAMETER
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode]>;
#endif
#ifdef MISSING_PROPERTY
def Bad : SupportCase<Key, [Signature, AnyMode, AnyEnabled]>;
#endif
#ifdef UNKNOWN_SLOT_FIELD
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode, AnyEnabled,
                            AnyValue<SlotField<Rhs, AddressSpace>>]>;
#endif
#ifdef UNKNOWN_PARAMETER_FIELD
def OtherParameter : BoolAttrCapabilityParameter<>;
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode, AnyEnabled,
                            AnyValue<OtherParameter>]>;
#endif
#ifdef TYPE_FIELD_CONSTRAINT
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode, AnyEnabled,
                            EnumEquals<LhsType, Local>]>;
#endif
#ifdef ENUM_WRONG_DOMAIN
def Bad : SupportCase<Key, [Signature, AnySpace, AnyEnabled,
                            EnumEquals<Mode, Local>]>;
#endif
#ifdef BOOL_WRONG_DOMAIN
def Bad : SupportCase<Key, [Signature, AnySpace, AnyEnabled,
                            BoolValueConstraint<Mode, 1>]>;
#endif
#ifdef ENUM_FOR_BOOL
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode,
                            EnumEquals<Enabled, ModeA>]>;
#endif
#ifdef EMPTY_ENUM
def Bad
    : SupportCase<Key, [Signature, AnySpace, AnyEnabled, EnumAnyOf<Mode, []>]>;
#endif
#ifdef DUP_CONSTRAINT
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode, AnyEnabled,
                            EnumEquals<Mode, ModeA>]>;
#endif
#ifdef DUP_SIGNATURE
def Bad : SupportCase<
              Key, [Signature,
                    TypeSignature<[LhsType, RhsType, AccType], [F16, F16, F32]>,
                    AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef DUP_ALIAS_SIGNATURE
def F16Alias : TypeAlias<F16>;
def Bad
    : SupportCase<Key, [Signature,
                        TypeSignature<[LhsType, RhsType, AccType], [F16Alias,
                                                                    F16, F32]>,
                        AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef DUP_PERMUTED_SIGNATURE
def Bad : SupportCase<
              Key, [Signature,
                    TypeSignature<[AccType, RhsType, LhsType], [F32, F16, F16]>,
                    AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef NO_SIGNATURE
def Bad : SupportCase<Key, [AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef UNSUPPORTED_CONSTRAINT
def UnknownConstraint : CapabilityConstraint;
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode, AnyEnabled,
                            UnknownConstraint]>;
#endif
#ifdef INVALID_CPP_NAME
def Bad : CapabilityKey<Schema, "not-valid">;
#endif
#ifdef INVALID_DEFAULT_CPP_NAME
def "not-valid" : CapabilityKey<Schema>;
#endif
#ifdef INVALID_TYPE_PREDICATE
def OpDependentType : Type<CPred<"$_op->getNumResults() == 1">, "bad type">;
def BadTypeSignature : TypeSignature<[LhsType], [OpDependentType]>;
#endif
#ifdef UNSUPPORTED_DOMAIN
def UnsupportedDomain : CapabilityValueDomain<>;
def BadProperty : CapabilitySlotProperty<UnsupportedDomain>;
#endif
#ifdef AMBIGUOUS_DOMAIN
def AmbiguousDomain : MLIRTypeCapabilityDomain,
                      BoolAttrCapabilityDomain<BoolAttr>;
def BadProperty : CapabilitySlotProperty<AmbiguousDomain>;
#endif
#ifdef TYPE_PARAMETER
def BadParameter : CapabilityParameter<MLIRTypeDomain>;
#endif
#ifdef SIGNATURE_VALUE_FIELD
def Bad
    : SupportCase<Key, [TypeSignature<[SlotField<Lhs, AddressSpace>], [F16]>,
                        AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef DUP_SIGNATURE_FIELD
def Bad : SupportCase<
              Key, [TypeSignature<[LhsType, LhsType, AccType], [F16, F16, F32]>,
                    AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef UNDECLARED_SIGNATURE_FIELD
def OtherSlot : CapabilitySlot<[ElementType]>;
def Bad : SupportCase<
              Key, [TypeSignature<[SlotField<OtherSlot, ElementType>], [F16]>,
                    AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef INCOMPLETE_SIGNATURE
def Bad : SupportCase<Key, [TypeSignature<[LhsType, RhsType], [F16, F16]>,
                            AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef INCONSISTENT_ALTERNATIVES
def Bad
    : SupportCase<Key, [Signature, TypeSignature<[LhsType, RhsType], [I8, I8]>,
                        AnySpace, AnyMode, AnyEnabled]>;
#endif
#ifdef OVERLAPPING_TYPE_COVERAGE
def Bad : SupportCase<Key, [Signature, AnyValue<LhsType>, AnySpace, AnyMode,
                            AnyEnabled]>;
#endif
#ifdef DERIVED_KEY_NAME_COLLISION
def HTTPKey : CapabilityKey<RequestSchema<[]>>;
def HttpKey : CapabilityKey<RequestSchema<[]>>;
#endif
#ifdef DERIVED_DOMAIN_NAME_COLLISION
def HTTPDomain : BoolAttrCapabilityDomain<BoolAttr>;
def HttpDomain : BoolAttrCapabilityDomain<BoolAttr>;
#endif
#ifdef MISSING_PROPERTY_DOMAIN
// An unused property must not evade its mandatory domain contract.
def MissingDomain : CapabilitySlotProperty<?>;
#endif
#ifdef UNKNOWN_REQUEST_FIELD
// A discriminator alone cannot make an arbitrary record a parameter.
def UnknownField : CapabilityRequestField<"parameter">;
def Bad : SupportCase<Key, [Signature, AnySpace, AnyMode, AnyEnabled,
                            AnyValue<UnknownField>]>;
#endif
#ifdef MUTATED_PARAMETER_KIND
// Parameters cannot masquerade as SlotFields, even if no rule references them.
def BadParameter : BoolAttrCapabilityParameter<> { let FieldKind = "slot"; }
#endif
#ifdef MUTATED_SLOT_KIND
def BadField : SlotField<Lhs, AddressSpace> { let FieldKind = "parameter"; }
def Bad
    : SupportCase<Key, [Signature, AnyMode, AnyEnabled, AnyValue<BadField>]>;
#endif

// clang-format off
// SLOT: error: duplicate semantic slot
// PROPERTY: error: duplicate property
// PARAMETER: error: duplicate parameter
// ARITY: error: TypeSignature Fields and Types must have equal lengths
// MISSING: error: non-type request field
// SLOTFIELD: error: property 'AddressSpace' is not declared on slot 'Rhs'
// FIELD: error: constraint field
// TYPEFIELD: error: TypeSignature must target an MLIRType-domain slot field
// ENUMDOMAIN: error: ODS enum case
// BOOLDOMAIN: error: BoolValueConstraint requires
// ENUMBOOL: error: EnumValueConstraint requires
// EMPTYENUM: error: EnumValueConstraint must contain
// CONSTRAINT: error: duplicate constraint
// SIGNATURE: error: duplicate type signature in SupportCase
// NOSIGNATURE: error: type request field {{.*}} requires a TypeSignature or case-level AnyValue
// UNKNOWNCONSTRAINT: error: unsupported CapabilityConstraint
// IDENTIFIER: error: invalid CapabilityKey CppName: not-valid
// PREDICATE: error: {{.*}}predicate
// UNSUPPORTEDDOMAIN: error: capability value domain must have exactly one MLIRType, enum-attribute, or BoolAttr carrier
// Multiple carrier bases are rejected by TableGen before generator validation.
// AMBIGUOUSDOMAIN: error: Already subclass of 'CapabilityValueDomain'!
// TYPEPARAMETER: error: capability parameters require an enum-attribute or BoolAttr domain; MLIRType domains are only supported on slot properties
// DUPFIELD: error: duplicate field in TypeSignature
// SIGNATUREFIELD: error: signature field {{.*}} is not declared by capability
// INCOMPLETE: error: type request field {{.*}} must occur in every TypeSignature alternative or have a case-level AnyValue
// OVERLAP: error: signature field {{.*}} also has a case-level constraint
// KEYNAME: error: duplicate CapabilityKey Name
// DOMAINNAME: error: duplicate CapabilityValueDomain Name: http_domain
// PROPERTYDOMAIN: error: CapabilitySlotProperty requires a Domain
// FIELDKIND: error: request field must be a SlotField or CapabilityParameter with its declared field kind
// clang-format on
