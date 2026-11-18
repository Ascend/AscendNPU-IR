// RUN: bishengir-target-spec-tblgen -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include -I %llvm_src_root/include \
// RUN:   -gen-target-capability-defs %s | FileCheck %s

include "bishengir/Target/Capability/CapabilityBase.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}
def ModeA : I32EnumAttrCase<"A", 3, "a">;
def ModeB : I32EnumAttrCase<"B", 9, "b">;
def ModeEnum : I32EnumAttr<"Mode", "test modes", [ModeA, ModeB]>;
def ModeAttr : EnumAttr<TestDialect, ModeEnum, "mode">;
def Mode : EnumAttrCapabilityParameter<ModeAttr>;
def Enabled : BoolAttrCapabilityParameter<>;
def SecondaryEnabled : BoolAttrCapabilityParameter<>;
def AnyEnabled : BoolAttrCapabilityParameter<>;
def Schema : RequestSchema<[], [Mode, Enabled, SecondaryEnabled, AnyEnabled]>;
def Key : CapabilityKey<Schema>;

// Parameters are already request fields. These constraints must retain the
// identity of each parameter, including parameters sharing the bool domain.
def Case : SupportCase<Key, [EnumAnyOf<Mode, [ModeA, ModeB]>,
                             BoolValueConstraint<Enabled, 1>,
                             BoolValueConstraint<SecondaryEnabled, 0>,
                             AnyValue<AnyEnabled>]>;

// AnyValue remains in the schema but emits no runtime value constraint.
// clang-format off
// CHECK: static const int64_t k[[CASE:CapabilityRecord[0-9]+]]Constraint0Values[] = {3, 9};
// CHECK-NEXT: static const int64_t k[[CASE]]Constraint1Values[] = {1};
// CHECK-NEXT: static const int64_t k[[CASE]]Constraint2Values[] = {0};
// CHECK-NEXT: static const ValueConstraintDesc k[[CASE]]Constraints[] =
// CHECK-SAME: CapabilityRequestFieldKind::Parameter, 0, static_cast<unsigned>(CapabilityParameterKind::Mode)}, k[[CASE]]Constraint0Values},
// CHECK-SAME: CapabilityRequestFieldKind::Parameter, 0, static_cast<unsigned>(CapabilityParameterKind::Enabled)}, k[[CASE]]Constraint1Values},
// CHECK-SAME: CapabilityRequestFieldKind::Parameter, 0, static_cast<unsigned>(CapabilityParameterKind::SecondaryEnabled)}, k[[CASE]]Constraint2Values}};
// CHECK: static const CapabilityRequestSchemaFieldDesc {{.*}}[] =
// CHECK-SAME: {{.*}}CapabilityParameterKind::Mode)}, CapabilityValueDomainKind::Mode}
// CHECK-SAME: {{.*}}CapabilityParameterKind::Enabled)}, CapabilityValueDomainKind::Bool}
// CHECK-SAME: {{.*}}CapabilityParameterKind::SecondaryEnabled)}, CapabilityValueDomainKind::Bool}
// CHECK-SAME: {{.*}}CapabilityParameterKind::AnyEnabled)}, CapabilityValueDomainKind::Bool}
// clang-format on
