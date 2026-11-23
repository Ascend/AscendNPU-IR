// RUN: bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

// A feature type may have the same name as a generated function parameter.
// Qualify the type in the function body so the parameter cannot hide it.
def NameFeature : EnumArchitectureFeature<"name">;
def Enabled : EnumArchitectureFeatureValue<NameFeature, "Enabled">;

def DavC220FeatureProfile : ArchitectureFeatureProfile<Dav_C220, [Enabled]>;
def DavM300FeatureProfile : ArchitectureFeatureProfile<Dav_M300, [Enabled]>;
def DavC310FeatureProfile : ArchitectureFeatureProfile<Dav_C310, [Enabled]>;
def Dav920R1FeatureProfile : ArchitectureFeatureProfile<Dav_920R1, [Enabled]>;

// CHECK: enum class name {
// CHECK: inline name symbolizename(::llvm::StringRef name) {
// CHECK-NEXT: return ::llvm::StringSwitch<::bishengir::npu::name>(name)
// CHECK-NEXT: .Case("enabled", name::kEnabled)
// CHECK-NEXT: .Default(name::kUnknown);
// CHECK: ::bishengir::npu::name name;
// CHECK: inline name getname(NPUArchitectureKind architecture)
// CHECK: return profile ? profile->name : name::kUnknown;
