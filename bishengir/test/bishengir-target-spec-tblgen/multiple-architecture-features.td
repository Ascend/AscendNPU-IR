// RUN: bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def ExecutionModelFeature : EnumArchitectureFeature<"ExecutionModel">;
def MemBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "MemBased">;
def RegBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "RegBased">;

// Adding a second feature needs no C++ field or accessor implementation.
def SynchronizationModelFeature
    : EnumArchitectureFeature<"SynchronizationModel">;
def ImplicitSynchronization
    : EnumArchitectureFeatureValue<SynchronizationModelFeature, "Implicit">;
def ExplicitSynchronization
    : EnumArchitectureFeatureValue<SynchronizationModelFeature, "Explicit">;

def DavC220FeatureProfile
    : ArchitectureFeatureProfile<Dav_C220, [MemBased, ImplicitSynchronization]>;
def DavM300FeatureProfile
    : ArchitectureFeatureProfile<Dav_M300, [RegBased, ExplicitSynchronization]>;
def DavC310FeatureProfile
    : ArchitectureFeatureProfile<Dav_C310, [RegBased, ExplicitSynchronization]>;

// CHECK: enum class ExecutionModel {
// CHECK: enum class SynchronizationModel {
// CHECK-NEXT: kExplicit,
// CHECK-NEXT: kImplicit,
// CHECK-NEXT: kUnknown,
// CHECK: stringifySynchronizationModel(SynchronizationModel value)
// CHECK: case SynchronizationModel::kExplicit: return "explicit";
// CHECK: case SynchronizationModel::kImplicit: return "implicit";
// CHECK: symbolizeSynchronizationModel(::llvm::StringRef name)
// CHECK: .Case("explicit", SynchronizationModel::kExplicit)
// CHECK: .Case("implicit", SynchronizationModel::kImplicit)
// CHECK: .Default(SynchronizationModel::kUnknown);
// CHECK: struct ArchitectureFeatureProfile {
// CHECK-NEXT: NPUArchitectureKind architecture;
// CHECK-NEXT: ::bishengir::npu::ExecutionModel executionModel;
// CHECK-NEXT: ::bishengir::npu::SynchronizationModel synchronizationModel;
// CHECK: kDavC220Profile{
// CHECK-NEXT: NPUArchitectureKind::kDavC220,
// CHECK-NEXT: ExecutionModel::kMemBased,
// CHECK-NEXT: SynchronizationModel::kImplicit
// CHECK: kDavC310Profile{
// CHECK-NEXT: NPUArchitectureKind::kDavC310,
// CHECK-NEXT: ExecutionModel::kRegBased,
// CHECK-NEXT: SynchronizationModel::kExplicit
// CHECK: inline ExecutionModel getExecutionModel(
// CHECK-SAME: NPUArchitectureKind architecture)
// CHECK: inline SynchronizationModel getSynchronizationModel(
// CHECK-SAME: NPUArchitectureKind architecture)
// CHECK: return profile ? profile->synchronizationModel
// CHECK-SAME: : SynchronizationModel::kUnknown;
