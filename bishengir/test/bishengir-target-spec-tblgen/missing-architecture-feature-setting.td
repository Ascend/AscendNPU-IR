// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def ExecutionModelFeature : EnumArchitectureFeature<"ExecutionModel">;
def MemBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "MemBased">;
def RegBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "RegBased">;

def SynchronizationModelFeature
    : EnumArchitectureFeature<"SynchronizationModel">;
def ExplicitSynchronization
    : EnumArchitectureFeatureValue<SynchronizationModelFeature, "Explicit">;

def DavC220FeatureProfile : ArchitectureFeatureProfile<Dav_C220, [MemBased]>;
def DavM300FeatureProfile
    : ArchitectureFeatureProfile<Dav_M300, [RegBased, ExplicitSynchronization]>;
def DavC310FeatureProfile
    : ArchitectureFeatureProfile<Dav_C310, [RegBased, ExplicitSynchronization]>;

// CHECK: error: missing value for architecture feature
// CHECK-SAME: 'synchronization-model' in architecture: dav-c220
