// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def ExecutionModelFeature : EnumArchitectureFeature<"ExecutionModel">;
def MemBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "MemBased">;
def RegBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "RegBased">;

def DavC220FeatureProfile
    : ArchitectureFeatureProfile<Dav_C220, [MemBased, RegBased]>;
def DavM300FeatureProfile : ArchitectureFeatureProfile<Dav_M300, [RegBased]>;
def DavC310FeatureProfile : ArchitectureFeatureProfile<Dav_C310, [RegBased]>;

// CHECK: error: multiple values for architecture feature
// CHECK-SAME: 'execution-model' in architecture: dav-c220
