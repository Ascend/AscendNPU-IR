// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def ExecutionModelFeature : EnumArchitectureFeature<"ExecutionModel">;
def MemBased : EnumArchitectureFeatureValue<ExecutionModelFeature, "MemBased">;

def DavC220FeatureProfile : ArchitectureFeatureProfile<Dav_C220, [MemBased]>;

// CHECK: error: missing feature profile for architecture: dav-c310
// CHECK: error: missing feature profile for architecture: dav-m300
