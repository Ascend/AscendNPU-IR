// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def CustomFeature : ArchitectureFeature<"CustomFeature">;

def DavC220FeatureProfile : ArchitectureFeatureProfile<Dav_C220, []>;
def DavM300FeatureProfile : ArchitectureFeatureProfile<Dav_M300, []>;
def DavC310FeatureProfile : ArchitectureFeatureProfile<Dav_C310, []>;

// CHECK: error: unsupported architecture feature kind: CustomFeature
