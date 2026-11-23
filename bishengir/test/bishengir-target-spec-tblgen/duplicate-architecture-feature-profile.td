// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatures.td"

def DuplicateDavC310FeatureProfile
    : ArchitectureFeatureProfile<Dav_C310, [RegBased]>;

// CHECK: error: duplicate feature profile for architecture: dav-c310
