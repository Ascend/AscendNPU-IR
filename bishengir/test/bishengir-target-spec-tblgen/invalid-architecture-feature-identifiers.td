// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def InvalidCppFeature : EnumArchitectureFeature<"Invalid-Feature">;
def InvalidCppValue
    : EnumArchitectureFeatureValue<InvalidCppFeature, "Invalid-Value">;

def InvalidAccessorFeature
    : EnumArchitectureFeature<"ValidFeature", "", "invalid-accessor">;
def ValidValue
    : EnumArchitectureFeatureValue<InvalidAccessorFeature, "ValidValue">;

def DavC220FeatureProfile
    : ArchitectureFeatureProfile<Dav_C220, [InvalidCppValue, ValidValue]>;
def DavM300FeatureProfile
    : ArchitectureFeatureProfile<Dav_M300, [InvalidCppValue, ValidValue]>;
def DavC310FeatureProfile
    : ArchitectureFeatureProfile<Dav_C310, [InvalidCppValue, ValidValue]>;

// CHECK-DAG: invalid architecture feature C++ name: Invalid-Feature
// CHECK-DAG: invalid architecture feature value C++ name: Invalid-Value
// CHECK-DAG: invalid architecture feature accessor: invalid-accessor
