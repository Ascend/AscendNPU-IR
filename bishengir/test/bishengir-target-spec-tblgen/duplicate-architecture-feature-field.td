// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

def URLModeFeature : EnumArchitectureFeature<"URLMode", "url-mode-uppercase">;
def URLModeValue : EnumArchitectureFeatureValue<URLModeFeature, "Enabled">;

def UrlModeFeature : EnumArchitectureFeature<"UrlMode", "url-mode-titlecase">;
def UrlModeValue : EnumArchitectureFeatureValue<UrlModeFeature, "Enabled">;

def DavC220FeatureProfile
    : ArchitectureFeatureProfile<Dav_C220, [URLModeValue, UrlModeValue]>;
def DavM300FeatureProfile
    : ArchitectureFeatureProfile<Dav_M300, [URLModeValue, UrlModeValue]>;
def DavC310FeatureProfile
    : ArchitectureFeatureProfile<Dav_C310, [URLModeValue, UrlModeValue]>;

// CHECK: error: duplicate architecture feature profile field: urlMode
