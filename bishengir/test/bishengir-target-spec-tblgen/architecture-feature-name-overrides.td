// RUN: bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"

// Acronyms are kept together when deriving profile fields and spellings.
// Overrides retain the ability to choose a stable external spelling or API.
def URLModeFeature
    : EnumArchitectureFeature<"URLMode", "external-url-mode", "readURLMode">;
def HTTPReady : EnumArchitectureFeatureValue<URLModeFeature, "HTTPReady">;
def LowLatency : EnumArchitectureFeatureValue<URLModeFeature, "LowLatency",
                                              "low-latency-mode">;

def DavC220FeatureProfile : ArchitectureFeatureProfile<Dav_C220, [HTTPReady]>;
def DavM300FeatureProfile : ArchitectureFeatureProfile<Dav_M300, [LowLatency]>;
def DavC310FeatureProfile : ArchitectureFeatureProfile<Dav_C310, [LowLatency]>;

// CHECK: enum class URLMode {
// CHECK: case URLMode::kHTTPReady: return "http-ready";
// CHECK: case URLMode::kLowLatency: return "low-latency-mode";
// CHECK: .Case("http-ready", URLMode::kHTTPReady)
// CHECK: .Case("low-latency-mode", URLMode::kLowLatency)
// CHECK: struct ArchitectureFeatureProfile {
// CHECK-NEXT: NPUArchitectureKind architecture;
// CHECK-NEXT: ::bishengir::npu::URLMode urlMode;
// CHECK: inline URLMode readURLMode(NPUArchitectureKind architecture)
// CHECK: return profile ? profile->urlMode : URLMode::kUnknown;
