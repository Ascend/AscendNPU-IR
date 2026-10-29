// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-target-capability-doc \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include -I %llvm_src_root/include \
// RUN:   -I %bishengir_src_root/unittests/Target/Capability %s | FileCheck %s
// clang-format on

// Several capabilities share TriadSchema. Each architecture table stays under
// its capability, including when completeness differs.
include "CapabilityTest.td"

// Capabilities are emitted in record-name order. Triad and partial_triad share
// one schema and differ in completeness. address_only keeps its slot property.
// CHECK: ## Capability `address_only`
// CHECK: `address_slot.address_space`
// CHECK: global
// CHECK: ## Capability `empty_complete`
// CHECK: ### `dav-c220`
// CHECK: Coverage: complete
// CHECK: matches no row is Unsupported
// CHECK: ### `dav-m300`
// CHECK: Coverage: complete
// CHECK: ## Capability `empty_partial`
// CHECK: Coverage: partial
// CHECK: matches no row is Unknown
// CHECK: ## Capability `partial_triad`
// CHECK: `lhs.layout`
// CHECK: Coverage: partial
// CHECK: matches no row is Unknown
// CHECK: ## Capability `triad`
// CHECK: `lhs.layout`
// CHECK: ### `dav-c220`
// CHECK: Coverage: complete
// CHECK: matches no row is Unsupported
// CHECK: ### `dav-m300`
// CHECK: Coverage: complete
// CHECK-NOT: quant_scale presence are outside
