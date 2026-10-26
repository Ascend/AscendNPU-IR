// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-target-capability-doc \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include -I %llvm_src_root/include %s | FileCheck %s
// clang-format on

// The production catalogue is the source of the support table.
include "bishengir/Target/NPU/DMA/Fixpipe/FixpipeCapabilityProfile.td"

// CHECK: # Target capability support
// CHECK: ## Capability `fixpipe`
// CHECK: `src.element_type` (MLIR type)
// CHECK: `dst.element_type` (MLIR type)
// CHECK: `dma_mode`
// CHECK: `pre_quant_mode`
// CHECK: `pre_relu`
// CHECK: `channel_split`

// C220 has separate NZ2ND and NZ2NZ (printed as normal) rows, and no NZ2DN.
// CHECK: ### `dav-c220`
// CHECK: Coverage: partial
// CHECK: matches no row is Unknown
// CHECK: nz2nd
// CHECK: normal
// CHECK-NOT: nz2dn
// CHECK: ### `dav-c310`
// CHECK: nz2nd, nz2dn, normal
// CHECK: F322F16
// CHECK: QF322F32_PRE
// CHECK: NO_RELU
// CHECK: false
// CHECK: ## No profile
// CHECK: returns Unknown
// CHECK: `dav-m300`
// CHECK: quant_scale presence are outside
