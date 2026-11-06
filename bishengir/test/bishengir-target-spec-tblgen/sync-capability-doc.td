// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-target-capability-doc \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include -I %llvm_src_root/include %s | FileCheck %s
// clang-format on

// The production catalogue is the source of the support table.
include "bishengir/Target/NPU/Sync/SyncCapabilityProfile.td"

// CHECK: # Target capability support
// CHECK: ## Capability `get_buf`
// get_buf and rls_buf are field-free schemas.
// CHECK: Fields:
// CHECK-NOT: `sync_mode`
// CHECK: ## Capability `rls_buf`
// CHECK: Fields:
// CHECK-NOT: `sync_mode`
// CHECK: ## Capability `try_wait`
// CHECK: `sync_mode`

// Only the intra_block counter query has a support row.
// CHECK: ### `dav-920r1`
// CHECK: Coverage: partial
// CHECK: matches no row is Unknown
// CHECK: intra_block
// CHECK: ### `dav-c310`
// CHECK: intra_block
// CHECK: ## No profile
// CHECK: returns Unknown
// CHECK: `dav-c220`
// CHECK: `dav-m300`
