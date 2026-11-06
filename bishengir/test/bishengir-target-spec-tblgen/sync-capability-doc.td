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

// Confirmed C310 rows are inherited by dav-920r1 under the
// forward-compatibility policy. Every catalogue architecture has a
// profile, so the "No profile" section is empty. The memory-based
// architectures carry surveyed empty coverages and answer Unknown.
// CHECK: ### `dav-920r1`
// CHECK: Coverage: partial
// CHECK: matches no row is Unknown
// CHECK: intra_block
// CHECK: ### `dav-c310`
// CHECK: intra_block
// CHECK: ## No profile
// CHECK: returns Unknown
// CHECK-NOT: `dav-c220`
// CHECK-NOT: `dav-m300`
