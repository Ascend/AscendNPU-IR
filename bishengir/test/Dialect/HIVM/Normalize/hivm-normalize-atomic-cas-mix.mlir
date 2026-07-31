// RUN: bishengir-opt --hivm-normalize-ops --hivm-mark-tightly-coupled-buffer --hivm-split-mix-kernel --hivm-infer-mem-scope %s -split-input-file -verify-diagnostics | FileCheck %s

// Verifies the cbuf->cbuf fix end-to-end on a MIX kernel. The CAS decomposition
// is wrapped in a VECTOR-tagged scope.scope so SplitMixKernel routes the whole
// group (copies + scratch + vcmp/vsel + lock) to _mix_aiv. After
// infer-hivm-mem-scope the scratch allocs in _mix_aiv are UB (not cbuf), so the
// value->scratch copies stay UB->UB (legal) instead of becoming illegal
// cbuf->cbuf on Ascend950. Without the scope wrapper the copies/scratch stay in
// _mix_aic, become cbuf after infer-hivm-mem-scope, and the value->scratch copy
// is rejected as illegal cbuf->cbuf. Interior scope ops are NOT individually
// tagged; CrossCoreGSS resolves their core type via the tagged scope ancestor
// (resolveCoreTypeViaScopeAncestor in SyncSolverIRTranslator).

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_mix_routes_to_aiv_mix_aic(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix
// The AIC (cube) clone must contain NO CAS scratch allocs, copies, lock, or
// scope — they all route to _mix_aiv. A stray alloc here would become cbuf and
// re-introduce the cbuf->cbuf copy.
// CHECK-NOT: scope.scope
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK-NOT: sync_block

// CHECK-LABEL: func.func @test_NormalizeAtomicOps_CAS_mix_routes_to_aiv_mix_aiv(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix
// CHECK: scope.scope : () -> () {
// Scratch allocs land INSIDE the scope and infer to UB (not cbuf). Interior ops
// are NOT individually tagged; CrossCoreGSS resolves their core type via the
// tagged scope ancestor.
// CHECK: memref.alloc(){{.*}}#hivm.address_space<ub>
// CHECK: memref.alloc(){{.*}}#hivm.address_space<ub>
// CHECK: memref.alloc(){{.*}}#hivm.address_space<ub>
// CHECK: %[[LOCK:.*]] = hivm.hir.create_sync_block_lock : memref<1xi64>
// CHECK: hivm.hir.sync_block_lock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: memref.copy {{.*}}
// CHECK: %[[CMP:.*]] = hivm.hir.vcmp
// CHECK: %[[SEL:.*]] = hivm.hir.vsel
// CHECK: hivm.hir.sync_block_unlock lock_var(%[[LOCK]] : memref<1xi64>)
// CHECK: scope.return
// CHECK: } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_NormalizeAtomicOps_CAS_mix_routes_to_aiv(%arg0: memref<?xi16>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %alloc = memref.alloc() : memref<256xi16>
    %alloc_0 = memref.alloc() : memref<256xi16>
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
    hivm.hir.atomic_cas ins(%alloc, %alloc_0 : memref<256xi16>, memref<256xi16>) outs(%reinterpret_cast : memref<256xi16, strided<[1]>>)
    return
  }
}
