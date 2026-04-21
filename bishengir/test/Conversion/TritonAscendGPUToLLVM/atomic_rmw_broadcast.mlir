// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Test that tensor tt.atomic_rmw with thread-redundant layout broadcasts the
// result via shared memory so that all SIMT threads see the leader's value.
//
//   @triton.jit
//   def _kernel_a(counter, output, src, H: tl.constexpr, H_PAD: tl.constexpr):
//       offs = tl.arange(0, H_PAD)
//       mask = offs < H
//       row  = tl.load(src + offs, mask=mask)
//       slot = tl.atomic_add(counter, 1)
//       tl.store(output + slot * H + offs, row, mask=mask)
//
// Without the broadcast fix, non-leader threads receive llvm.mlir.undef as the
// atomic return value, causing slot*H+offs to compute wrong addresses.  The bug
// is hidden when slot==0 (0*H==0 regardless of broadcast) and only manifests
// for non-zero slots.
//
// The blocked layout below maps 128 threads (4 warps x 32 lanes) to a single
// tensor element, making 127 of 128 threads redundant.  This triggers the
// leader-only conditional branch around the atomic, which in turn requires
// shared-memory broadcast of the result.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {

  // ---- Tensor atomic: broadcast MUST be generated ----------------------------
  //
  // CHECK-LABEL: @tensor_atomic_add_i32_broadcast
  // CHECK:       ascend_dpx.atomic_add
  // CHECK:       ascend_dpx.sync_threads
  tt.func @tensor_atomic_add_i32_broadcast(
      %ptr: tensor<1x!tt.ptr<i32>, #blocked>,
      %val: tensor<1xi32, #blocked>,
      %mask: tensor<1xi1, #blocked>,
      %out: tensor<1x!tt.ptr<i32>, #blocked>) {
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %mask
        {allocation.offset = 0 : i32}
        : (tensor<1x!tt.ptr<i32>, #blocked>, tensor<1xi32, #blocked>,
           tensor<1xi1, #blocked>) -> tensor<1xi32, #blocked>
    tt.store %out, %0 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // ---- Scalar atomic WITH allocation.offset: broadcast MUST be generated -----
  //
  // This matches the production pattern where a scalar atomic result is used
  // in address arithmetic that feeds a tt.splat → tensor store.  The allocation
  // pass sets allocation.offset to signal that broadcast is needed.
  //
  // CHECK-LABEL: @scalar_atomic_add_i64_with_alloc_broadcast
  // CHECK:       ascend_dpx.atomic_add
  // CHECK:       ascend_dpx.sync_threads
  tt.func @scalar_atomic_add_i64_with_alloc_broadcast(
      %ptr: !tt.ptr<i64>, %val: i64, %out: !tt.ptr<i64>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %true
        {allocation.offset = 0 : i32}
        : (!tt.ptr<i64>, i64, i1) -> i64
    tt.store %out, %0 : !tt.ptr<i64>
    tt.return
  }

  // ---- Scalar atomic without allocation.offset: broadcast must NOT be generated
  //
  // CHECK-LABEL: @scalar_atomic_add_i32_no_broadcast
  // CHECK:       ascend_dpx.atomic_add
  // CHECK-NOT:   ascend_dpx.sync_threads
  tt.func @scalar_atomic_add_i32_no_broadcast(
      %ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %true
        : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }
}
