// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// atomic op coverage: every tt.atomic_rmw RMWOp kind in the SR scope must
// lower to its ascend_dpx.atomic_* counterpart. scalar_atomic_rmw.mlir
// already covers add / fadd / max / exch on i32/f32; this file covers the
// remaining ops in 规格2 集合 (and / or / xor / min / umin / umax).
//
// All ops use the scalar (single-element) path with a `true` mask so the
// CHECKs focus on the rmw_kind table; per-thread scalarize behavior is
// covered separately by atomic_rmw_broadcast.mlir and scalar_atomic_rmw.mlir.

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {

  // CHECK-LABEL: @scalar_atomic_and_i32
  // CHECK: ascend_dpx.atomic_and
  tt.func @scalar_atomic_and_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw and, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_or_i32
  // CHECK: ascend_dpx.atomic_or
  tt.func @scalar_atomic_or_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw or, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_xor_i32
  // CHECK: ascend_dpx.atomic_xor
  tt.func @scalar_atomic_xor_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw xor, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_min_i32
  // CHECK: ascend_dpx.atomic_min
  tt.func @scalar_atomic_min_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw min, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_umin_i32
  // CHECK: ascend_dpx.atomic_umin
  tt.func @scalar_atomic_umin_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw umin, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_umax_i32
  // CHECK: ascend_dpx.atomic_umax
  tt.func @scalar_atomic_umax_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw umax, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }
}
