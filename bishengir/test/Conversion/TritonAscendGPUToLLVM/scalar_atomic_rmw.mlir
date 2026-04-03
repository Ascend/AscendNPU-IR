// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Test that scalar (non-tensor) tt.atomic_rmw is correctly lowered to
// ascend_dpx atomic ops. Before the fix, this would crash because the
// conversion unconditionally assumed the result type was a RankedTensorType.

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {
  // CHECK-LABEL: @scalar_atomic_add_i32
  // CHECK: ascend_dpx.atomic_add
  tt.func @scalar_atomic_add_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_add_f32
  // CHECK: ascend_dpx.atomic_add
  tt.func @scalar_atomic_add_f32(%ptr: !tt.ptr<f32>, %val: f32, %out: !tt.ptr<f32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %out, %0 : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_max_i32
  // CHECK: ascend_dpx.atomic_max
  tt.func @scalar_atomic_max_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw max, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_exch_i32
  // CHECK: ascend_dpx.atomic_exchange
  tt.func @scalar_atomic_exch_i32(%ptr: !tt.ptr<i32>, %val: i32, %out: !tt.ptr<i32>) {
    %true = arith.constant true
    %0 = tt.atomic_rmw exch, acq_rel, gpu, %ptr, %val, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }
}
