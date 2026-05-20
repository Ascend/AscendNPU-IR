// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// atomic_cas independent path:
//   tt.atomic_cas → ascend_dpx.atomic_cas
//
// CAS does not flow through tt.atomic_rmw (which only covers
// add/and/or/xor/min/max/umin/umax/exch). It has its own conversion pattern
// (AtomicCASOpConversion in LoadStoreOpToLLVM.cpp) that lowers a
// (ptr, cmp, val) triple directly to ascend_dpx.AtomicCASOp, one per
// scalarized lane.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 8 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {
  // CHECK-LABEL: @tensor_atomic_cas_i32
  // CHECK: ascend_dpx.atomic_cas
  tt.func @tensor_atomic_cas_i32(
      %ptr: tensor<256x!tt.ptr<i32>, #blocked>,
      %cmp: tensor<256xi32, #blocked>,
      %val: tensor<256xi32, #blocked>) {
    %0 = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val
        : (tensor<256x!tt.ptr<i32>, #blocked>,
           tensor<256xi32, #blocked>,
           tensor<256xi32, #blocked>) -> tensor<256xi32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @tensor_atomic_cas_i64
  // CHECK: ascend_dpx.atomic_cas
  tt.func @tensor_atomic_cas_i64(
      %ptr: tensor<256x!tt.ptr<i64>, #blocked>,
      %cmp: tensor<256xi64, #blocked>,
      %val: tensor<256xi64, #blocked>) {
    %0 = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val
        : (tensor<256x!tt.ptr<i64>, #blocked>,
           tensor<256xi64, #blocked>,
           tensor<256xi64, #blocked>) -> tensor<256xi64, #blocked>
    tt.return
  }

  // Scalar-address atomic_cas: the result type is a plain scalar, not a
  // RankedTensorType. AtomicCASOpConversion must handle this without
  // dyn_cast-ing the result to RankedTensorType unconditionally (which
  // previously segfaulted on the scalar form).
  // CHECK-LABEL: @scalar_atomic_cas_i32
  // CHECK: ascend_dpx.atomic_cas
  tt.func @scalar_atomic_cas_i32(
      %ptr: !tt.ptr<i32>, %cmp: i32, %val: i32, %out: !tt.ptr<i32>) {
    %0 = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val
        : (!tt.ptr<i32>, i32, i32) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: @scalar_atomic_cas_i64
  // CHECK: ascend_dpx.atomic_cas
  tt.func @scalar_atomic_cas_i64(
      %ptr: !tt.ptr<i64>, %cmp: i64, %val: i64, %out: !tt.ptr<i64>) {
    %0 = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val
        : (!tt.ptr<i64>, i64, i64) -> i64
    tt.store %out, %0 : !tt.ptr<i64>
    tt.return
  }
}
