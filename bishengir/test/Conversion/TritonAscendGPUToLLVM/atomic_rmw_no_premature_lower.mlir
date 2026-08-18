// RUN: bishengir-opt -canonicalize -cse %s | FileCheck %s
//
// atomic_rmw mid-stage delayed lowering:
//   Through the Triton / TritonGPU mid-stage passes (canonicalize, CSE,
//   reordering, layout assignment, etc.) `tt.atomic_rmw` MUST stay in
//   tensor form. Scalarize-to-per-thread happens only inside the final
//   `convert-triton-ascend-gpu-to-llvm` pass (which produces
//   `ascend_dpx.atomic_*`). This is a regression guard: an accidental
//   pattern that scalarizes earlier would break layout/vectorization
//   analysis that depends on the tensor-shaped semantics.

// CHECK-LABEL: @tensor_atomic_rmw_survives_mid_passes
// CHECK: tt.atomic_rmw
// CHECK-NOT: ascend_dpx.atomic
// CHECK: tt.return

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
  tt.func @tensor_atomic_rmw_survives_mid_passes(
      %ptr: tensor<256x!tt.ptr<i32>, #blocked>,
      %val: tensor<256xi32, #blocked>,
      %mask: tensor<256xi1, #blocked>) {
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %mask
        : (tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>,
           tensor<256xi1, #blocked>) -> tensor<256xi32, #blocked>
    tt.return
  }
}
