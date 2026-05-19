// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// atomic_rmw mask handling:
//   tensor<i1> mask is per-lane lowered into an llvm.cond_br that branches
//   between the atomic block (executes on masked-in lanes) and the exit
//   block (masked-out lanes skip the atomic and receive llvm.mlir.undef as
//   the value of the result block argument).
//
// CHECK-LABEL: @scalar_atomic_add_with_mask
// CHECK:      llvm.cond_br %{{[0-9]+}}, ^[[ATOMIC_BB:bb[0-9]+]], ^[[EXIT_BB:bb[0-9]+]](%{{[0-9]+}} : i32)
// CHECK:      ^[[ATOMIC_BB]]:
// CHECK:        ascend_dpx.atomic_add
// CHECK:        llvm.br ^[[EXIT_BB]](%{{[0-9]+}} : i32)
// CHECK:      ^[[EXIT_BB]](%{{[0-9]+}}: i32):

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {
  tt.func @scalar_atomic_add_with_mask(
      %ptr: !tt.ptr<i32>, %val: i32, %mask: i1, %out: !tt.ptr<i32>) {
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %mask
        : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %0 : !tt.ptr<i32>
    tt.return
  }
}
