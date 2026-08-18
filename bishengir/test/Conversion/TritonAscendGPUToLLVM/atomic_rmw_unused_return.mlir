// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm -canonicalize -cse %s | FileCheck %s
//
// atomic_rmw unused-return DCE:
//   when the value returned by tt.atomic_rmw is not consumed, the lowering
//   still emits ascend_dpx.atomic_add for the side effect, but the
//   downstream CSE/DCE folds away the result-bearing block argument /
//   shared-memory broadcast scaffolding.

// CHECK-LABEL: @atomic_add_no_use
// CHECK:       ascend_dpx.atomic_add
// CHECK-NOT:   ascend_dpx.store {{.*}} : <3>, vector<{{.*}}>
// CHECK-NOT:   ascend_dpx.sync_threads
// CHECK-NOT:   ascend_dpx.load {{.*}} : (!llvm.ptr<3>)

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 1111 : i32
} {
  tt.func @atomic_add_no_use(%ptr: !tt.ptr<i32>, %val: i32) {
    %true = arith.constant true
    %0 = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %true
        : (!tt.ptr<i32>, i32, i1) -> i32
    // %0 is never used.
    tt.return
  }
}
