// RUN: bishengir-opt %s -debug-only=triton-remap -triton-remap | FileCheck %s

// CHECK-LABEL: @remap_fdiv_rn

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @remap_fdiv_rn(%arg11: f32, %arg12: f32)
    attributes {nvvm.kernel = 1} {
    // CHECK-NOT: llvm.call @__nv_fdiv_rn
    // CHECK: remap_fdiv_rn_vf_simt
    // Check: fdiv
    %0 = llvm.call @__nv_fdiv_rn(%arg11, %arg12) : (f32, f32) -> f32

    // in case applyPatternsGreedily remove the fdiv ops
    %1 = llvm.call @__nv_rsqrtf(%0) : (f32) -> f32
    llvm.return
  }

  llvm.func @__nv_fdiv_rn(f32, f32) -> f32
  llvm.func @__nv_rsqrtf(f32) -> f32
}
