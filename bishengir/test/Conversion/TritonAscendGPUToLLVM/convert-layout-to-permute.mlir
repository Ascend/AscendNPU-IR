// RUN: bishengir-opt --allocate-ascend-shared-memory --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s

// This test exercises the layout-conversion path that now maps through
// `ascend_dpx.permute` before being lowered further by the DPX-to-HIVM pass.

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [2, 1, 16], warpsPerCTA = [32, 1, 1], order = [2, 0, 1]}>
#linear = #ttg.linear<{register = [[0, 0, 64], [0, 0, 1], [0, 0, 2]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [1, 0, 0]], warp = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.num-warps" = 32 : i32} {
  // CHECK-LABEL: @convert_layout_to_permute
  // CHECK:       ascend_dpx.permute
  tt.func public @convert_layout_to_permute(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x2x128x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<64x2x128x!tt.ptr<f16>, #blocked>
    %res = ttg.convert_layout %1 : tensor<64x2x128xf16, #blocked> -> tensor<64x2x128xf16, #linear>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x2x128x!tt.ptr<f16>, #linear>
    tt.store %3, %res : tensor<64x2x128x!tt.ptr<f16>, #linear>
    tt.return
  }
}
