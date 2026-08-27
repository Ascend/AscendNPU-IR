// RUN: bishengir-opt %s -decompose-reduction -optimize-layouts --mlir-print-ir-after=optimize-layouts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 11 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.shared = 221184 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_zeros_sum(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg2: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1x1xi32, #blocked>
    // CHECK: ttg.convert_layout
    // CHECK: "tt.reduce"
    // CHECK: %[[C1:.+]] = arith.sitofp %[[C2:.+]]
    // CHECK-NOT: ttg.convert_layout
    %0 = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %4 = arith.addi %arg4, %arg5 : i32
      tt.reduce.return %4 : i32
    }) : (tensor<1x1xi32, #blocked>) -> tensor<1xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<1xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1xi32, #blocked1>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked1>
    %3 = arith.sitofp %1 : tensor<1xi32, #blocked1> to tensor<1xf32, #blocked1>
    tt.store %2, %3 : tensor<1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
