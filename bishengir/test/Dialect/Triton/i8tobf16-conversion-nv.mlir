/// RUN: bishengir-opt %s -convert-triton-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @i8tobf16(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %4 = arith.muli %1, %cst : tensor<16x1xi32, #blocked>
    %5 = tt.broadcast %4 : tensor<16x1xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<1x32xi32, #blocked> -> tensor<16x32xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<16x32xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<16x32x!tt.ptr<i8>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<16x32x!tt.ptr<i8>, #blocked>, tensor<16x32xi32, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x32x!tt.ptr<bf16>, #blocked>
    %11 = tt.addptr %10, %7 : tensor<16x32x!tt.ptr<bf16>, #blocked>, tensor<16x32xi32, #blocked>
    %12 = tt.load %9 : tensor<16x32x!tt.ptr<i8>, #blocked>
    // CHECK: llvm.extractvalue
    // CHECK: llvm.extractvalue
    // CHECK: llvm.extractvalue
    // CHECK: llvm.extractvalue
    // CHECK-DAG: llvm.sitofp %[[R0:.+]] : i8 to bf16
    // CHECK-NEXT: llvm.sitofp %[[R1:.+]] : i8 to bf16
    // CHECK-NEXT: llvm.sitofp %[[R2:.+]] : i8 to bf16
    // CHECK-NEXT: llvm.sitofp %[[R3:.+]] : i8 to bf16
    %13 = arith.sitofp %12 : tensor<16x32xi8, #blocked> to tensor<16x32xbf16, #blocked>
    tt.store %11, %13 : tensor<16x32x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
