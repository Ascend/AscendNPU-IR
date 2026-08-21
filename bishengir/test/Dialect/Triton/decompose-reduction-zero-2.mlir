// RUN: bishengir-opt -decompose-reduction %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 32, 1], order = [2, 0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 8, 4], order = [2, 0, 1]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64]], warp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], block = []}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 31 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_unk_fused_embedding_eq_sum_where_zeros__2(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg7: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg8: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x64x1xi64, #blocked>
    %cst_0 = arith.constant dense<11734> : tensor<1x64x1xi64, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<1x64x1xi64, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<1x64x1xi64, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x64x128xf32, #blocked>
    %c18856_i32 = arith.constant 18856 : i32
    %c50_i32 = arith.constant 50 : i32
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %0, %c1_i32 : i32
    %4 = arith.minsi %3, %arg3 : i32
    %5 = arith.cmpi slt, %0, %4 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>>
    %8 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %10 = tt.expand_dims %8 {axis = 1 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x128xi32, #blocked>
    %11 = tt.expand_dims %9 {axis = 1 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x1x128xi32, #blocked1>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x64x1xi32, #blocked>
    %15 = arith.muli %0, %c50_i32 : i32
    %16 = tt.splat %15 : i32 -> tensor<1x64x1xi32, #blocked>
    %17 = arith.addi %14, %16 : tensor<1x64x1xi32, #blocked>
    %18 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x64x1x!tt.ptr<i64>, #blocked>
    %19 = tt.addptr %18, %17 : tensor<1x64x1x!tt.ptr<i64>, #blocked>, tensor<1x64x1xi32, #blocked>
    %20 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x64x128x!tt.ptr<f32>, #blocked>
    %21 = arith.muli %0, %c18856_i32 : i32
    %22 = tt.splat %21 : i32 -> tensor<1x1x128xi32, #blocked1>
    %23 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x1x128x!tt.ptr<f32>, #blocked1>
    %24 = tt.splat %5 : i1 -> tensor<1x64x1xi1, #blocked>
    %25 = tt.splat %5 : i1 -> tensor<1x64x128xi1, #blocked>
    %26 = tt.splat %5 : i1 -> tensor<1x1x128xi1, #blocked1>
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%27: i32):  // 2 preds: ^bb0, ^bb2
    %28 = arith.cmpi slt, %27, %2 : i32
    cf.cond_br %28, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %29 = arith.muli %27, %c128_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<1x1x128xi32, #blocked>
    %31 = tt.splat %29 : i32 -> tensor<1x1x128xi32, #blocked1>
    %32 = arith.addi %30, %10 : tensor<1x1x128xi32, #blocked>
    %33 = arith.addi %31, %11 : tensor<1x1x128xi32, #blocked1>
    %34 = tt.load %19, %24, %cst_2 : tensor<1x64x1x!tt.ptr<i64>, #blocked>
    %35 = arith.cmpi eq, %34, %cst : tensor<1x64x1xi64, #blocked>
    %36 = arith.select %35, %cst_2, %34 : tensor<1x64x1xi1, #blocked>, tensor<1x64x1xi64, #blocked>
    %37 = arith.addi %36, %cst_0 : tensor<1x64x1xi64, #blocked>
    %38 = arith.cmpi slt, %36, %cst_2 : tensor<1x64x1xi64, #blocked>
    %39 = arith.select %38, %37, %36 : tensor<1x64x1xi1, #blocked>, tensor<1x64x1xi64, #blocked>
    %40 = arith.muli %39, %cst_1 : tensor<1x64x1xi64, #blocked>
    %41 = tt.broadcast %40 : tensor<1x64x1xi64, #blocked> -> tensor<1x64x128xi64, #blocked>
    %42 = arith.extsi %32 : tensor<1x1x128xi32, #blocked> to tensor<1x1x128xi64, #blocked>
    %43 = tt.broadcast %42 : tensor<1x1x128xi64, #blocked> -> tensor<1x64x128xi64, #blocked>
    %44 = arith.addi %43, %41 : tensor<1x64x128xi64, #blocked>
    %45 = tt.addptr %20, %44 : tensor<1x64x128x!tt.ptr<f32>, #blocked>, tensor<1x64x128xi64, #blocked>
    %46 = tt.load %45, %25, %cst_3 : tensor<1x64x128x!tt.ptr<f32>, #blocked>
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 32, 1], order = [2, 0, 1]}>
    // CHECK-DAG: [[BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 4, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 32], order = [2, 0, 1]}>
    // CHECK{LITERAL}: #linear = #ttg.linear<{register = [], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]], warp = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64]], block = []}>
    // CHECK-NOT: tt.reshape
    // CHECK: %[[C0:.+]] = ttg.convert_layout [[CST:%.*]] : tensor<1x64x128xf32, [[BLOCKED]]> -> tensor<1x64x128xf32, [[BLOCKED2]]>
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"(%[[C0]]) <{axis = 1 : i32}>  
    // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[REDUCE:.*]] : tensor<1x128xf32, #ttg.slice<{dim = 1, parent = [[BLOCKED2]]}>> -> tensor<1x1x128xf32, #linear>
    // CHECK: ttg.convert_layout %[[RESHAPE:.*]] :
    %47 = "tt.reduce"(%46) <{axis = 1 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
      %53 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %53 : f32
    }) {allocation.offset = 0 : i32} : (tensor<1x64x128xf32, #blocked>) -> tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %48 = tt.reshape %47 : tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x128xf32, #linear>
    %49 = arith.addi %33, %22 : tensor<1x1x128xi32, #blocked1>
    %50 = tt.addptr %23, %49 : tensor<1x1x128x!tt.ptr<f32>, #blocked1>, tensor<1x1x128xi32, #blocked1>
    %51 = ttg.convert_layout %48 {allocation.offset = 0 : i32} : tensor<1x1x128xf32, #linear> -> tensor<1x1x128xf32, #blocked1>
    tt.store %50, %51, %26 : tensor<1x1x128x!tt.ptr<f32>, #blocked1>
    %52 = arith.addi %27, %c1_i32 overflow<nsw> : i32
    cf.br ^bb1(%52 : i32)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}

