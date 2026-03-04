// RUN: bishengir-opt -split-input-file -optimize-layouts %s | FileCheck %s

// CHECK-LABEL: triton_red_fused_embedding_sum_0_grid_any

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 16], warpsPerCTA = [32, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 32], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, "ttg.enable-bishengir-simt-optimization" = 11 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 32768 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_red_fused_embedding_sum_0_grid_any(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg7: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg8: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4000> : tensor<128x1xi32, #blocked>
    %cst_0 = arith.constant dense<4000> : tensor<1x32xi32, #blocked>
    %cst_1 = arith.constant dense<4000> : tensor<1x32xi32, #blocked1>
    %cst_2 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_3 = arith.constant dense<0> : tensor<128x32xi64, #blocked>
    %cst_4 = arith.constant dense<128> : tensor<128x32xi64, #blocked>
    %cst_5 = arith.constant dense<9000> : tensor<128x32xi64, #blocked>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked1>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %6 = tt.splat %1 : i32 -> tensor<128x1xi32, #blocked>
    %7 = tt.splat %1 : i32 -> tensor<128x1xi32, #blocked2>
    %8 = arith.addi %6, %4 : tensor<128x1xi32, #blocked>
    %9 = arith.addi %7, %5 : tensor<128x1xi32, #blocked2>
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %12 = tt.expand_dims %10 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %13 = tt.expand_dims %11 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %14 = arith.divsi %8, %cst_2 : tensor<128x1xi32, #blocked>
    %15 = arith.remsi %8, %cst_2 : tensor<128x1xi32, #blocked>
    %16 = arith.muli %14, %cst : tensor<128x1xi32, #blocked>
    %17 = tt.broadcast %16 : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %18 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<128x32x!tt.ptr<i64>, #blocked>
    %19 = arith.extsi %15 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
    %20 = tt.broadcast %19 : tensor<128x1xi64, #blocked> -> tensor<128x32xi64, #blocked>
    %21 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    cf.br ^bb1(%c0_i32, %cst_6 : i32, tensor<128x32xf32, #blocked1>)
  ^bb1(%22: i32, %23: tensor<128x32xf32, #blocked1>):  // 2 preds: ^bb0, ^bb2
    %24 = arith.cmpi slt, %22, %c4000_i32 : i32
    cf.cond_br %24, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %25 = tt.splat %22 : i32 -> tensor<1x32xi32, #blocked1>
    %26 = tt.splat %22 : i32 -> tensor<1x32xi32, #blocked>
    %27 = arith.addi %25, %12 : tensor<1x32xi32, #blocked1>
    %28 = arith.addi %26, %13 : tensor<1x32xi32, #blocked>
    %29 = arith.cmpi slt, %27, %cst_1 : tensor<1x32xi32, #blocked1>
    %30 = arith.cmpi slt, %28, %cst_0 : tensor<1x32xi32, #blocked>
    %31 = tt.broadcast %28 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %32 = arith.addi %31, %17 : tensor<128x32xi32, #blocked>
    %33 = tt.addptr %18, %32 : tensor<128x32x!tt.ptr<i64>, #blocked>, tensor<128x32xi32, #blocked>
    %34 = tt.broadcast %29 : tensor<1x32xi1, #blocked1> -> tensor<128x32xi1, #blocked1>
    %35 = tt.broadcast %30 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
    %36 = tt.load %33, %35, %cst_3 : tensor<128x32x!tt.ptr<i64>, #blocked>
    %37 = arith.addi %36, %cst_5 : tensor<128x32xi64, #blocked>
    %38 = arith.cmpi slt, %36, %cst_3 : tensor<128x32xi64, #blocked>
    %39 = arith.select %38, %37, %36 : tensor<128x32xi1, #blocked>, tensor<128x32xi64, #blocked>
    %40 = arith.muli %39, %cst_4 : tensor<128x32xi64, #blocked>
    %41 = arith.addi %20, %40 : tensor<128x32xi64, #blocked>
    %42 = tt.addptr %21, %41 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi64, #blocked>
    // CHECK-NOT: ttg.convert_layout
    %43 = ttg.convert_layout %42 {allocation.offset = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #blocked> -> tensor<128x32x!tt.ptr<f32>, #blocked1>
    %44 = tt.load %43, %34, %cst_6 : tensor<128x32x!tt.ptr<f32>, #blocked1>
    %45 = arith.addf %23, %44 : tensor<128x32xf32, #blocked1>
    %46 = arith.select %34, %45, %23 : tensor<128x32xi1, #blocked1>, tensor<128x32xf32, #blocked1>
    %47 = arith.addi %22, %c32_i32 overflow<nsw> : i32
    cf.br ^bb1(%47, %46 : i32, tensor<128x32xf32, #blocked1>)
  ^bb3:  // pred: ^bb1
    // CHECK: ttg.convert_layout
    %48 = ttg.convert_layout %23 : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked3>
    %49 = "tt.reduce"(%48) <{axis = 1 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
      %54 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %54 : f32
    }) : (tensor<128x32xf32, #blocked3>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    // CHECK: ttg.convert_layout
    %50 = ttg.convert_layout %49 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %51 = tt.expand_dims %50 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
    %52 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked2>
    %53 = tt.addptr %52, %9 : tensor<128x1x!tt.ptr<f32>, #blocked2>, tensor<128x1xi32, #blocked2>
    tt.store %53, %51 : tensor<128x1x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// -----

// CHECK-LABEL: test_linear_layout
// TODO: Add optimizations to linear layouts. For now make sure they are ignored, as they give runtime/percision errors.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, "ttg.enable-bishengir-simt-optimization" = 11 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 221184 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_linear_layout(%57 : tensor<1x1x16xi32, #ttg.linear<{register = [], lane = [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], warp = [[0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 0], [0, 0, 0]], block = []}>>, %20 : i1, %61: tensor<1x1x16x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 32, 1], order = [2, 0, 1]}>>) attributes {noinline = false} {

    // CHECK: ttg.convert_layout
    %62 = ttg.convert_layout %57 : tensor<1x1x16xi32, #ttg.linear<{register = [], lane = [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], warp = [[0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 0], [0, 0, 0]], block = []}>> -> tensor<1x1x16xi32, #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 32, 1], order = [2, 0, 1]}>>
    %63 = tt.splat %20 : i1 -> tensor<1x1x16xi1, #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 32, 1], order = [2, 0, 1]}>>
    tt.store %61, %62, %63 : tensor<1x1x16x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 32, 1], order = [2, 0, 1]}>>
    tt.return
  }
}