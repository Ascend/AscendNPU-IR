// RUN: bishengir-opt %s --decompose-reduction 2>&1 | FileCheck %s
// CHECK: %[[Tensor:.*]] = tt.reshape %{{.*}} : tensor<4x1x16xi8, #linear> -> tensor<2x2x1x16xi8, #linear1>
// CHECK-NEXT: %{{.*}} = "tt.reduce"(%[[Tensor]]) <{axis = 0 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 1, 16], warpsPerCTA = [2, 16, 1], order = [2, 0, 1]}>
#linear = #ttg.linear<{register = [], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0]], warp = [[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 1, 0, 0]], warp = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], block = []}>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, "ttg.enable-bishengir-simt-optimization" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 221184 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_flip_3d(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<4x1x1xi32, #linear>
    %cst_0 = arith.constant dense<16> : tensor<4x1x1xi32, #blocked>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #linear}>}>>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #linear}>}>> -> tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #linear}>>
    %3 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %4 = tt.expand_dims %2 {axis = 2 : i32} : tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #linear}>> -> tensor<4x1x1xi32, #linear>
    %5 = tt.expand_dims %3 {axis = 2 : i32} : tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<4x1x1xi32, #blocked>
    %6 = arith.muli %4, %cst : tensor<4x1x1xi32, #linear>
    %7 = arith.muli %5, %cst_0 : tensor<4x1x1xi32, #blocked>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #linear}>}>>
    %9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %10 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #linear}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %11 = tt.expand_dims %9 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.expand_dims %10 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<1x1x16xi32, #linear>
    %13 = tt.expand_dims %11 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x16xi32, #blocked>
    %14 = tt.broadcast %6 : tensor<4x1x1xi32, #linear> -> tensor<4x1x16xi32, #linear>
    %15 = tt.broadcast %7 : tensor<4x1x1xi32, #blocked> -> tensor<4x1x16xi32, #blocked>
    %16 = tt.broadcast %12 : tensor<1x1x16xi32, #linear> -> tensor<4x1x16xi32, #linear>
    %17 = tt.broadcast %13 : tensor<1x1x16xi32, #blocked> -> tensor<4x1x16xi32, #blocked>
    %18 = arith.addi %14, %16 : tensor<4x1x16xi32, #linear>
    %19 = arith.addi %15, %17 : tensor<4x1x16xi32, #blocked>
    %20 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x1x16x!tt.ptr<i8>, #linear>
    %21 = tt.addptr %20, %18 : tensor<4x1x16x!tt.ptr<i8>, #linear>, tensor<4x1x16xi32, #linear>
    %22 = tt.load %21 : tensor<4x1x16x!tt.ptr<i8>, #linear>
    %23 = tt.reshape %22 : tensor<4x1x16xi8, #linear> -> tensor<2x2x1x16xi8, #linear1>
    %24 = "tt.reduce"(%23) <{axis = 0 : i32}> ({
    ^bb0(%arg5: i8, %arg6: i8):
      %35 = arith.xori %arg5, %arg6 : i8
      tt.reduce.return %35 : i8
    }) : (tensor<2x2x1x16xi8, #linear1>) -> tensor<2x1x16xi8, #ttg.slice<{dim = 0, parent = #linear1}>>
    %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<2x1x16xi8, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x2x1x16xi8, #linear1>
    %26 = tt.broadcast %25 : tensor<1x2x1x16xi8, #linear1> -> tensor<2x2x1x16xi8, #linear1>
    %27 = arith.xori %23, %26 : tensor<2x2x1x16xi8, #linear1>
    %28 = "tt.reduce"(%27) <{axis = 1 : i32}> ({
    ^bb0(%arg5: i8, %arg6: i8):
      %35 = arith.xori %arg5, %arg6 : i8
      tt.reduce.return %35 : i8
    }) : (tensor<2x2x1x16xi8, #linear1>) -> tensor<2x1x16xi8, #ttg.slice<{dim = 1, parent = #linear1}>>
    %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<2x1x16xi8, #ttg.slice<{dim = 1, parent = #linear1}>> -> tensor<2x1x1x16xi8, #linear1>
    %30 = tt.broadcast %29 : tensor<2x1x1x16xi8, #linear1> -> tensor<2x2x1x16xi8, #linear1>
    %31 = arith.xori %27, %30 : tensor<2x2x1x16xi8, #linear1>
    %32 = tt.reshape %31 : tensor<2x2x1x16xi8, #linear1> -> tensor<4x1x16xi8, #blocked>
    %33 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x1x16x!tt.ptr<i8>, #blocked>
    %34 = tt.addptr %33, %19 : tensor<4x1x16x!tt.ptr<i8>, #blocked>, tensor<4x1x16xi32, #blocked>
    tt.store %34, %32 : tensor<4x1x16x!tt.ptr<i8>, #blocked>
    tt.return
  }
}


