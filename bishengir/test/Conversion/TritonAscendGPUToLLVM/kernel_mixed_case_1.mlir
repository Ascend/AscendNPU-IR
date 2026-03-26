// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// In this test case the kernel comprises 2 reduction operations. This is because
// the reduction is across 2 dimensions, so the compiler applies the reduction across
// the first dimension (0) first causing the shape to be reduced and then applies 
// another reduction to the result further reducing the shape.
//
// In this case, we find that the first reduction is not profiting from rewriting as
// packed warps of reduction ops. The second one, however, does. Thus there will be
// both the original triton reduction implementation AND the packed-warp version in this
// one file. We will know them because the first one uses xx bfly instructions and the second
// one is wrapped in an if statement with no bfly instructions.
//
//
// CHECK: @triton_reduce_max_dim_two_axis
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: ascend_dpx.shfl.bfly
// CHECK: scf.if [[COND:%[0-9]+]] -> (i8) {
// CHECK-NEXT:  llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  llvm.add
// CHECK-NEXT:  llvm.getelementptr [[ADDR:%[0-9]+]]
// CHECK-NEXT:  ascend_dpx.load
// CHECK-NEXT:  llvm.mlir.constant(128 : i32) : i32
// CHECK-NEXT:  llvm.add
// CHECK-NEXT:  llvm.getelementptr [[ADDR]]
// CHECK-NEXT:  ascend_dpx.load
// CHECK-NEXT:  llvm.intr.smax
// CHECK-NEXT:  llvm.mlir.constant(256 : i32) : i32
// CHECK-NEXT:  llvm.add
// CHECK-NEXT:  llvm.getelementptr [[ADDR]]
// CHECK-NEXT:  ascend_dpx.load
// CHECK-NEXT:  llvm.intr.smax
// CHECK-NEXT:  llvm.mlir.constant(384 : i32) : i32
// CHECK-NEXT:  llvm.add
// CHECK-NEXT:  llvm.getelementptr [[ADDR]]
// CHECK-NEXT:  ascend_dpx.load
// CHECK-NEXT:  llvm.intr.smax
// CHECK-NEXT:  scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT:  llvm.mlir.undef : i8
// CHECK-NEXT:  scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: ascend_dpx.sync_threads


#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 4, 2], order = [2, 0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 8], threadsPerWarp = [4, 1, 1, 1, 8], warpsPerCTA = [4, 2, 2, 1, 1], order = [4, 0, 1, 2, 3]}>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 1111 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_reduce_max_dim_two_axis(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<2x1x1xi32, #blocked>
    %cst_0 = arith.constant dense<64> : tensor<16x1x1x1x1xi32, #blocked1>
    %cst_1 = arith.constant dense<2> : tensor<16x1x1x1x1xi32, #blocked1>
    %cst_2 = arith.constant dense<64> : tensor<1x2x1x1x1xi32, #blocked1>
    %cst_3 = arith.constant dense<2> : tensor<1x2x1x1x1xi32, #blocked1>
    %cst_4 = arith.constant dense<64> : tensor<1x1x2x1x1xi32, #blocked1>
    %0 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<2x1x64x!tt.ptr<i8>, #blocked>
    %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<2x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<2x1x1xi32, #blocked>
    %4 = arith.muli %3, %cst : tensor<2x1x1xi32, #blocked>
    %5 = tt.broadcast %4 : tensor<2x1x1xi32, #blocked> -> tensor<2x1x64xi32, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x64xi32, #blocked>
    %9 = tt.broadcast %8 : tensor<1x1x64xi32, #blocked> -> tensor<2x1x64xi32, #blocked>
    %10 = arith.addi %5, %9 : tensor<2x1x64xi32, #blocked>
    %11 = tt.addptr %0, %10 : tensor<2x1x64x!tt.ptr<i8>, #blocked>, tensor<2x1x64xi32, #blocked>
    %12 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<16x2x2x1x64x!tt.ptr<i8>, #blocked1>
    %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>}>> -> tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>>
    %15 = tt.expand_dims %14 {axis = 2 : i32} : tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>> -> tensor<16x1x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>>
    %16 = tt.expand_dims %15 {axis = 3 : i32} : tensor<16x1x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>> -> tensor<16x1x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked1}>>
    %17 = tt.expand_dims %16 {axis = 4 : i32} : tensor<16x1x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked1}>> -> tensor<16x1x1x1x1xi32, #blocked1>
    %18 = arith.muli %17, %cst_0 : tensor<16x1x1x1x1xi32, #blocked1>
    %19 = arith.muli %18, %cst_1 : tensor<16x1x1x1x1xi32, #blocked1>
    %20 = arith.muli %19, %cst_1 : tensor<16x1x1x1x1xi32, #blocked1>
    %21 = tt.broadcast %20 : tensor<16x1x1x1x1xi32, #blocked1> -> tensor<16x2x1x1x1xi32, #blocked1>
    %22 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>}>>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>}>> -> tensor<1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>>
    %24 = tt.expand_dims %23 {axis = 2 : i32} : tensor<1x2xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>> -> tensor<1x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>>
    %25 = tt.expand_dims %24 {axis = 3 : i32} : tensor<1x2x1xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>> -> tensor<1x2x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked1}>>
    %26 = tt.expand_dims %25 {axis = 4 : i32} : tensor<1x2x1x1xi32, #ttg.slice<{dim = 4, parent = #blocked1}>> -> tensor<1x2x1x1x1xi32, #blocked1>
    %27 = arith.muli %26, %cst_2 : tensor<1x2x1x1x1xi32, #blocked1>
    %28 = arith.muli %27, %cst_3 : tensor<1x2x1x1x1xi32, #blocked1>
    %29 = tt.broadcast %28 : tensor<1x2x1x1x1xi32, #blocked1> -> tensor<16x2x1x1x1xi32, #blocked1>
    %30 = arith.addi %21, %29 : tensor<16x2x1x1x1xi32, #blocked1>
    %31 = tt.broadcast %30 : tensor<16x2x1x1x1xi32, #blocked1> -> tensor<16x2x2x1x1xi32, #blocked1>
    %32 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>}>> -> tensor<1x2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>>
    %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<1x2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>}>> -> tensor<1x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>>
    %35 = tt.expand_dims %34 {axis = 3 : i32} : tensor<1x1x2xi32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 4, parent = #blocked1}>}>> -> tensor<1x1x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked1}>>
    %36 = tt.expand_dims %35 {axis = 4 : i32} : tensor<1x1x2x1xi32, #ttg.slice<{dim = 4, parent = #blocked1}>> -> tensor<1x1x2x1x1xi32, #blocked1>
    %37 = arith.muli %36, %cst_4 : tensor<1x1x2x1x1xi32, #blocked1>
    %38 = tt.broadcast %37 : tensor<1x1x2x1x1xi32, #blocked1> -> tensor<16x2x2x1x1xi32, #blocked1>
    %39 = arith.addi %31, %38 : tensor<16x2x2x1x1xi32, #blocked1>
    %40 = tt.broadcast %39 : tensor<16x2x2x1x1xi32, #blocked1> -> tensor<16x2x2x1x64xi32, #blocked1>
    %41 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked1}>}>}>}>>
    %42 = tt.expand_dims %41 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked1}>}>}>}>> -> tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked1}>}>}>>
    %43 = tt.expand_dims %42 {axis = 1 : i32} : tensor<1x64xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked1}>}>}>> -> tensor<1x1x64xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked1}>}>>
    %44 = tt.expand_dims %43 {axis = 2 : i32} : tensor<1x1x64xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked1}>}>> -> tensor<1x1x1x64xi32, #ttg.slice<{dim = 3, parent = #blocked1}>>
    %45 = tt.expand_dims %44 {axis = 3 : i32} : tensor<1x1x1x64xi32, #ttg.slice<{dim = 3, parent = #blocked1}>> -> tensor<1x1x1x1x64xi32, #blocked1>
    %46 = tt.broadcast %45 : tensor<1x1x1x1x64xi32, #blocked1> -> tensor<16x2x2x1x64xi32, #blocked1>
    %47 = arith.addi %40, %46 : tensor<16x2x2x1x64xi32, #blocked1>
    %48 = tt.addptr %12, %47 : tensor<16x2x2x1x64x!tt.ptr<i8>, #blocked1>, tensor<16x2x2x1x64xi32, #blocked1>
    %49 = tt.load %48 : tensor<16x2x2x1x64x!tt.ptr<i8>, #blocked1>
    %50 = "tt.reduce"(%49) <{axis = 1 : i32}> ({
    ^bb0(%arg5: i8, %arg6: i8):
      %53 = arith.maxsi %arg5, %arg6 : i8
      tt.reduce.return %53 : i8
    }) {allocation.offset = 0 : i32} : (tensor<16x2x2x1x64xi8, #blocked1>) -> tensor<16x2x1x64xi8, #ttg.slice<{dim = 1, parent = #blocked1}>>
    gpu.barrier
    %51 = "tt.reduce"(%50) <{axis = 0 : i32}> ({
    ^bb0(%arg5: i8, %arg6: i8):
      %53 = arith.maxsi %arg5, %arg6 : i8
      tt.reduce.return %53 : i8
    }) {allocation.offset = 0 : i32} : (tensor<16x2x1x64xi8, #ttg.slice<{dim = 1, parent = #blocked1}>>) -> tensor<2x1x64xi8, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>>
    gpu.barrier
    %52 = ttg.convert_layout %51 {allocation.offset = 0 : i32, layout.action = "dontremove", layout.cost_estimate = -0.134374946 : f32, layout.priority = 90 : i32} : tensor<2x1x64xi8, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked1}>}>> -> tensor<2x1x64xi8, #blocked>
    tt.store %11, %52 : tensor<2x1x64x!tt.ptr<i8>, #blocked>
    tt.return
  }
}


