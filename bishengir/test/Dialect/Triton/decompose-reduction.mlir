// RUN: bishengir-opt -split-input-file -decompose-reduction %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 16], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 4], order = [0, 1]}>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 16 : i32} {
  tt.func public @reduce_axis1_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked>
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 16], order = [1, 0]}>
    // CHECK-DAG: [[BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [16, 1], order = [1, 0]}>
    // CHECK-NOT: tt.reshape
    // CHECK: %[[C0:.+]] = ttg.convert_layout [[CST:%.*]] : tensor<128x32xf32, [[BLOCKED]]> -> tensor<128x32xf32, [[BLOCKED2]]>
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"(%[[C0]]) <{axis = 1 : i32}>  
    %13 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %18 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %18 : f32
    }) : (tensor<128x32xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = ttg.convert_layout %13 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 1, 16], warpsPerCTA = [2, 16, 1], order = [2, 0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [4, 8, 1], order = [2, 0, 1]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_unk_fused_embedding_eq_sum_where_zeros__8(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg7: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg8: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x16x32xf32, #blocked>
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [2, 1, 16], warpsPerCTA = [2, 16, 1], order = [2, 0, 1]}>
    // CHECK-DAG: [[BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 4, 1], threadsPerWarp = [4, 4, 2], warpsPerCTA = [1, 1, 32], order = [2, 0, 1]}>
    // CHECK-DAG: [[BLOCKED3:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [4, 8, 1], order = [2, 0, 1]}>
    // CHECK: %[[C0:.+]] = ttg.convert_layout [[CST:%.*]] : tensor<4x16x32xf32, [[BLOCKED]]> -> tensor<4x16x32xf32, [[BLOCKED2]]>
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"(%[[C0]]) <{axis = 1 : i32}>  
    %38 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
      %48 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<4x16x32xf32, #blocked>) -> tensor<4x32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %39 = ttg.convert_layout %38 : tensor<4x32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<4x32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<4x1x32xf32, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [16, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 1024 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @reduce_axis1_kernel_128x256(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<256> : tensor<128x1xi32, #blocked>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = arith.muli %4, %cst : tensor<128x1xi32, #blocked>
    %7 = tt.broadcast %1 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %8 = tt.broadcast %6 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %9 = arith.addi %7, %8 : tensor<128x256xi32, #blocked>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %9 : tensor<128x256x!tt.ptr<f32>, #blocked>, tensor<128x256xi32, #blocked>
    %12 = tt.load %11 : tensor<128x256x!tt.ptr<f32>, #blocked>
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [16, 2], order = [1, 0]}>
    // CHECK-DAG: [[BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 4, 8], warpsPerCTA = [16, 2, 1], order = [2, 1, 0]}>
    // CHECK-DAG: [[BLOCKED3:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
    // CHECK: [[RESHAPE2:%.*]] = tt.reshape [[RESHAPE:%.*]] : tensor<128x256xf32, [[BLOCKED]]> -> tensor<128x8x32xf32, [[BLOCKED2]]>
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"([[RESHAPE2]]) <{axis = 2 : i32}>  
    // CHECK: %[[REDUCE2:.+]] = ttg.convert_layout [[REDUCE:%.*]] :  tensor<128x8xf32, #ttg.slice<{dim = 2, parent = [[BLOCKED2]]}>> -> tensor<128x8xf32, [[BLOCKED3]]>
    // CHECK-NEXT: %[[REDUCE3:.*]] = "tt.reduce"(%[[REDUCE2]]) <{axis = 1 : i32}>  
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %18 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %18 : f32
    }) {allocation.offset = 0 : i32} : (tensor<128x256xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = ttg.convert_layout %13 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %16 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %17 = tt.addptr %16, %5 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    tt.store %17, %15 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 16], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 8192 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @reduce_axis1_kernel_128x2048(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<2048> : tensor<128x1xi32, #blocked>
    %0 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2048xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x2048xi32, #blocked>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = arith.muli %4, %cst : tensor<128x1xi32, #blocked>
    %7 = tt.broadcast %1 : tensor<1x2048xi32, #blocked> -> tensor<128x2048xi32, #blocked>
    %8 = tt.broadcast %6 : tensor<128x1xi32, #blocked> -> tensor<128x2048xi32, #blocked>
    %9 = arith.addi %7, %8 : tensor<128x2048xi32, #blocked>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x2048x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %9 : tensor<128x2048x!tt.ptr<f32>, #blocked>, tensor<128x2048xi32, #blocked>
    %12 = tt.load %11 : tensor<128x2048x!tt.ptr<f32>, #blocked>
    // CHECK: tt.reshape 
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>  
    // CHECK: ttg.convert_layout
    // CHECK: tt.reshape 
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>  
    // CHECK: ttg.convert_layout
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 1 : i32}>  
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %18 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %18 : f32
    }) {allocation.offset = 0 : i32} : (tensor<128x2048xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = ttg.convert_layout %13 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %16 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %17 = tt.addptr %16, %5 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    tt.store %17, %15 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 32], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @reduce_axis1_kernel_128x32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<32> : tensor<128x1xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = arith.muli %4, %cst : tensor<128x1xi32, #blocked>
    %7 = tt.broadcast %1 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %8 = tt.broadcast %6 : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %9 = arith.addi %7, %8 : tensor<128x32xi32, #blocked>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %9 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %12 = tt.load %11 : tensor<128x32x!tt.ptr<f32>, #blocked>
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 32], order = [1, 0]}>
    // CHECK-DAG: [[BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
    // CHECK-NOT: tt.reshape
    // CHECK: %[[C0:.+]] = ttg.convert_layout [[CST:%.*]] : tensor<128x32xf32, [[BLOCKED]]> -> tensor<128x32xf32, [[BLOCKED2]]>
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"(%[[C0]]) <{axis = 1 : i32}>  
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %18 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %18 : f32
    }) : (tensor<128x32xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = ttg.convert_layout %13 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %16 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %17 = tt.addptr %16, %5 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    tt.store %17, %15 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}


// -----
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 32], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @reduce_axis1_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<32> : tensor<128x1xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %6 = arith.muli %4, %cst : tensor<128x1xi32, #blocked>
    %7 = tt.broadcast %1 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %8 = tt.broadcast %6 : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %9 = arith.addi %7, %8 : tensor<128x32xi32, #blocked>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %9 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %12 = tt.load %11 : tensor<128x32x!tt.ptr<f32>, #blocked>
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 32], order = [1, 0]}>
    // CHECK-DAG: [[BLOCKED2:#.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
    // CHECK: ttg.convert_layout [[CST:%.*]] : tensor<128x32xf32, [[BLOCKED]]> -> tensor<128x32xf32, [[BLOCKED2]]>
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %26 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %26 : f32
    }) : (tensor<128x32xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = ttg.convert_layout %13 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %17 = tt.addptr %16, %5 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    tt.store %17, %15 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    %18 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %19 = tt.addptr %18, %9 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %20 = tt.load %19 : tensor<128x32x!tt.ptr<f32>, #blocked>
    // CHECK: %[[C0:.+]] = ttg.convert_layout [[CST:%.*]] : tensor<128x32xf32, [[BLOCKED]]> -> tensor<128x32xf32, [[BLOCKED2]]>
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"(%[[C0]]) <{axis = 1 : i32}>  
    %21 = "tt.reduce"(%20) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %26 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %26 : f32
    }) : (tensor<128x32xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = ttg.convert_layout %21 {allocation.offset = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %24 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %25 = tt.addptr %24, %5 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    tt.store %25, %23 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
