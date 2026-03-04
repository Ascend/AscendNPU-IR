// RUN: bishengir-opt -split-input-file -allocate-shared-memory %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 16], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 8], order = [0, 1]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, ttg.global_scratch_memory_alignment = 1 : i32,
    ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32,
    ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32,
    "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @reduce_axis1_kernel_128x2048_with_no_ttgshared(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
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
    // CHECK: "tt.reduce"({{%.*}}) <{axis = 1 : i32}>
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {allocation.offset = 0 : i32}
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %18 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %18 : f32
    }) : (tensor<128x2048xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: ttg.convert_layout {{%.*}} {allocation.offset = 0 : i32}
    %14 = ttg.convert_layout %13 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
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
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, ttg.global_scratch_memory_alignment = 1 : i32,
    ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32,
    ttg.shared = 8192 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32,
    "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
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
    // CHECK: "tt.reduce"({{%.*}}) <{axis = 1 : i32}>
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {allocation.offset = 0 : i32}
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %18 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %18 : f32
    }) : (tensor<128x2048xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: ttg.convert_layout {{%.*}} {allocation.offset = 0 : i32}
    %14 = ttg.convert_layout %13 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
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
module attributes {"ttg.enable-bishengir-simt-optimization" = 1 : i32, ttg.global_scratch_memory_alignment = 1 : i32,
    ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32,
    ttg.shared = 1048576 : i32, ttg.target = "cuda:89", ttg.tensor_memory_size = 0 : i32,
    "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 32 : i32} {
  tt.func public @reduce_add_reduce_axis1_kernel_128x2048_need_larger_ttgshared(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
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
    // CHECK: "tt.reduce"({{%.*}})
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {allocation.offset = 0 : i32}
    %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %24 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %24 : f32
    }) : (tensor<128x2048xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: ttg.convert_layout {{%.*}} {allocation.offset = 0 : i32}
    %14 = ttg.convert_layout %13 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %16 = tt.broadcast %15 :  tensor<128x1xf32, #blocked1> ->  tensor<128x2048xf32, #blocked1>
    %17 = ttg.convert_layout %16 : tensor<128x2048xf32, #blocked1> -> tensor<128x2048xf32, #blocked>
    %18 = arith.addf %12, %17 : tensor<128x2048xf32, #blocked>
    // CHECK: "tt.reduce"({{%.*}})
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {{%.*}}
    // CHECK-NEXT: {allocation.offset = 0 : i32}
    %19 = "tt.reduce"(%18) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %25 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %25 : f32
    }) : (tensor<128x2048xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: ttg.convert_layout {{%.*}} {allocation.offset = 0 : i32}
    %20 = ttg.convert_layout %19 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
    %22 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %23 = tt.addptr %22, %5 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    tt.store %23, %21 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}