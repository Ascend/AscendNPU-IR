// RUN: bishengir-opt -split-input-file -optimize-layouts %s | FileCheck %s

// CHECK-LABEL: _fwd_kernel_stage1
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, "ttg.enable-bishengir-simt-optimization" = 11 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @_fwd_kernel_stage1(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.addptr %arg4, %0 : !tt.ptr<i32>, i32
    %6 = tt.load %5 : !tt.ptr<i32>
    %7 = tt.addptr %arg5, %6 : !tt.ptr<i64>, i32
    %8 = tt.splat %7 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.expand_dims %4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %10 = arith.extsi %9 : tensor<1x256xi32, #blocked> to tensor<1x256xi64, #blocked>
    %11 = tt.broadcast %10 : tensor<1x256xi64, #blocked> -> tensor<32x256xi64, #blocked>
    %12 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked1>
    %13 = tt.addptr %12, %3 : tensor<256x!tt.ptr<bf16>, #blocked1>, tensor<256xi32, #blocked1>
    %14 = tt.load %13 : tensor<256x!tt.ptr<bf16>, #blocked1>
    %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>, #blocked>
    // CHECK: ttg.convert_layout
    %17 = ttg.convert_layout %14 : tensor<256xbf16, #blocked1> -> tensor<256xbf16, #ttg.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<256xbf16, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xbf16, #blocked>
    %19 = tt.broadcast %18 : tensor<1x256xbf16, #blocked> -> tensor<32x256xbf16, #blocked>
    %20 = tt.splat %arg3 : f32 -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x256x!tt.ptr<bf16>, #blocked>
    %22 = arith.extf %19 : tensor<32x256xbf16, #blocked> to tensor<32x256xf32, #blocked>
    %23:3 = scf.for %arg21 = %2 to %2 step %c32_i32 iter_args(%arg22 = %cst, %arg23 = %cst_1, %arg24 = %cst_2) -> (tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, f32, f32)  : i32 {
      %30 = tt.addptr %8, %15 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %31 = tt.load %30 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi64, #blocked>
      %33 = tt.broadcast %32 : tensor<32x1xi64, #blocked> -> tensor<32x256xi64, #blocked>
      %34 = arith.addi %33, %11 : tensor<32x256xi64, #blocked>
      %35 = tt.addptr %16, %34 : tensor<32x256x!tt.ptr<bf16>, #blocked>, tensor<32x256xi64, #blocked>
      %36 = tt.load %35 : tensor<32x256x!tt.ptr<bf16>, #blocked>
      %37 = arith.extf %36 : tensor<32x256xbf16, #blocked> to tensor<32x256xf32, #blocked>
      %38 = arith.mulf %22, %37 : tensor<32x256xf32, #blocked>
      %39 = "tt.reduce"(%38) <{axis = 1 : i32}> ({
      ^bb0(%arg25: f32, %arg26: f32):
        %57 = arith.addf %arg25, %arg26 : f32
        tt.reduce.return %57 : f32
      }) : (tensor<32x256xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %40 = arith.mulf %39, %20 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %41 = tt.addptr %21, %34 : tensor<32x256x!tt.ptr<bf16>, #blocked>, tensor<32x256xi64, #blocked>
      %42 = tt.load %41 : tensor<32x256x!tt.ptr<bf16>, #blocked>
      %43 = "tt.reduce"(%40) <{axis = 0 : i32}> ({
      ^bb0(%arg25: f32, %arg26: f32):
        %57 = arith.maxnumf %arg25, %arg26 : f32
        tt.reduce.return %57 : f32
      }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32
      %44 = arith.maxnumf %43, %arg24 : f32
      %45 = arith.subf %arg24, %44 : f32
      %46 = math.exp %45 : f32
      %47 = tt.splat %44 : f32 -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %48 = arith.subf %40, %47 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %49 = math.exp %48 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %50 = tt.expand_dims %49 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xf32, #blocked>
      %51 = arith.extf %42 : tensor<32x256xbf16, #blocked> to tensor<32x256xf32, #blocked>
      %52 = tt.broadcast %50 : tensor<32x1xf32, #blocked> -> tensor<32x256xf32, #blocked>
      %53 = arith.mulf %52, %51 : tensor<32x256xf32, #blocked>
      %54 = "tt.reduce"(%53) <{axis = 0 : i32}> ({
      ^bb0(%arg25: f32, %arg26: f32):
        %57 = arith.addf %arg25, %arg26 : f32
        tt.reduce.return %57 : f32
      }) : (tensor<32x256xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %55 = "tt.reduce"(%49) <{axis = 0 : i32}> ({
      ^bb0(%arg25: f32, %arg26: f32):
        %57 = arith.addf %arg25, %arg26 : f32
        tt.reduce.return %57 : f32
      }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32
      %56 = arith.addf %55, %46 : f32
      scf.yield %54, %56, %44 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, f32, f32
    }
    %24 = tt.splat %arg6 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
    %25 = tt.addptr %24, %3 : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
    %26 = tt.splat %23#1 : f32 -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %27 = arith.divf %23#0, %26 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    // CHECK: ttg.convert_layout
    %28 = ttg.convert_layout %27 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<256xf32, #blocked1>
    tt.store %25, %28 : tensor<256x!tt.ptr<f32>, #blocked1>
    %29 = math.log %23#1 : f32
    tt.store %arg7, %29 : !tt.ptr<f32>
    tt.return
  }
}