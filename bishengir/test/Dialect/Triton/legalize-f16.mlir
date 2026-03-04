// RUN: bishengir-opt %s -split-input-file -legalize-f16-for-triton -cse -sccp -canonicalize -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// -----
// CHECK-LABEL: @triton_gelu_tanh_dtype2_shape2
module {
  tt.func public @triton_gelu_tanh_dtype2_shape2(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg7: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<4096xbf16>
    %cst_0 = arith.constant dense<7.968750e-01> : tensor<4096xbf16>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.get_num_programs x : i32
    %3 = arith.extsi %2 : i32 to i64
    %4 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %5 = arith.extsi %4 : tensor<4096xi32> to tensor<4096xi64>
    %6 = arith.extsi %arg2 : i32 to i64
    %7 = tt.splat %6 : i64 -> tensor<4096xi64>
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>>
    %9 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>>
    scf.for %arg8 = %c0_i32 to %arg4 step %c1_i32  : i32 {
      %10 = arith.extsi %arg8 : i32 to i64
      %11 = arith.muli %10, %3 : i64
      %12 = arith.addi %1, %11 : i64
      %13 = arith.muli %12, %c4096_i64 : i64
      %14 = tt.splat %13 : i64 -> tensor<4096xi64>
      %15 = arith.addi %14, %5 : tensor<4096xi64>
      %16 = arith.cmpi slt, %15, %7 : tensor<4096xi64>
      %17 = tt.addptr %8, %15 : tensor<4096x!tt.ptr<bf16>>, tensor<4096xi64>
      %18 = tt.load %17, %16, %cst : tensor<4096x!tt.ptr<bf16>>
      // CHECK-NOT: arith.mulf {{.*}} : tensor<4096xbf16>
      // CHECK: %[[EXT:.*]] = arith.extf {{.*}} : tensor<4096xbf16> to tensor<4096xf32>
      // CHECK: arith.mulf %[[EXT]], {{.*}} : tensor<4096xf32>
      %19 = arith.mulf %18, %cst_0 : tensor<4096xbf16>
      %20 = arith.extf %19 : tensor<4096xbf16> to tensor<4096xf32>
      %21 = arith.mulf %19, %cst_0 : tensor<4096xbf16>
      %22 = arith.truncf %20 : tensor<4096xf32> to tensor<4096xbf16>
      %23 = tt.addptr %9, %15 : tensor<4096x!tt.ptr<bf16>>, tensor<4096xi64>
      tt.store %23, %22, %16 : tensor<4096x!tt.ptr<bf16>>
      tt.store %23, %21, %16 : tensor<4096x!tt.ptr<bf16>>
    }
    tt.return
  }
}
// -----

// CHECK-LABEL: @triton_eq_2
module {
  tt.func public @triton_eq_2(%arg0: bf16, %arg1: bf16) -> i1{
    // CHECK-NOT: arith.cmpf oeq, %{{.*}}, %{{.*}} : bf16
    // CHECK: arith.cmpf oeq, %{{.*}}, %{{.*}} : f32
    %0 = arith.cmpf oeq, %arg0, %arg0 : bf16
    tt.return %0 : i1
  }
}
// -----
module {
  // CHECK-LABEL: @triton_sum_3D_dim0
  tt.func public @triton_sum_3D_dim0(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<16x1x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<1x64xi32> -> tensor<1x1x64xi32>
    %4 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<16x1xi32> -> tensor<16x1x1xi32>
    %6 = arith.muli %5, %cst : tensor<16x1x1xi32>
    %7 = tt.broadcast %3 : tensor<1x1x64xi32> -> tensor<16x1x64xi32>
    %8 = tt.broadcast %6 : tensor<16x1x1xi32> -> tensor<16x1x64xi32>
    %9 = arith.addi %7, %8 : tensor<16x1x64xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x1x64x!tt.ptr<f16>>
    %11 = tt.addptr %10, %9 : tensor<16x1x64x!tt.ptr<f16>>, tensor<16x1x64xi32>
    
    // CHECK: %[[LOAD:.*]] = tt.load %[[PTR:.*]] : tensor<16x1x64x!tt.ptr<f16>>
    %12 = tt.load %11 : tensor<16x1x64x!tt.ptr<f16>>
 
    // CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[LOAD]] : tensor<16x1x64xf16> to tensor<16x1x64xf32>
    
    // CHECK-NEXT: %[[REDUCE:.*]] = "tt.reduce"(%[[EXT]]) <{axis = 0 : i32}> ({
    // CHECK-NEXT: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
    // CHECK-NEXT:   %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
    // CHECK-NEXT:   tt.reduce.return %[[ADD]] : f32
    // CHECK-NEXT: }) : (tensor<16x1x64xf32>) -> tensor<1x64xf32>
    %13 = "tt.reduce"(%12) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f16, %arg3: f16):
      %16 = arith.addf %arg2, %arg3 : f16
      tt.reduce.return %16 : f16
    }) : (tensor<16x1x64xf16>) -> tensor<1x64xf16>
 
    // CHECK-NEXT: %[[TRUNC:.*]] = arith.truncf %[[REDUCE]] : tensor<1x64xf32> to tensor<1x64xf16>
    
    %14 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>>
    %15 = tt.addptr %14, %2 : tensor<1x64x!tt.ptr<f16>>, tensor<1x64xi32>
    
    // CHECK: tt.store %[[DST:.*]], %[[TRUNC]] : tensor<1x64x!tt.ptr<f16>>
    tt.store %15, %13 : tensor<1x64x!tt.ptr<f16>>
    tt.return
  }
}

// -----
module {

  // CHECK-LABEL: @triton_sum_3D_dim0_dim1
  tt.func public @triton_sum_3D_dim0_dim1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {gpu.block = #gpu.block<x>}, %arg3: i32 {gpu.block = #gpu.block<y>}, %arg4: i32 {gpu.block = #gpu.block<z>}) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<4x1x1xi32>
    %cst_0 = arith.constant dense<16> : tensor<4x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<1x16x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %6 = tt.expand_dims %5 {axis = 2 : i32} : tensor<1x16xi32> -> tensor<1x16x1xi32>
    %7 = arith.muli %6, %cst_1 : tensor<1x16x1xi32>
    %8 = tt.broadcast %4 : tensor<1x1x8xi32> -> tensor<1x16x8xi32>
    %9 = tt.broadcast %7 : tensor<1x16x1xi32> -> tensor<1x16x8xi32>
    %10 = arith.addi %8, %9 : tensor<1x16x8xi32>
    %11 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %13 = arith.muli %12, %cst_0 : tensor<4x1x1xi32>
    %14 = arith.muli %13, %cst : tensor<4x1x1xi32>
    %15 = tt.broadcast %10 : tensor<1x16x8xi32> -> tensor<4x16x8xi32>
    %16 = tt.broadcast %14 : tensor<4x1x1xi32> -> tensor<4x16x8xi32>
    %17 = arith.addi %15, %16 : tensor<4x16x8xi32>
    %18 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x16x8x!tt.ptr<f16>>
    %19 = tt.addptr %18, %17 : tensor<4x16x8x!tt.ptr<f16>>, tensor<4x16x8xi32>

    // CHECK: %[[LOAD:.*]] = tt.load %[[PTR:.*]] : tensor<4x16x8x!tt.ptr<f16>>
    %20 = tt.load %19 : tensor<4x16x8x!tt.ptr<f16>>

    // CHECK-NEXT: %[[EXT:.*]] = arith.extf %[[LOAD]] : tensor<4x16x8xf16> to tensor<4x16x8xf32>

    // CHECK-NEXT: %[[RED1:.*]] = "tt.reduce"(%[[EXT]]) <{axis = 2 : i32}> ({
    // CHECK-NEXT: ^bb0(%[[ARG1_L0:.*]]: f32, %[[ARG2_L0:.*]]: f32):
    // CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[ARG1_L0]], %[[ARG2_L0]] : f32
    // CHECK-NEXT:   tt.reduce.return %[[ADD1]] : f32
    // CHECK-NEXT: }) : (tensor<4x16x8xf32>) -> tensor<4x16xf32>
    %21 = "tt.reduce"(%20) <{axis = 2 : i32}> ({
    ^bb0(%arg5: f16, %arg6: f16):
      %25 = arith.addf %arg5, %arg6 : f16
      tt.reduce.return %25 : f16
    }) : (tensor<4x16x8xf16>) -> tensor<4x16xf16>

    // CHECK-NEXT: %[[RED2:.*]] = "tt.reduce"(%[[RED1]]) <{axis = 0 : i32}> ({
    // CHECK-NEXT: ^bb0(%[[ARG1_L1:.*]]: f32, %[[ARG2_L1:.*]]: f32):
    // CHECK-NEXT:   %[[ADD2:.*]] = arith.addf %[[ARG1_L1]], %[[ARG2_L1]] : f32
    // CHECK-NEXT:   tt.reduce.return %[[ADD2]] : f32
    // CHECK-NEXT: }) : (tensor<4x16xf32>) -> tensor<16xf32>
    %22 = "tt.reduce"(%21) <{axis = 0 : i32}> ({
    ^bb0(%arg5: f16, %arg6: f16):
      %25 = arith.addf %arg5, %arg6 : f16
      tt.reduce.return %25 : f16
    }) : (tensor<4x16xf16>) -> tensor<16xf16>

    // CHECK-NEXT: %[[TRUNC:.*]] = arith.truncf %[[RED2]] : tensor<16xf32> to tensor<16xf16>
    %23 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %24 = tt.addptr %23, %1 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>

    // CHECK: tt.store %[[DST:.*]], %[[TRUNC]] : tensor<16x!tt.ptr<f16>>
    tt.store %24, %22 : tensor<16x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-NOT: "tt.reduce"{{.*}}f16