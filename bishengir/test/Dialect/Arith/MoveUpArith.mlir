// RUN: bishengir-opt -move-up-arith -split-input-file %s | FileCheck %s

func.func private @test_arith_moveup_in_nested_scf_for(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<64x128xf32>, %arg4: tensor<8x64x16xf16>) -> (tensor<64x128xf32>, tensor<8x64x16xf16>) {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 0 : index
  %2:2 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<64x128xf32>, tensor<8x64x16xf16>)  : i32 {
    %3 = arith.index_cast %arg5 : i32 to index
    %4:2 = scf.for %arg8 = %arg0 to %arg1 step %arg2 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<64x128xf32>, tensor<8x64x16xf16>)  : i32 {
      %5 = arith.muli %arg8, %arg1 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7 = tensor.extract_slice %arg3[%3, %6] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
      %8 = arith.subf %7, %7 : tensor<1x64xf32>
      %9 = tensor.insert_slice %8 into %arg9[%3, %6] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
      %10 = arith.muli %arg8, %arg0 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = tensor.extract_slice %arg4[%11, %3, 0] [4, 1, 16] [1, 1, 1] : tensor<8x64x16xf16> to tensor<4x1x16xf16>
      %13 = tensor.insert_slice %12 into %arg10[%11, %3, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<8x64x16xf16>
      scf.yield %9, %13 : tensor<64x128xf32>, tensor<8x64x16xf16>
    }
    scf.yield %4#0, %4#1 : tensor<64x128xf32>, tensor<8x64x16xf16>
  }
  return %2#0, %2#1 : tensor<64x128xf32>, tensor<8x64x16xf16>
}

// CHECK-LABEL: func private @test_arith_moveup_in_nested_scf_for
// CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<64x128xf32>, tensor<8x64x16xf16>) : i32 {
// CHECK-NEXT: {{.*}} = arith.index_cast {{.*}} : i32 to index
// CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<64x128xf32>, tensor<8x64x16xf16>) : i32 {
// CHECK-NEXT: {{.*}} = arith.muli {{.*}} : i32
// CHECK-NEXT: {{.*}} = arith.index_cast {{.*}} : i32 to index
// CHECK-NEXT: {{.*}} = arith.muli {{.*}} : i32
// CHECK-NEXT: {{.*}} = arith.index_cast {{.*}} : i32 to index
// CHECK: {{.*}} = arith.subf {{.*}} : tensor<1x64xf32>


// -----

func.func @test_arith_moveup_basic_non_arith_prev(%arg0: tensor<64xf32>, %arg1: i32, %arg2: i32) -> (i32, f32) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %0 = tensor.extract_slice %arg0[%c0] [32] [1] : tensor<64xf32> to tensor<32xf32>
  %1 = tensor.extract %0[%c0] : tensor<32xf32>
  %2 = arith.muli %arg1, %arg2 : i32
  %3 = arith.addi %2, %arg1 : i32
  %4 = arith.subf %1, %cst : f32
  return %3, %4 : i32, f32
}

// CHECK-LABEL: func.func @test_arith_moveup_basic_non_arith_prev
// CHECK: {{.*}} = arith.muli {{.*}} : i32
// CHECK-NEXT: {{.*}} = arith.addi {{.*}} : i32
// CHECK: tensor.extract_slice {{.*}} : tensor<64xf32> to tensor<32xf32>
// CHECK: tensor.extract {{.*}} : tensor<32xf32>
// CHECK: {{.*}} = arith.subf {{.*}} : f32

// -----

func.func @test_arith_moveup_follow_defining_op(%arg0: f32, %arg1: tensor<f32>, %arg2: f32) -> (f32) {
  %c1 = arith.constant 1.0 : f32
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = arith.addf %arg0, %c1 : f32
  %2 = arith.subf %1, %arg2 : f32
  %3 = arith.divf %2, %0 : f32
  return %3 : f32
}

// CHECK-LABEL: func.func @test_arith_moveup_follow_defining_op
// CHECK: tensor.extract {{.*}} : tensor<f32>
// CHECK: {{.*}} = arith.addf {{.*}} : f32
// CHECK-NEXT: {{.*}} = arith.subf {{.*}} : f32
// CHECK-NEXT: {{.*}} = arith.divf {{.*}} : f32

// -----

func.func @test_arith_moveup_scf_for_nested(%arg0: i32, %arg1: i32, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %1:1 = scf.for %arg3 = %arg0 to %arg1 step %c1 iter_args(%arg4 = %init) -> (tensor<64x128xf32>) : i32 {
    %2 = arith.index_cast %arg3 : i32 to index
    %4 = tensor.extract_slice %arg2[%2, %c0] [1,64] [1,1] : tensor<64x128xf32> to tensor<1x64xf32>
    %5 = arith.muli %arg3, %arg3 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = tensor.insert_slice %4 into %arg4[%6, %c0] [1,64] [1,1] : tensor<1x64xf32> into tensor<64x128xf32>
    scf.yield %7 : tensor<64x128xf32>
  }
  return %1#0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_arith_moveup_scf_for_nested
// CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<64x128xf32>) : i32 {
// CHECK-NEXT: {{.*}} = arith.muli {{.*}} : i32
// CHECK-NEXT: {{.*}} = arith.index_cast {{.*}} : i32 to index
// CHECK-NEXT: {{.*}} = arith.index_cast {{.*}} : i32 to index
// CHECK: tensor.extract_slice {{.*}} : tensor<64x128xf32> to tensor<1x64xf32>
// CHECK: tensor.insert_slice {{.*}} : tensor<1x64xf32> into tensor<64x128xf32>
// CHECK: scf.yield {{.*}} : tensor<64x128xf32>

// -----

func.func @test_arith_moveup_comprehensive_all_arith_op(%arg0: i32, %arg1: f32, %arg2: f16) -> (i32, f32, f16) {
  %c1_i32 = arith.constant 1 : i32
  %c2_f32 = arith.constant 2.0 : f32
  %1 = arith.addi %arg0, %c1_i32 : i32
  %2 = arith.mulf %arg1, %c2_f32 : f32
  %3 = arith.cmpi sgt, %1, %c1_i32 : i32
  %4 = arith.select %3, %1, %c1_i32 : i32
  %5 = arith.addf %2, %arg1 : f32
  %6 = arith.truncf %5 : f32 to f16
  %7 = arith.addf %6, %arg2 : f16
  return %4, %2, %7 : i32, f32, f16
}

// CHECK-LABEL: func.func @test_arith_moveup_comprehensive_all_arith_op
// CHECK: {{.*}} = arith.addi {{.*}} : i32
// CHECK: {{.*}} = arith.mulf {{.*}} : f32
// CHECK: {{.*}} = arith.cmpi sgt, {{.*}} : i32
// CHECK: {{.*}} = arith.select {{.*}} : i32
// CHECK: {{.*}} = arith.addf {{.*}} : f32
// CHECK: {{.*}} = arith.truncf {{.*}} : f32 to f16
// CHECK: {{.*}} = arith.addf {{.*}} : f16
// CHECK: return {{.*}} : i32, f32, f16