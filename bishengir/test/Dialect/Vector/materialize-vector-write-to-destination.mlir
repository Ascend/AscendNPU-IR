// RUN: bishengir-opt %s -materialize-vector-write-to-destination -cse -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_recursive_insert_slice_0
// CHECK: %[[slice0:.*]] = tensor.extract_slice %[[arg6:.*]][0, {{.*}}]
// CHECK: %[[write0:.*]] = vector.transfer_write {{.*}}, %[[slice0]]
// CHECK: %[[insert0:.*]] = tensor.insert_slice %[[write0]] into %[[arg6]]
// CHECK: %[[slice1:.*]] = tensor.extract_slice %inserted_slice
// CHECK: %[[write1:.*]] = vector.transfer_write {{.*}}, %[[slice1]]
// CHECK: %[[slice2:.*]] = tensor.extract_slice
// CHECK: %[[write2:.*]] = vector.transfer_write {{.*}}, %[[slice2]]
module {
  func.func @test_recursive_insert_slice_0(%arg0: tensor<8x64x16xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128xf32>, %arg4: index) -> (tensor<8x64x16xf16>, tensor<64xf32>) 
  attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<4x1x16xf16>
    %1 = tensor.empty() : tensor<1xf32>
    %2:2 = scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg6 = %arg0, %arg7 = %arg1) -> (tensor<8x64x16xf16>, tensor<64xf32>)  : i32 {
      %3 = arith.index_cast %arg5 : i32 to index
      %extracted_slice = tensor.extract_slice %arg2[%3] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%3, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%3, 64] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
      %extracted = tensor.extract %extracted_slice[%arg4] : tensor<1xf32>
      %4 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = vector.broadcast %extracted : f32 to vector<1x64xf32>
      %6 = arith.subf %4, %5 : vector<1x64xf32>
      %7 = math.exp %6 : vector<1x64xf32>
      %8 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %9 = arith.subf %8, %5 : vector<1x64xf32>
      %10 = math.exp %9 : vector<1x64xf32>
      %11 = vector.shape_cast %7 : vector<1x64xf32> to vector<4x1x16xf32>
      %12 = arith.truncf %11 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<4x1x16xf32> to vector<4x1x16xf16>
      %13 = vector.transfer_write %12, %0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, tensor<4x1x16xf16>
      %inserted_slice = tensor.insert_slice %13 into %arg6[0, %3, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<8x64x16xf16>
      %14 = vector.shape_cast %10 : vector<1x64xf32> to vector<4x1x16xf32>
      %15 = arith.truncf %14 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<4x1x16xf32> to vector<4x1x16xf16>
      %16 = vector.transfer_write %15, %0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, tensor<4x1x16xf16>
      %inserted_slice_2 = tensor.insert_slice %16 into %inserted_slice[4, %3, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<8x64x16xf16>
      %17 = vector.transfer_read %1[%c0], %cst {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %18 = arith.addf %7, %10 : vector<1x64xf32>
      %19 = vector.multi_reduction <add>, %18, %17 [1] : vector<1x64xf32> to vector<1xf32>
      %20 = vector.transfer_write %19, %1[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice_3 = tensor.insert_slice %20 into %arg7[%3] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice_2, %inserted_slice_3 : tensor<8x64x16xf16>, tensor<64xf32>
    }
    return %2#0, %2#1 : tensor<8x64x16xf16>, tensor<64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_recursive_insert_slice_1
// CHECK: %[[index:.*]] = arith.index_cast
// CHECK: %[[slice0:.*]] = tensor.extract_slice {{.*}}[%[[index]], 0] [1, 64]
// CHECK: %[[write0:.*]] = vector.transfer_write {{.*}}, %[[slice0]]
// CHECK: %[[insert0:.*]] = tensor.insert_slice %[[write0]]
// CHECK: %[[slice1:.*]] = tensor.extract_slice %[[insert0:.*]][%[[index]], 64] [1, 64]
// CHECK: %[[write1:.*]] = vector.transfer_write {{.*}}, %[[slice1]]
// CHECK: tensor.insert_slice %[[write1]] into %[[insert0]][%[[index]], 64] [1, 64] 
func.func @test_recursive_insert_slice_1(%arg0: tensor<64x128xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x128xf32>, %arg3: vector<1x64xf32>, %arg4: tensor<1x64xf32>) -> tensor<64x128xf32> 
attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg6 = %arg0) -> (tensor<64x128xf32>)  : i32 {
    %1 = arith.index_cast %arg5 : i32 to index
    %extracted_slice = tensor.extract_slice %arg2[%1, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%1, 64] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
    %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %3 = arith.mulf %2, %arg3 : vector<1x64xf32>
    %4 = vector.transfer_write %3, %arg4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    %inserted_slice = tensor.insert_slice %4 into %arg6[%1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
    %5 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %6 = arith.mulf %5, %arg3 : vector<1x64xf32>
    %7 = vector.transfer_write %6, %arg4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    %inserted_slice_1 = tensor.insert_slice %7 into %inserted_slice[%1, 64] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
    scf.yield %inserted_slice_1 : tensor<64x128xf32>
  }
  return %0 : tensor<64x128xf32>
}

// -----

// Test if transfer_write folds well with insert_slice when there are reshapes/arith_ops in between
// CHECK-LABEL: func.func @test_dominates_0
// CHECK: %[[expanded:.*]] = tensor.expand_shape
// CHECK: %[[write:.*]] = vector.transfer_write
// CHECK-NEXT: tensor.insert_slice %[[write]] into %[[expanded]]
func.func @test_dominates_0(%arg0: tensor<64x128xf32>, %arg1: tensor<8192xf32>, %arg2: tensor<64x128xf32>, %arg3: vector<1x64xf32>, %arg4: tensor<1x64xf32>) -> tensor<64x128xf32> 
attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg6 = %arg0) -> (tensor<64x128xf32>)  : i32 {
    %1 = arith.index_cast %arg5 : i32 to index
    %extracted_slice = tensor.extract_slice %arg2[%1, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%1, 64] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
    %2 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %3 = arith.mulf %2, %arg3 : vector<1x64xf32>
    %4 = vector.transfer_write %3, %arg4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    %5 = arith.index_cast %arg5 : i32 to index
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [64, 128] : tensor<8192xf32> into tensor<64x128xf32>
    %inserted_slice = tensor.insert_slice %4 into %expanded[%5, 64] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x128xf32>
    scf.yield %inserted_slice : tensor<64x128xf32>
  }
  return %0 : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: func @redirect_write_dst_to_iter_arg_0(
// CHECK: scf.for {{.*}} iter_args(%[[IA:.*]] = {{.*}})
// CHECK:   %[[VEC:.*]] = vector.transfer_read
// CHECK:   %[[ADD:.*]] = arith.addf %[[VEC]], %[[VEC]]
// CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[ADD]], %[[IA]]
// CHECK:   scf.yield %[[WRITE]]
func.func @redirect_write_dst_to_iter_arg_0(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> 
attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %f0 = arith.constant 0.0 : f32

  %res = scf.for %iv = %c0 to %c100 step %c1 iter_args(%ia = %arg0) -> (tensor<1x64xf32>) {
    %empty = tensor.empty() : tensor<1x64xf32>
    %v = vector.transfer_read %ia[%c0, %c0], %f0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %upd = arith.addf %v, %v : vector<1x64xf32>
    %next = vector.transfer_write %upd, %empty[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    scf.yield %next : tensor<1x64xf32>
  }
  return %res : tensor<1x64xf32>
}

// -----

// support function arg destination may cause problem: issue !1004
// TODO: remove it after support clone-tensor-empty before outline-scope

// CHECK-LABEL: func @redirect_write_dst_to_iter_arg_1(
// CHECK: scf.for {{.*}} iter_args(%[[IA:.*]] = {{.*}})
// CHECK:   %[[VEC:.*]] = vector.transfer_read
// CHECK:   %[[ADD:.*]] = arith.addf %[[VEC]], %[[VEC]]
// CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[ADD]], %[[IA]]
// CHECK:   scf.yield %[[WRITE]]
func.func @redirect_write_dst_to_iter_arg_1(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32> 
attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %f0 = arith.constant 0.0 : f32

  %res = scf.for %iv = %c0 to %c100 step %c1 iter_args(%ia = %arg0) -> (tensor<1x64xf32>) {
    %v = vector.transfer_read %ia[%c0, %c0], %f0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %upd = arith.addf %v, %v : vector<1x64xf32>
    %next = vector.transfer_write %upd, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    scf.yield %next : tensor<1x64xf32>
  }
  return %res : tensor<1x64xf32>
}

// -----

// CHECK-LABEL: func @redirect_write_dst_to_iter_arg_fail_0
// CHECK: %[[CONST:.*]] = arith.constant
// CHECK: vector.transfer_write {{.*}}, %[[CONST]]
func.func @redirect_write_dst_to_iter_arg_fail_0(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> 
attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.0 : f32
  %cst_tensor = arith.constant dense<0.0> : tensor<1x64xf32>

  %res = scf.for %iv = %c0 to %c100 step %c1 iter_args(%ia = %arg0) -> (tensor<1x64xf32>) {
    %v = vector.broadcast %cst : f32 to vector<1x64xf32>
    // Destination is a constant tensor, not tensor.empty, should not match
    %next = vector.transfer_write %v, %cst_tensor[%c0, %c0] : vector<1x64xf32>, tensor<1x64xf32>
    scf.yield %next : tensor<1x64xf32>
  }
  return %res : tensor<1x64xf32>
}

// -----

// test should fail for func not marked as manual vf
// CHECK-LABEL: func @redirect_write_dst_to_iter_arg_fail_1(
// CHECK-SAME: %[[arg0:.*]]: tensor<1x64xf32>, %[[arg1:.*]]: tensor<1x64xf32>)
// CHECK: vector.transfer_write {{.*}}, %[[arg1]]
func.func @redirect_write_dst_to_iter_arg_fail_1(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32> 
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %f0 = arith.constant 0.0 : f32

  %res = scf.for %iv = %c0 to %c100 step %c1 iter_args(%ia = %arg0) -> (tensor<1x64xf32>) {
    %v = vector.transfer_read %ia[%c0, %c0], %f0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %upd = arith.addf %v, %v : vector<1x64xf32>
    %next = vector.transfer_write %upd, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    scf.yield %next : tensor<1x64xf32>
  }
  return %res : tensor<1x64xf32>
}
