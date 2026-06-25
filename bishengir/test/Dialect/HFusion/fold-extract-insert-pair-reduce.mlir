// RUN: bishengir-opt %s -hfusion-fold-extract-insert-pair | FileCheck %s

// Test case: Fold redundant extract_slice -> transfer_write -> insert_slice pattern
// in a reduction loop where the insert_slice is unnecessary.
//
// Pattern:
//   %slice = tensor.extract_slice %src[offsets] [sizes] [strides]
//   %written = vector.transfer_write %vec, %slice[0, 0] {mask}
//   %result = tensor.insert_slice %written into %slice[0, 0] [sizes] [1, 1]
//
// Should be folded to:
//   %result = %written

// -----

// CHECK-LABEL: @test_fold_extract_slice_insert_slice_in_loop
// CHECK-NOT: tensor.insert_slice {{.*}} into %extracted_slice
// CHECK: %inserted_slice = tensor.insert_slice {{.*}} into %arg4
func.func @test_fold_extract_slice_insert_slice_in_loop(%arg0: tensor<1x8777xf16>, %arg1: tensor<1xf32>, %arg2: tensor<1x8777xf32>) -> (tensor<1xf32>, tensor<1x8777xf32>) {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c8777 = arith.constant 8777 : index
  %c0 = arith.constant 0 : index
  %0:2 = scf.for %arg3 = %c0 to %c8777 step %c64 iter_args(%arg4 = %arg2, %arg5 = %cst_0) -> (tensor<1x8777xf32>, vector<1x64xf32>) {
    %4 = affine.min affine_map<(d0) -> (-d0 + 8777, 64)>(%arg3)
    %extracted_slice = tensor.extract_slice %arg4[0, %arg3] [1, %4] [1, 1] : tensor<1x8777xf32> to tensor<1x?xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg3] [1, %4] [1, 1] : tensor<1x8777xf16> to tensor<1x?xf16>
    %5 = vector.create_mask %c1, %4 : vector<1x64xi1>
    %6 = vector.transfer_read %extracted_slice_2[%c0, %c0], %cst, %5 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
    %7 = arith.extf %6 : vector<1x64xf16> to vector<1x64xf32>
    %8 = vector.transfer_write %7, %extracted_slice[%c0, %c0], %5 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x?xf32>
    %inserted_slice = tensor.insert_slice %8 into %extracted_slice[0, 0] [1, %4] [1, 1] : tensor<1x?xf32> into tensor<1x?xf32>
    %9 = arith.mulf %7, %7 : vector<1x64xf32>
    %10 = arith.select %5, %9, %cst_0 : vector<1x64xi1>, vector<1x64xf32>
    %11 = arith.addf %10, %arg5 {reductionOp} : vector<1x64xf32>
    %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg4[0, %arg3] [1, %4] [1, 1] : tensor<1x?xf32> into tensor<1x8777xf32>
    scf.yield %inserted_slice_3, %11 : tensor<1x8777xf32>, vector<1x64xf32>
  } {reductionLoop}
  %1 = vector.transfer_read %arg1[%c0], %cst_1 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
  %2 = vector.multi_reduction <add>, %0#1, %1 [1] : vector<1x64xf32> to vector<1xf32>
  %3 = vector.transfer_write %2, %arg1[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
  return %3, %0#0 : tensor<1xf32>, tensor<1x8777xf32>
}


// -----

// CHECK-LABEL: @test_fold_extract_slice_insert_slice_static_shape
// CHECK-NOT: tensor.insert_slice {{.*}} into %slice
// The redundant insert_slice should be eliminated.
func.func @test_fold_extract_slice_insert_slice_static_shape(%arg0: tensor<4x64xf32>, %arg1: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> tensor<4x64xf32> {
    %slice = tensor.extract_slice %arg3[%c0, %c0] [4, 64] [1, 1] : tensor<4x64xf32> to tensor<4x64xf32>
    %vec = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
    %written = vector.transfer_write %vec, %slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
    %inserted = tensor.insert_slice %written into %slice[0, 0] [4, 64] [1, 1] : tensor<4x64xf32> into tensor<4x64xf32>
    scf.yield %inserted : tensor<4x64xf32>
  }
  return %0 : tensor<4x64xf32>
}

// -----
// CHECK-LABEL: @test_offsets_do_not_match
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[%c1, %c0]
func.func @test_offsets_do_not_match(%arg0: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %slice = tensor.extract_slice %arg0[%c0, %c0] [4, 64] [1, 1] : tensor<4x64xf32> to tensor<4x64xf32>
  %vec = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
  %written = vector.transfer_write %vec, %slice[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
  %inserted = tensor.insert_slice %written into %slice[%c1, %c0] [4, 64] [1, 1] : tensor<4x64xf32> into tensor<4x64xf32>
  return %inserted : tensor<4x64xf32>
}


