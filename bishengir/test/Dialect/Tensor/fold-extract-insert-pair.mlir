// RUN: bishengir-opt %s -fold-extract-insert-pair | FileCheck %s

// Fold a tensor.extract -> tensor.insert round trip on a statically
// one-element tensor back to the source tensor.

// CHECK-LABEL: @test_fold_extract_insert_pair
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
func.func @test_fold_extract_insert_pair(%arg0: i32, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %tmp = tensor.empty() : tensor<1xf32>
  %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg1, %arg2 : tensor<1xf32>, tensor<1xf32>) outs(%tmp : tensor<1xf32>) -> tensor<1xf32>
  %extracted_14 = tensor.extract %sub[%c0] : tensor<1xf32>
  %inserted_15 = tensor.insert %extracted_14 into %tmp[%c0] : tensor<1xf32>
  %exp = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%inserted_15 : tensor<1xf32>) outs(%tmp : tensor<1xf32>) -> tensor<1xf32>
  return %exp : tensor<1xf32>
}

// -----

// Fold redundant extract_slice -> transfer_write -> insert_slice pattern:
//
//   %slice = tensor.extract_slice %src[offsets] [sizes] [strides]
//   %written = vector.transfer_write %vec, %slice[offsets] {mask}
//   %result = tensor.insert_slice %written into %slice[offsets] [sizes] [1, 1]
//
// into:
//
//   %result = %written

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

// Negative: the transfer_write and insert_slice offsets differ, so the
// insert_slice must be kept.
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

// -----

// The write-back insert uses different offsets than the extract: no fold.
// CHECK-LABEL: func.func @no_fold_mismatched_offsets(
// CHECK: tensor.extract_slice %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16x16xf32>
// CHECK: scf.for {{.*}} iter_args({{%.*}} = %{{.*}}) -> (tensor<16x16xf32>) {
// CHECK: tensor.insert_slice %{{.*}} into %arg0[%arg2, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16xf32> into tensor<16x16x16xf32>
func.func @no_fold_mismatched_offsets(%arg0: tensor<16x16x16xf32>, %lb: index, %ub: index, %step: index) -> tensor<16x16x16xf32> {
  %extracted = tensor.extract_slice %arg0[%lb, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16x16xf32>
  %r = scf.for %i = %lb to %ub step %step iter_args(%arg = %extracted) -> (tensor<16x16xf32>) {
    scf.yield %arg : tensor<16x16xf32>
  }
  %inserted = tensor.insert_slice %r into %arg0[%ub, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16xf32> into tensor<16x16x16xf32>
  return %inserted : tensor<16x16x16xf32>
}

// -----

// An insert_slice into an extracted slice, written back into the same tensor
// at the same position, folds into a single insert_slice on the original
// tensor. This is the chain FoldSliceIntoScfForPattern creates inside the
// loop body.
// CHECK-LABEL: func.func @test_fold_insert_into_extracted_slice(
// CHECK: %[[INS:.*]] = tensor.insert_slice %{{.*}} into %arg0[%arg1, %arg2, 0] [1, 1, 16] [1, 1, 1] : tensor<16xf32> into tensor<16x16x16xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK: return %[[INS]]
func.func @test_fold_insert_into_extracted_slice(%arg0: tensor<16x16x16xf32>, %arg1: index, %arg2: index, %src: tensor<16xf32>) -> tensor<16x16x16xf32> {
  %extracted = tensor.extract_slice %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16x16xf32>
  %inner = tensor.insert_slice %src into %extracted[%arg2, 0] [1, 16] [1, 1] : tensor<16xf32> into tensor<16x16xf32>
  %res = tensor.insert_slice %inner into %arg0[%arg1, 0, 0] [1, 16, 16] [1, 1, 1] : tensor<16x16xf32> into tensor<16x16x16xf32>
  return %res : tensor<16x16x16xf32>
}
