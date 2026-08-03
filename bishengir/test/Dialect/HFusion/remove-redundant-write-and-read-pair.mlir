// RUN: bishengir-opt %s -remove-redundant-write-and-read-pair -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// Test 1: Same shape — transfer_read is replaced directly by write's vector
func.func @remove_redundant_write_and_read_pair(%arg0: tensor<2x80xf16>, %arg1: tensor<2x80xf16>, %arg2: tensor<2x80xf16>, %arg3: tensor<2x80xf16>) -> tensor<2x80xf16> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c80 = arith.constant 80 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> tensor<2x80xf16> {
    %1 = scf.for %arg6 = %c0 to %c80 step %c64 iter_args(%arg7 = %arg5) -> tensor<2x80xf16> {
      %2 = affine.min affine_map<(d0) -> (-d0 + 80, 64)>(%arg6)
      %extracted_slice_0 = tensor.extract_slice %arg0[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %extracted_slice_4 = tensor.extract_slice %arg2[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %4 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %5 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst_0, %4 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
      %6 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst_0, %4 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
      %7 = arith.addf %5, %6 : vector<1x64xf16>
      %8 = vector.transfer_write %7, %extracted_slice_4[%c0, %c0], %4 {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x?xf16>
      %inserted_slice = tensor.insert_slice %8 into %extracted_slice_4[0, 0] [1, %2] [1, 1] : tensor<1x?xf16> into tensor<1x?xf16>
      %12 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %13 = vector.transfer_read %inserted_slice[%c0, %c0], %cst_0, %12 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
      %20 = arith.subf %13, %5 : vector<1x64xf16>
      %extracted_slice_8 = tensor.extract_slice %arg7[%arg4, %arg6] [1, %2] [1, 1] : tensor<2x80xf16> to tensor<1x?xf16>
      %22 = vector.transfer_write %20, %extracted_slice_8[%c0, %c0], %12 {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x?xf16>
      %inserted_slice_9 = tensor.insert_slice %22 into %arg7[%arg4, %arg6] [1, %2] [1, 1] : tensor<1x?xf16> into tensor<2x80xf16>
      scf.yield %inserted_slice_9 : tensor<2x80xf16>
    }
    scf.yield %1 : tensor<2x80xf16>
  }
  return %0 : tensor<2x80xf16>
}

// CHECK-LABEL:   func.func @remove_redundant_write_and_read_pair
// CHECK: %[[ADD:.*]] = arith.addf
// CHECK: %[[SUB:.*]] = arith.subf %[[ADD]], %{{.*}}

// Test 2: Different shape (same element count) — shape_cast is inserted
// transfer_write outputs vector<1x64xf16>, but transfer_read expects vector<64xf16>
func.func @shape_cast_different_vector_shape(%arg0: tensor<2x64xf16>, %arg1: tensor<2x64xf16>, %arg2: tensor<2x64xf16>) -> tensor<2x64xf16> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg2) -> tensor<2x64xf16> {
    %1 = tensor.extract_slice %arg0[%arg3, %c0] [1, %c64] [1, 1] : tensor<2x64xf16> to tensor<1x?xf16>
    %2 = tensor.extract_slice %arg1[%arg3, %c0] [1, %c64] [1, 1] : tensor<2x64xf16> to tensor<1x?xf16>
    %3 = tensor.extract_slice %arg4[%arg3, %c0] [1, %c64] [1, 1] : tensor<2x64xf16> to tensor<1x?xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
    %5 = vector.transfer_read %2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x?xf16>, vector<1x64xf16>
    %6 = arith.addf %4, %5 : vector<1x64xf16>
    %7 = vector.transfer_write %6, %3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x?xf16>
    %inserted = tensor.insert_slice %7 into %3[0, 0] [1, %c64] [1, 1] : tensor<1x?xf16> into tensor<1x?xf16>
    %8 = vector.transfer_read %inserted[%c0, %c0], %cst_0 {in_bounds = [true]} : tensor<1x?xf16>, vector<64xf16>
    %9 = arith.mulf %8, %8 : vector<64xf16>
    %10 = vector.transfer_write %9, %3[%c0, %c0] {in_bounds = [true]} : vector<64xf16>, tensor<1x?xf16>
    %inserted2 = tensor.insert_slice %10 into %arg4[%arg3, %c0] [1, %c64] [1, 1] : tensor<1x?xf16> into tensor<2x64xf16>
    scf.yield %inserted2 : tensor<2x64xf16>
  }
  return %0 : tensor<2x64xf16>
}

// CHECK-LABEL:   func.func @shape_cast_different_vector_shape
// CHECK: %[[ADD:.*]] = arith.addf
// CHECK-NOT: vector.transfer_read
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[ADD]] : vector<1x64xf16> to vector<64xf16>
// CHECK: arith.mulf %[[CAST]], %[[CAST]]

// Test 3: widened masked read right after the write -- the read is replaced by a
// broadcast of the written vector and the write becomes dead.
func.func @fold_widened_read_after_write(%arg0: tensor<16x1xf32>, %arg1: tensor<16x1x16xf32>, %arg2: tensor<16x1xf32>, %arg7: tensor<16x1xf32>) -> tensor<16x1xf32> attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<1x1x64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %arg8 = %c0 to %c16 step %c1 iter_args(%arg9 = %arg7) -> (tensor<16x1xf32>) {
    %extracted_slice = tensor.extract_slice %arg9[%arg8, 0] [1, 1] [1, 1] : tensor<16x1xf32> to tensor<1x1xf32>
    %extracted_slice_1 = tensor.extract_slice %arg0[%arg8, 0] [1, 1] [1, 1] : tensor<16x1xf32> to tensor<1x1xf32>
    %extracted_slice_2 = tensor.extract_slice %arg1[%arg8, 0, 0] [1, 1, 16] [1, 1, 1] : tensor<16x1x16xf32> to tensor<1x1x16xf32>
    %extracted_slice_3 = tensor.extract_slice %arg2[%arg8, 0] [1, 1] [1, 1] : tensor<16x1xf32> to tensor<1x1xf32>
    %1 = vector.constant_mask [1, 1, 16] : vector<1x1x64xi1>
    %2 = vector.transfer_read %extracted_slice_2[%c0, %c0, %c0], %cst_0, %1 {in_bounds = [true, true, true]} : tensor<1x1x16xf32>, vector<1x1x64xf32>
    %3 = arith.select %1, %2, %cst : vector<1x1x64xi1>, vector<1x1x64xf32>
    %4 = arith.addf %3, %cst {reductionOp} : vector<1x1x64xf32>
    %5 = vector.transfer_read %extracted_slice_3[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x1xf32>, vector<1x1xf32>
    %6 = vector.multi_reduction <add>, %4, %5 {withoutInitMergeOp} [2] : vector<1x1x64xf32> to vector<1x1xf32>
    %7 = vector.transfer_write %6, %extracted_slice_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<1x1xf32>
    %8 = vector.constant_mask [1, 1] : vector<1x64xi1>
    %9 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst_0, %8 {in_bounds = [true, true]} : tensor<1x1xf32>, vector<1x64xf32>
    %10 = vector.transfer_read %7[%c0, %c0], %cst_0, %8 {in_bounds = [true, true]} : tensor<1x1xf32>, vector<1x64xf32>
    %11 = arith.divf %9, %10 : vector<1x64xf32>
    %12 = vector.transfer_write %11, %extracted_slice[%c0, %c0], %8 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x1xf32>
    %inserted_slice = tensor.insert_slice %12 into %arg9[%arg8, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<16x1xf32>
    scf.yield %inserted_slice : tensor<16x1xf32>
  }
  return %0 : tensor<16x1xf32>
}

// CHECK-LABEL: @fold_widened_read_after_write
// CHECK:       %[[RED:.*]] = vector.multi_reduction <add>
// CHECK-NOT:   vector.transfer_write %[[RED]]
// CHECK:       %[[BC:.*]] = vector.broadcast %[[RED]] : vector<1x1xf32> to vector<1x64xf32>
// CHECK:       arith.divf %{{.*}}, %[[BC]] : vector<1x64xf32>

// Test 4: rank 4, constant_mask. The written vector<1x1x1x1xf32> covers the
// whole masked-in region [1, 1, 1, 1] of the read, so the read is replaced by a
// broadcast and the write dies.
func.func @fold_widened_read_rank_4(%arg0: tensor<8x4x2x1xf32>, %arg1: tensor<8x4x2x64xf32>, %arg2: tensor<8x4x2x64xf32>) -> tensor<8x4x2x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %arg2) -> (tensor<8x4x2x64xf32>) {
    %scale_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<8x4x2x1xf32> to tensor<1x1x1x1xf32>
    %row_slice = tensor.extract_slice %arg1[%arg3, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<8x4x2x64xf32> to tensor<1x1x1x64xf32>
    %out_slice = tensor.extract_slice %arg4[%arg3, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<8x4x2x64xf32> to tensor<1x1x1x64xf32>
    %1 = vector.transfer_read %scale_slice[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : tensor<1x1x1x1xf32>, vector<1x1x1x1xf32>
    %2 = arith.addf %1, %1 : vector<1x1x1x1xf32>
    %3 = vector.transfer_write %2, %scale_slice[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xf32>, tensor<1x1x1x1xf32>
    %mask = vector.constant_mask [1, 1, 1, 1] : vector<1x1x1x64xi1>
    %4 = vector.transfer_read %3[%c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true]} : tensor<1x1x1x1xf32>, vector<1x1x1x64xf32>
    %5 = vector.transfer_read %row_slice[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : tensor<1x1x1x64xf32>, vector<1x1x1x64xf32>
    %6 = arith.divf %5, %4 : vector<1x1x1x64xf32>
    %7 = vector.transfer_write %6, %out_slice[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x64xf32>, tensor<1x1x1x64xf32>
    %inserted_slice = tensor.insert_slice %7 into %arg4[%arg3, 0, 0, 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<1x1x1x64xf32> into tensor<8x4x2x64xf32>
    scf.yield %inserted_slice : tensor<8x4x2x64xf32>
  }
  return %0 : tensor<8x4x2x64xf32>
}

// CHECK-LABEL: func.func @fold_widened_read_rank_4
// CHECK:       %[[ADD:.*]] = arith.addf
// CHECK-NOT:   vector.transfer_write %[[ADD]]
// CHECK:       %[[BC:.*]] = vector.broadcast %[[ADD]] : vector<1x1x1x1xf32> to vector<1x1x1x64xf32>
// CHECK:       %[[ROW:.*]] = vector.transfer_read
// CHECK:       arith.divf %[[ROW]], %[[BC]]

// Test 5: rank 5, create_mask with constant bounds. Same fold, exercises the
// create_mask branch of getStaticMaskSizes.
func.func @fold_widened_read_rank_5(%arg0: tensor<2x1x1x1x1xf32>, %arg1: tensor<2x1x1x1x128xf32>) -> vector<1x1x1x1x128xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %scale_slice = tensor.extract_slice %arg0[0, 0, 0, 0, 0] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : tensor<2x1x1x1x1xf32> to tensor<1x1x1x1x1xf32>
  %row_slice = tensor.extract_slice %arg1[0, 0, 0, 0, 0] [1, 1, 1, 1, 128] [1, 1, 1, 1, 1] : tensor<2x1x1x1x128xf32> to tensor<1x1x1x1x128xf32>
  %0 = vector.transfer_read %scale_slice[%c0, %c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x1xf32>, vector<1x1x1x1x1xf32>
  %1 = arith.mulf %0, %0 : vector<1x1x1x1x1xf32>
  %2 = vector.transfer_write %1, %scale_slice[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>
  %mask = vector.create_mask %c1, %c1, %c1, %c1, %c1 : vector<1x1x1x1x128xi1>
  %3 = vector.transfer_read %2[%c0, %c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x1xf32>, vector<1x1x1x1x128xf32>
  %4 = vector.transfer_read %row_slice[%c0, %c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true, true]} : tensor<1x1x1x1x128xf32>, vector<1x1x1x1x128xf32>
  %5 = arith.subf %4, %3 : vector<1x1x1x1x128xf32>
  return %5 : vector<1x1x1x1x128xf32>
}

// CHECK-LABEL: func.func @fold_widened_read_rank_5
// CHECK:       %[[MUL:.*]] = arith.mulf
// CHECK-NOT:   vector.transfer_write %[[MUL]]
// CHECK:       %[[BC:.*]] = vector.broadcast %[[MUL]] : vector<1x1x1x1x1xf32> to vector<1x1x1x1x128xf32>
// CHECK:       %[[ROW:.*]] = vector.transfer_read
// CHECK:       arith.subf %[[ROW]], %[[BC]]

// Negative test 1: the read consumes more elements than the write produced
// (mask is [1, 2] while only 1x1 was written), so nothing must be folded.
func.func @no_fold_read_escapes_write(%arg0: tensor<1x2xf32>, %v: vector<1x1xf32>) -> vector<1x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %m = vector.constant_mask [1, 2] : vector<1x64xi1>
  %w = vector.transfer_write %v, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, tensor<1x2xf32>
  %r = vector.transfer_read %w[%c0, %c0], %cst_0, %m {in_bounds = [true, true]} : tensor<1x2xf32>, vector<1x64xf32>
  return %r : vector<1x64xf32>
}

// CHECK-LABEL: @no_fold_read_escapes_write
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK-NOT:   vector.broadcast

// Negative test 2: rank 4, the read consumes elements the write never produced. The mask
// is [1, 1, 2, 1] while the write only covered 1 element in dim 2, so element
// [0, 0, 1, 0] still comes from the original tensor. Must not fold.
func.func @no_fold_read_escapes_write_rank_4(%arg0: tensor<1x1x2x1xf32>, %v: vector<1x1x1x1xf32>) -> vector<1x1x2x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xf32>, tensor<1x1x2x1xf32>
  %mask = vector.constant_mask [1, 1, 2, 1] : vector<1x1x2x64xi1>
  %1 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true]} : tensor<1x1x2x1xf32>, vector<1x1x2x64xf32>
  return %1 : vector<1x1x2x64xf32>
}

// CHECK-LABEL: func.func @no_fold_read_escapes_write_rank_4
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK-NOT:   vector.broadcast
// CHECK:       return

// Negative test 3: rank 4, the region checks pass but the result is not a single row
// (dim 2 is 4). VecBroadcastOpPattern in VectorToHIVMAVE builds the dup type
// from the trailing dimension only, so such a broadcast would not legalize.
// Must not fold.
func.func @no_fold_multi_row_result_rank_4(%arg0: tensor<1x1x1x1xf32>, %v: vector<1x1x1x1xf32>) -> vector<1x1x4x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xf32>, tensor<1x1x1x1xf32>
  %mask = vector.constant_mask [1, 1, 1, 1] : vector<1x1x4x64xi1>
  %1 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true]} : tensor<1x1x1x1xf32>, vector<1x1x4x64xf32>
  return %1 : vector<1x1x4x64xf32>
}

// CHECK-LABEL: func.func @no_fold_multi_row_result_rank_4
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK-NOT:   vector.broadcast
// CHECK:       return

// Negative test 4: rank 4, the written vector holds more than one element, so widening
// cannot be expressed as a single broadcast. Must not fold.
func.func @no_fold_multi_element_write_rank_4(%arg0: tensor<1x1x2x1xf32>, %v: vector<1x1x2x1xf32>) -> vector<1x1x2x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x2x1xf32>, tensor<1x1x2x1xf32>
  %mask = vector.constant_mask [1, 1, 2, 1] : vector<1x1x2x64xi1>
  %1 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true]} : tensor<1x1x2x1xf32>, vector<1x1x2x64xf32>
  return %1 : vector<1x1x2x64xf32>
}

// CHECK-LABEL: func.func @no_fold_multi_element_write_rank_4
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK-NOT:   vector.broadcast
// CHECK:       return

// Negative test 5: rank 4, the mask bound in dim 2 is dynamic, so the masked-in region
// cannot be proven to fit into the written vector. Must not fold.
func.func @no_fold_dynamic_mask_rank_4(%arg0: tensor<1x1x2x1xf32>, %v: vector<1x1x1x1xf32>, %n: index) -> vector<1x1x2x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.transfer_write %v, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xf32>, tensor<1x1x2x1xf32>
  %mask = vector.create_mask %c1, %c1, %n, %c1 : vector<1x1x2x64xi1>
  %1 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true]} : tensor<1x1x2x1xf32>, vector<1x1x2x64xf32>
  return %1 : vector<1x1x2x64xf32>
}

// CHECK-LABEL: func.func @no_fold_dynamic_mask_rank_4
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK-NOT:   vector.broadcast
// CHECK:       return

// Negative test 6: rank 4, the write itself is masked, so it does not define the whole
// region the read consumes. Must not fold.
func.func @no_fold_masked_write_rank_4(%arg0: tensor<1x1x1x1xf32>, %v: vector<1x1x1x1xf32>, %n: index) -> vector<1x1x1x64xf32> attributes {hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %wmask = vector.create_mask %c1, %c1, %c1, %n : vector<1x1x1x1xi1>
  %0 = vector.transfer_write %v, %arg0[%c0, %c0, %c0, %c0], %wmask {in_bounds = [true, true, true, true]} : vector<1x1x1x1xf32>, tensor<1x1x1x1xf32>
  %mask = vector.constant_mask [1, 1, 1, 1] : vector<1x1x1x64xi1>
  %1 = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst_0, %mask {in_bounds = [true, true, true, true]} : tensor<1x1x1x1xf32>, vector<1x1x1x64xf32>
  return %1 : vector<1x1x1x64xf32>
}

// CHECK-LABEL: func.func @no_fold_masked_write_rank_4
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK-NOT:   vector.broadcast
// CHECK:       return