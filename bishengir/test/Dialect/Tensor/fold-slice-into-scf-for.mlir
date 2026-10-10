// RUN: bishengir-opt -fold-slice-into-scf-for -fold-extract-insert-pair --split-input-file %s | FileCheck %s
// RUN: bishengir-opt -fold-slice-into-scf-for -fold-extract-insert-pair -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" --split-input-file %s | FileCheck %s --check-prefix=CHECK-ONE-SHOT

module {
  // CHECK-LABEL: func.func @fold_slice_into_for(
  // CHECK: tensor.extract_slice
  // CHECK: tensor.extract_slice
  // CHECK-NOT: tensor.extract_slice
  // CHECK: tensor.insert_slice
  // CHECK-NOT: tensor.insert_slice
  func.func @fold_slice_into_for(%arg0: tensor<16x16x16xf32>, %arg1: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %1 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg1) -> (tensor<16x16x16xf32>) {
      %extracted_slice = tensor.extract_slice %arg3[%arg2, %c0, %c0] [1, 16, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16x16xf32>
      %2 = scf.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %extracted_slice) -> (tensor<16x16xf32>) {
        %extracted_slice_0 = tensor.extract_slice %arg0[%arg2, %arg4, 0] [1, 1, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16xf32>
        %extracted_slice_1 = tensor.extract_slice %arg5[%arg4, %c0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<16xf32>
        %3 = vector.transfer_read %extracted_slice_0[%c0], %cst, %0 {in_bounds = [true]} : tensor<16xf32>, vector<64xf32>
        %4 = vector.transfer_write %3, %extracted_slice_1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
        %inserted_slice_2 = tensor.insert_slice %4 into %arg5[%arg4, %c0] [1, 16] [1, 1] : tensor<16xf32> into tensor<16x16xf32>
        scf.yield %inserted_slice_2 : tensor<16x16xf32>
      }
      %inserted_slice = tensor.insert_slice %2 into %arg3[%arg2, %c0, %c0] [1, 16, 16] [1, 1, 1] : tensor<16x16xf32> into tensor<16x16x16xf32>
      scf.yield %inserted_slice : tensor<16x16x16xf32>
    }
    return %1 : tensor<16x16x16xf32>
  }
  // CHECK-LABEL: func.func @test_not_conflict_in_vf_with_nest_loop
  // CHECK-ONE-SHOT-LABEL: func.func @test_not_conflict_in_vf_with_nest_loop
  func.func @test_not_conflict_in_vf_with_nest_loop(%arg0: tensor<16x16x16xf32>, %arg1: memref<16x16x16xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16x16xf32>
    %1 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%0 : tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %1) -> (tensor<16x16x16xf32>)  : i32 {
      %3 = tensor.empty() : tensor<16x16x16xf32>
      // CHECK-ONE-SHOT: %[[CALL_RESULT:.*]] = func.call @fold_slice_into_for
      %4 = func.call @fold_slice_into_for(%arg3, %3) {hivm.vector_function, no_inline} : (tensor<16x16x16xf32>, tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
      // CHECK-ONE-SHOT: scf.yield %[[CALL_RESULT]] : memref<16x16x16xf32>
      scf.yield %4 : tensor<16x16x16xf32>
    }
    return
  }
}
