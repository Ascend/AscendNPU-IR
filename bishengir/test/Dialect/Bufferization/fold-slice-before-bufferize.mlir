// RUN: bishengir-opt -fold-slice-into-scf-for -merge-consecutive-insert-extract-slice -fold-extract-insert-pair --split-input-file %s | FileCheck %s
// RUN: bishengir-opt -fold-slice-into-scf-for -merge-consecutive-insert-extract-slice -cse -fold-extract-insert-pair -tensor-copy-insertion="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down" --split-input-file %s | FileCheck %s --check-prefix=CHECK-RAW

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
  // CHECK-RAW-LABEL: func.func @test_not_conflict_in_vf_with_nest_loop
  func.func @test_not_conflict_in_vf_with_nest_loop(%arg0: tensor<16x16x16xf32>, %arg1: memref<16x16x16xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16x16xf32>
    %1 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%0 : tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %1) -> (tensor<16x16x16xf32>)  : i32 {
      %3 = tensor.empty() : tensor<16x16x16xf32>
      // CHECK-RAW: %[[CALL_RESULT:.*]] = func.call @fold_slice_into_for
      %4 = func.call @fold_slice_into_for(%arg3, %3) {hivm.vector_function, no_inline} : (tensor<16x16x16xf32>, tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
      // CHECK-RAW: scf.yield %[[CALL_RESULT]] : tensor<16x16x16xf32>
      scf.yield %4 : tensor<16x16x16xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @multi_use_loop_result(
// CHECK-RAW-LABEL: func.func @multi_use_loop_result(
// CHECK: %[[OUTER:.*]] = scf.for
// CHECK: %[[INNER:.*]] = scf.for
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INNER]][%{{.*}}, %{{.*}}, %{{.*}}] [1, 16, 16] [1, 1, 1] : tensor<16x16x16xf32> to tensor<16x16xf32>
// CHECK: hivm.hir.debug {{.*}} %[[SLICE]] : tensor<16x16xf32>
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: scf.yield %[[INNER]] : tensor<16x16x16xf32>
// CHECK: return %[[OUTER]] : tensor<16x16x16xf32>
func.func @multi_use_loop_result(%arg0: tensor<16x16x16xf32>, %arg1: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> attributes {hivm.vector_function, no_inline} {
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
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " slice : "} %2 : tensor<16x16xf32>
    scf.yield %inserted_slice : tensor<16x16x16xf32>
  }
  return %1 : tensor<16x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @multi_use_insert_result(
// CHECK-RAW-LABEL: func.func @multi_use_insert_result(
// CHECK: %[[OUTER:.*]] = scf.for
// CHECK: %[[INNER:.*]] = scf.for
// CHECK: hivm.hir.debug {{.*}} %[[INNER]] : tensor<16x16x16xf32>
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: scf.yield %[[INNER]] : tensor<16x16x16xf32>
// CHECK: return %[[OUTER]] : tensor<16x16x16xf32>
func.func @multi_use_insert_result(%arg0: tensor<16x16x16xf32>, %arg1: tensor<16x16x16xf32>) -> tensor<16x16x16xf32> attributes {hivm.vector_function, no_inline} {
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
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " full : "} %inserted_slice : tensor<16x16x16xf32>
    scf.yield %inserted_slice : tensor<16x16x16xf32>
  }
  return %1 : tensor<16x16x16xf32>
}

// -----
// CHECK-LABEL: func.func @fold_slice_and_cse(
// CHECK-RAW-LABEL: func.func @fold_slice_and_cse(
// CHECK-RAW: tensor.extract_slice
// CHECK-RAW-NOT: tensor.extract_slice
// CHECK-RAW: tensor.insert_slice
// CHECK-RAW-NOT: tensor.insert_slice
// CHECK-RAW-NOT: copy
func.func @fold_slice_and_cse(%arg0: tensor<2x1031xi64>, %arg2: tensor<2x1031xi64>) -> tensor<2x1031xi64> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0> : vector<1x64xi64>
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c1031 = arith.constant 1031 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg2) -> (tensor<2x1031xi64>) {
    %1 = scf.for %arg5 = %c0 to %c1031 step %c64 iter_args(%arg6 = %arg4) -> (tensor<2x1031xi64>) {
      %2 = affine.min affine_map<(d0) -> (-d0 + 1031, 64)>(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5] [1, %2] [1, 1] : tensor<2x1031xi64> to tensor<1x?xi64>
      %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0] [1, %2] [1, 1] : tensor<1x?xi64> to tensor<1x?xi64>
      %3 = vector.create_mask %c1, %c1, %2 : vector<1x1x64xi1>
      %4 = vector.create_mask %c1, %2 : vector<1x64xi1>
      %5 = vector.shape_cast %3 : vector<1x1x64xi1> to vector<1x64xi1>
      %6 = vector.transfer_write %cst, %extracted_slice_0[%c0, %c0], %4 {in_bounds = [true, true]} : vector<1x64xi64>, tensor<1x?xi64>
      %inserted_slice = tensor.insert_slice %6 into %extracted_slice[0, 0] [1, %2] [1, 1] : tensor<1x?xi64> into tensor<1x?xi64>
      %inserted_slice_1 = tensor.insert_slice %inserted_slice into %arg6[%arg3, %arg5] [1, %2] [1, 1] : tensor<1x?xi64> into tensor<2x1031xi64>
      scf.yield %inserted_slice_1 : tensor<2x1031xi64>
    }
    scf.yield %1 : tensor<2x1031xi64>
  }
  return %0 : tensor<2x1031xi64>
}
