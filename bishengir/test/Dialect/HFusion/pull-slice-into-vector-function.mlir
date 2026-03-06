// RUN: bishengir-opt %s --hfusion-pull-slice-into-vector-function -split-input-file | FileCheck %s

// -----

// Test Scenario 1 (passthrough): return flows from arg via scf.for iter_args;
// pattern inserts extract_slice on call result. Uses stride [2,1] for non-standard stride.
//
// Before: %slice = extract_slice(%full); %x = call @vf(%slice); return %x
// After:  call @vf(%full, ...) returns full; pattern inserts extract_slice on result
module {
  // CHECK-LABEL: func @vf_passthrough(
  // CHECK-SAME: tensor<16x32xf32>
  // CHECK: tensor.extract_slice
  // CHECK: scf.for
  // CHECK: return
  func.func @vf_passthrough(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %arg0) -> (tensor<4x8xf32>) {
      %1 = tensor.empty() : tensor<4x8xf32>
      %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
          ins(%iter : tensor<4x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
      scf.yield %2 : tensor<4x8xf32>
    }
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func @test_pull_extract_insert_slice_scenario_1(
  // CHECK: %[[CALL:.*]] = call @vf_passthrough(%arg0
  // CHECK-SAME: {hivm.vector_function}
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[CALL]]
  // CHECK: return %[[SLICE]]
  func.func @test_pull_extract_insert_slice_scenario_1(%arg0: tensor<16x32xf32>) -> tensor<4x8xf32> {
    %slice = tensor.extract_slice %arg0[0, 0] [4, 8] [2, 1]
        : tensor<16x32xf32> to tensor<4x8xf32>
    %x = func.call @vf_passthrough(%slice) {hivm.vector_function}
        : (tensor<4x8xf32>) -> tensor<4x8xf32>
    return %x : tensor<4x8xf32>
  }
}

// -----

// Test Scenario 2 (read-modify-write): extract_slice -> call -> insert_slice.
// Before: %a = extract_slice(%full); %b = call @vf(%a); %c = insert_slice(%b, %full)
// After:  %c = call @vf(%full, offsets, sizes, strides); VF does extract+compute+insert internally
module {
  // CHECK-LABEL: func @vf_extract_insert(
  // CHECK-SAME: tensor<16x32xf32>
  // CHECK: tensor.extract_slice
  // CHECK: linalg.elemwise
  // CHECK: tensor.insert_slice
  // CHECK: return
  func.func @vf_extract_insert(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
      attributes {hivm.vector_function} {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
        ins(%arg0 : tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func @test_pull_extract_insert_slice_scenario_2(
  // CHECK: %[[CALL:.*]] = call @vf_extract_insert(%arg0
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-NOT: tensor.insert_slice
  // CHECK: return %[[CALL]]
  func.func @test_pull_extract_insert_slice_scenario_2(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    // extract 4x8 with stride [2,1]: non-standard stride triggers the pattern
    %slice = tensor.extract_slice %arg0[0, 0] [4, 8] [2, 1]
        : tensor<16x32xf32> to tensor<4x8xf32>
    %b = func.call @vf_extract_insert(%slice) {hivm.vector_function}
        : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %c = tensor.insert_slice %b into %arg0[0, 0] [4, 8] [2, 1]
        : tensor<4x8xf32> into tensor<16x32xf32>
    return %c : tensor<16x32xf32>
  }
}

// -----

// Test rank-reduce: 2D to 1D extract_slice;
// Same handling as Scenario 1.
module {
  // CHECK-LABEL: func @vf_rank_reduction(
  // CHECK-SAME: tensor<2x64xf32>
  // CHECK: tensor.extract_slice
  // CHECK: linalg.elemwise
  // CHECK: tensor.insert_slice
  // CHECK: return
  func.func @vf_rank_reduction(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.vector_function} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }

  // CHECK-LABEL: func @test_rank_reduction(
  // CHECK: %[[CALL:.*]] = call @vf_rank_reduction(%arg0
  // CHECK-SAME: {hivm.vector_function}
  // CHECK: return %[[CALL]]
  func.func @test_rank_reduction(%arg0: tensor<2x64xf32>, %arg1: index) -> tensor<2x64xf32> {
    %c0 = arith.constant 0 : index
    %slice = tensor.extract_slice %arg0[%arg1, 0] [1, 64] [1, 1]
        : tensor<2x64xf32> to tensor<64xf32>
    %x = func.call @vf_rank_reduction(%slice) {hivm.vector_function}
        : (tensor<64xf32>) -> tensor<64xf32>
    %4 = tensor.insert_slice %x into %arg0[%arg1, 0] [1, 64] [1, 1] : tensor<64xf32> into tensor<2x64xf32>
    return %4 : tensor<2x64xf32>
  }
}
