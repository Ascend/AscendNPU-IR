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

// -----

// Test: stride-1 extract_slice at non-zero static offset.
module {
  // CHECK-LABEL: func @vf_nonzero_offset(
  // CHECK-SAME: tensor<12x64xf16>
  // CHECK: tensor.extract_slice
  // CHECK: return
  func.func @vf_nonzero_offset(%arg0: tensor<4x64xf16>) -> tensor<4x64xf16>
      attributes {hivm.vector_function} {
    %0 = tensor.empty() : tensor<4x64xf16>
    %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
        ins(%arg0 : tensor<4x64xf16>) outs(%0 : tensor<4x64xf16>) -> tensor<4x64xf16>
    return %1 : tensor<4x64xf16>
  }

  // CHECK-LABEL: func @test_nonzero_static_offset(
  // CHECK-NOT: tensor.extract_slice{{.*}}[1, 0]{{.*}}tensor<12x64xf16>
  // CHECK: call @vf_nonzero_offset(%arg0
  // CHECK-SAME: {hivm.vector_function}
  func.func @test_nonzero_static_offset(%arg0: tensor<12x64xf16>) -> tensor<12x64xf16> {
    %slice = tensor.extract_slice %arg0[1, 0] [4, 64] [1, 1]
        : tensor<12x64xf16> to tensor<4x64xf16>
    %x = func.call @vf_nonzero_offset(%slice) {hivm.vector_function}
        : (tensor<4x64xf16>) -> tensor<4x64xf16>
    %r = tensor.insert_slice %x into %arg0[1, 0] [4, 64] [1, 1]
        : tensor<4x64xf16> into tensor<12x64xf16>
    return %r : tensor<12x64xf16>
  }
}

// -----

// Test: 1D extract_slice with size change (tensor<128xbf16> → tensor<32xbf16>).
// The old drop_begin left an empty range for rank-1, silently accepting all
// 1-D size changes as "standard".
module {
  // CHECK-LABEL: func @vf_1d_slice(
  // CHECK-SAME: tensor<128xbf16>
  func.func @vf_1d_slice(%arg0: tensor<32xbf16>) -> tensor<32xbf16>
      attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst
        : tensor<32xbf16>, vector<32xbf16>
    %1 = vector.transfer_write %0, %arg0[%c0]
        : vector<32xbf16>, tensor<32xbf16>
    return %1 : tensor<32xbf16>
  }

  // CHECK-LABEL: func @test_1d_size_change(
  // CHECK-NOT: tensor.extract_slice{{.*}}tensor<128xbf16> to tensor<32xbf16>
  // CHECK: call @vf_1d_slice(%arg0
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-SAME: tensor<128xbf16>
  func.func @test_1d_size_change(%arg0: tensor<128xbf16>) {
    %slice = tensor.extract_slice %arg0[0] [32] [1]
        : tensor<128xbf16> to tensor<32xbf16>
    %x = func.call @vf_1d_slice(%slice) {hivm.vector_function}
        : (tensor<32xbf16>) -> tensor<32xbf16>
    return
  }
}

// -----

// Test: Swap extract_slice + expand_shape into expand_shape + extract_slice,
// then pull the extract_slice into VF.  Covers both rank-restore (2D→1D→2D)
// and rank-increase (2D→2D→3D) scenarios.
module {
  // Rank-restore: 3x64 → 64 → 1x64  becomes  3x64 → 3x64 → 1x64
  // CHECK-LABEL: func @vf_rank_restore(
  // CHECK-SAME: tensor<3x64xf16>
  func.func @vf_rank_restore(%arg0: tensor<1x64xf16>) -> tensor<1x64xf16>
      attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst
        {in_bounds = [true, true]} : tensor<1x64xf16>, vector<1x64xf16>
    %1 = vector.transfer_write %0, %arg0[%c0, %c0]
        {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x64xf16>
    return %1 : tensor<1x64xf16>
  }

  // CHECK-LABEL: func @test_swap_rank_restore(
  // CHECK-NOT: tensor.expand_shape
  // CHECK: call @vf_rank_restore(%arg0
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-SAME: tensor<3x64xf16>
  func.func @test_swap_rank_restore(%arg0: tensor<3x64xf16>) {
    %slice = tensor.extract_slice %arg0[1, 0] [1, 64] [1, 1]
        : tensor<3x64xf16> to tensor<64xf16>
    %expanded = tensor.expand_shape %slice [[0, 1]] output_shape [1, 64]
        : tensor<64xf16> into tensor<1x64xf16>
    %x = func.call @vf_rank_restore(%expanded) {hivm.vector_function}
        : (tensor<1x64xf16>) -> tensor<1x64xf16>
    return
  }
}

// -----

// Test: Swap rank-increasing extract_slice + expand_shape (2D→2D→3D).
// 12x64 → 4x64 → 4x1x64  becomes  12x64 → 12x1x64 → 4x1x64
module {
  // CHECK-LABEL: func @vf_rank_increase(
  // CHECK-SAME: tensor<12x1x64xf16>
  func.func @vf_rank_increase(%arg0: tensor<4x1x64xf16>) -> tensor<4x1x64xf16>
      attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst
        {in_bounds = [true, true, true]} : tensor<4x1x64xf16>, vector<4x1x64xf16>
    %1 = vector.transfer_write %0, %arg0[%c0, %c0, %c0]
        {in_bounds = [true, true, true]} : vector<4x1x64xf16>, tensor<4x1x64xf16>
    return %1 : tensor<4x1x64xf16>
  }

  // CHECK-LABEL: func @test_swap_rank_increase(
  // CHECK-NOT: tensor.expand_shape{{.*}}tensor<4x64xf16>
  // CHECK: tensor.expand_shape{{.*}}tensor<12x64xf16> into tensor<12x1x64xf16>
  // CHECK: call @vf_rank_increase(
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-SAME: tensor<12x1x64xf16>
  func.func @test_swap_rank_increase(%arg0: tensor<12x64xf16>) {
    %slice = tensor.extract_slice %arg0[2, 0] [4, 64] [1, 1]
        : tensor<12x64xf16> to tensor<4x64xf16>
    %expanded = tensor.expand_shape %slice [[0], [1, 2]] output_shape [4, 1, 64]
        : tensor<4x64xf16> into tensor<4x1x64xf16>
    %x = func.call @vf_rank_increase(%expanded) {hivm.vector_function}
        : (tensor<4x1x64xf16>) -> tensor<4x1x64xf16>
    return
  }
}
