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

// -----
module {
  // CHECK-LABEL: func @test_slice_to_unit_keeps_src_dim(
  // CHECK: %[[E:.*]] = tensor.expand_shape %arg0
  // CHECK-SAME: output_shape [1, 1, 1, 1, 1, 1, 2]
  // CHECK-SAME: tensor<2xi32> into tensor<1x1x1x1x1x1x2xi32>
  // CHECK: tensor.extract_slice %[[E]]
  // CHECK-SAME: tensor<1x1x1x1x1x1x2xi32> to tensor<1x1x1x1x1x1x1xi32>
  // CHECK-NOT: tensor<2xi32> into tensor<1x1x1x1x1x1x1xi32>
  func.func @test_slice_to_unit_keeps_src_dim(%arg0: tensor<2xi32>, %arg1: index)
      -> tensor<1x1x1x1x1x1x1xi32> {
    %slice = tensor.extract_slice %arg0[%arg1] [1] [1]
        : tensor<2xi32> to tensor<1xi32>
    %expanded = tensor.expand_shape %slice [[0, 1, 2, 3, 4, 5, 6]]
        output_shape [1, 1, 1, 1, 1, 1, 1]
        : tensor<1xi32> into tensor<1x1x1x1x1x1x1xi32>
    return %expanded : tensor<1x1x1x1x1x1x1xi32>
  }
}


// -----
// Test: Swap rank-preserving extract_slice + rank-reducing collapse_shape
// into collapse_shape + extract_slice, then pull the slice into VF.
// 1x16x4x16 -> 1x1x4x16 -> 1x64  becomes  1x16x4x16 -> 16x64 -> 1x64
module {
  // CHECK-LABEL: func @vf_collapse_rank_reduce(
  // CHECK-SAME: tensor<16x64xf16>
  func.func @vf_collapse_rank_reduce(%arg0: tensor<1x64xf16>) -> tensor<1x64xf16>
      attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst
        {in_bounds = [true, true]} : tensor<1x64xf16>, vector<1x64xf16>
    %1 = vector.transfer_write %0, %arg0[%c0, %c0]
        {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x64xf16>
    return %1 : tensor<1x64xf16>
  }
  // CHECK-LABEL: func @test_swap_collapse_rank_reduce(
  // CHECK-NOT: tensor.extract_slice{{.*}}tensor<1x16x4x16xf16> to tensor<1x1x4x16xf16>
  // CHECK: tensor.collapse_shape{{.*}}tensor<1x16x4x16xf16> into tensor<16x64xf16>
  // CHECK: call @vf_collapse_rank_reduce(
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-SAME: tensor<16x64xf16>
  func.func @test_swap_collapse_rank_reduce(%arg0: tensor<1x16x4x16xf16>) {
    %slice = tensor.extract_slice %arg0[0, 3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1]
        : tensor<1x16x4x16xf16> to tensor<1x1x4x16xf16>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2, 3]]
        : tensor<1x1x4x16xf16> into tensor<1x64xf16>
    %x = func.call @vf_collapse_rank_reduce(%collapsed) {hivm.vector_function}
        : (tensor<1x64xf16>) -> tensor<1x64xf16>
    return
  }
}

// -----

// Test: Swap extract_slice + collapse_shape when the collapsed slice is not
// contiguous but is a regular strided line. This mirrors taking one dynamic
// column from tensor<32x4xf32>, collapsing tensor<32x1xf32> to tensor<32xf32>,
// and passing it to a VF.
module {
  // CHECK-LABEL: func @vf_collapse_strided_line(
  // CHECK-SAME: tensor<128xf32>
  // CHECK-SAME: index
  // CHECK-SAME: tensor<32xf32>
  // CHECK: tensor.extract_slice %{{.*}}[%{{.*}}] [32] [4] : tensor<128xf32> to tensor<32xf32>
  func.func @vf_collapse_strided_line(%arg0: tensor<32xf32>,
                                      %arg1: tensor<32xf32>) -> tensor<32xf32>
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : vector<32xf32>
    %0 = vector.transfer_read %arg0[%c0], %cst
        {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
    %1 = arith.addf %0, %cst_0 : vector<32xf32>
    %2 = vector.transfer_write %1, %arg1[%c0]
        {in_bounds = [true]} : vector<32xf32>, tensor<32xf32>
    return %2 : tensor<32xf32>
  }

  // CHECK-LABEL: func @test_swap_collapse_dynamic_offset_strided_line(
  // CHECK-SAME: %[[ARG0:.*]]: tensor<32x4xf32>
  // CHECK-SAME: %[[COL:.*]]: index
  // CHECK-SAME: %[[OUT:.*]]: tensor<32xf32>
  // CHECK: %[[FULL:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1]] : tensor<32x4xf32> into tensor<128xf32>
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape{{.*}}tensor<32x1xf32> into tensor<32xf32>
  // CHECK: %[[CALL:.*]] = call @vf_collapse_strided_line(%[[FULL]], %[[COL]], %[[OUT]])
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-SAME: (tensor<128xf32>, index, tensor<32xf32>) -> tensor<32xf32>
  // CHECK: return %[[CALL]]
  func.func @test_swap_collapse_dynamic_offset_strided_line(%arg0: tensor<32x4xf32>,
                                                            %col: index,
                                                            %out: tensor<32xf32>)
      -> tensor<32xf32> {
    %slice = tensor.extract_slice %arg0[0, %col] [32, 1] [1, 1]
        : tensor<32x4xf32> to tensor<32x1xf32>
    %collapsed = tensor.collapse_shape %slice [[0, 1]]
        : tensor<32x1xf32> into tensor<32xf32>
    %0 = func.call @vf_collapse_strided_line(%collapsed, %out) {hivm.vector_function}
        : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}

// -----

// Test: VF operand is built by inserting one extracted slice into another
// extracted slice. Both slices and the insert_slice should be pulled into the
// VF body, so the caller passes the two full source tensors instead of the
// materialized 1-D tensor.
module {
  // CHECK-LABEL: func @vf_recursive_slice_operand(
  // CHECK-SAME: tensor<2x32xf32>
  // CHECK-SAME: tensor<2x32xf32>
  // CHECK: %[[PATCH:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [1, %{{.*}}] [1, 1] : tensor<2x32xf32> to tensor<?xf32>
  // CHECK: %[[BASE:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [1, 32] [1, 1] : tensor<2x32xf32> to tensor<32xf32>
  // CHECK: %[[MERGED:.*]] = tensor.insert_slice %[[PATCH]] into %[[BASE]][0] [%{{.*}}] [1] : tensor<?xf32> into tensor<32xf32>
  // CHECK: vector.transfer_read %[[MERGED]]
  func.func @vf_recursive_slice_operand(%arg0: tensor<32xf32>,
                                        %arg1: tensor<2x32x64xf32>,
                                        %arg2: index,
                                        %arg3: index,
                                        %arg4: index,
                                        %arg5: index,
                                        %arg6: index,
                                        %arg7: tensor<32x64xf32>)
      -> tensor<32x64xf32> attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]}
        : tensor<32xf32>, vector<32xf32>
    %1 = vector.transfer_write %0, %arg7[%c0, %c0] {in_bounds = [true]}
        : vector<32xf32>, tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }

  // CHECK-LABEL: func @test_pull_recursive_slice_operand(
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.insert_slice
  // CHECK: call @vf_recursive_slice_operand(%arg0
  // CHECK-SAME: %arg1
  // CHECK-SAME: {hivm.vector_function, no_inline}
  func.func @test_pull_recursive_slice_operand(%arg0: tensor<2x32xf32>,
                                              %arg1: tensor<2x32xf32>,
                                              %arg2: tensor<2x32x64xf32>,
                                              %arg3: index,
                                              %arg4: index,
                                              %arg5: tensor<32x64xf32>)
      -> tensor<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %extracted_slice = tensor.extract_slice %arg1[%arg3, 0] [1, %arg4] [1, 1]
        : tensor<2x32xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, 0] [1, 32] [1, 1]
        : tensor<2x32xf32> to tensor<32xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %extracted_slice_0[0] [%arg4] [1]
        : tensor<?xf32> into tensor<32xf32>
    %0 = func.call @vf_recursive_slice_operand(%inserted_slice, %arg2, %arg3, %c0, %c64, %c32, %c1, %arg5)
        {hivm.vector_function, no_inline}
        : (tensor<32xf32>, tensor<2x32x64xf32>, index, index, index, index, index, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// Test: read-modify-write matching must take priority over passthrough.
//
// The VF return value directly flows from the sliced argument, so the
// passthrough / identity-like path can also match. However, because the call
// result is inserted back into the same source tensor with the same slice
// parameters, this is a read-modify-write update. The caller should use the
// full tensor returned by the VF directly, without extracting from that result
// and inserting it back into the old tensor.
module {
  func.func @vf_identity_slice(%arg0: tensor<32x64xf32>)
      -> tensor<32x64xf32> attributes {hivm.vector_function} {
    return %arg0 : tensor<32x64xf32>
  }

  // CHECK-LABEL: func @test_read_modify_write_priority_over_passthrough(
  // CHECK: %[[CALL:.*]] = call @vf_identity_slice(%arg0
  // CHECK-SAME: {hivm.vector_function}
  // CHECK-SAME: tensor<2x32x64xf32>
  // CHECK-NOT: tensor.extract_slice %[[CALL]]
  // CHECK-NOT: tensor.insert_slice
  // CHECK: return %[[CALL]]
  func.func @test_read_modify_write_priority_over_passthrough(
      %arg0: tensor<2x32x64xf32>, %iv: index)
      -> tensor<2x32x64xf32> {
    %slice = tensor.extract_slice %arg0[%iv, 0, 0]
        [1, 32, 64] [1, 1, 1]
        : tensor<2x32x64xf32> to tensor<32x64xf32>

    %x = func.call @vf_identity_slice(%slice) {hivm.vector_function}
        : (tensor<32x64xf32>) -> tensor<32x64xf32>

    %updated = tensor.insert_slice %x into %arg0[%iv, 0, 0]
        [1, 32, 64] [1, 1, 1]
        : tensor<32x64xf32> into tensor<2x32x64xf32>

    return %updated : tensor<2x32x64xf32>
  }
}