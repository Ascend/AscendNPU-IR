// RUN: bishengir-opt --arith-vector-mask-analyze -split-input-file %s | FileCheck %s

// Issue 1069: computing the low 32 lanes of the final add needs all 64 lanes
// of the first add. Keep masks at the top so dominance cleanup cannot hide
// incorrect propagation through the deinterleave/concatenation chain.
// CHECK-LABEL: func.func @deinterleave_concat
// CHECK: %[[MASK:.*]] = vector.constant_mask [32]
// CHECK-NEXT: annotation.mark %[[MASK]] {mask_op_idx = 0 : i32}
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: %[[FINAL:.*]] = arith.addf
// CHECK-NEXT: annotation.mark %[[FINAL]] {reached_mask_ops_idx = 0 : i32}
func.func @deinterleave_concat(
    %src: memref<128xf32, #hivm.address_space<ub>>,
    %dst: memref<32xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [32] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.0 : f32
  %lo = vector.transfer_read %src[%c0], %zero {in_bounds = [true]} : memref<128xf32, #hivm.address_space<ub>>, vector<64xf32>
  %hi = vector.transfer_read %src[%c64], %zero {in_bounds = [true]} : memref<128xf32, #hivm.address_space<ub>>, vector<64xf32>
  %concat0 = builtin.unrealized_conversion_cast %lo, %hi : vector<64xf32>, vector<64xf32> to vector<128xf32>
  %even0, %odd0 = vector.deinterleave %concat0 : vector<128xf32> -> vector<64xf32>
  %upstream = arith.addf %even0, %odd0 : vector<64xf32>
  %concat1 = builtin.unrealized_conversion_cast %upstream, %upstream : vector<64xf32>, vector<64xf32> to vector<128xf32>
  %even1, %odd1 = vector.deinterleave %concat1 : vector<128xf32> -> vector<64xf32>
  %final = arith.addf %even1, %odd1 : vector<64xf32>
  vector.transfer_write %final, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

// -----

// Interleave changes lane positions and must not itself be marked.
// CHECK-LABEL: func.func @interleave
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @interleave(%dst: memref<32xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [32] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<32xf32>
  %upstream = arith.addf %one, %one : vector<32xf32>
  %interleaved = vector.interleave %upstream, %upstream : vector<32xf32> -> vector<64xf32>
  vector.transfer_write %interleaved, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @shuffle_permutation
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @shuffle_permutation(%dst: memref<2xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2] : vector<4xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<4xf32>
  %upstream = arith.addf %one, %one : vector<4xf32>
  %shuffled = vector.shuffle %upstream, %upstream [2, 1, 0, 3] : vector<4xf32>, vector<4xf32>
  vector.transfer_write %shuffled, %dst[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}

// -----

// Identity shuffle must retain propagation, including the upstream producer.
// CHECK-LABEL: func.func @shuffle_identity
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NEXT: annotation.mark %[[UPSTREAM]] {reached_mask_ops_idx = 0 : i32}
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle
// CHECK-NEXT: annotation.mark %[[SHUFFLE]] {reached_mask_ops_idx = 0 : i32}
func.func @shuffle_identity(%dst: memref<2xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2] : vector<4xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<4xf32>
  %upstream = arith.addf %one, %one : vector<4xf32>
  %shuffled = vector.shuffle %upstream, %upstream [0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
  vector.transfer_write %shuffled, %dst[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}

// -----

// Single-input casts with the same flat element count preserve lane identity.
// CHECK-LABEL: func.func @cast_flat_identity
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NEXT: annotation.mark %[[UPSTREAM]] {reached_mask_ops_idx = 0 : i32}
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: annotation.mark %[[CAST]] {reached_mask_ops_idx = 0 : i32}
func.func @cast_flat_identity(%dst: memref<32xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [32] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<2x32xf32>
  %upstream = arith.addf %one, %one : vector<2x32xf32>
  %flat = builtin.unrealized_conversion_cast %upstream : vector<2x32xf32> to vector<64xf32>
  vector.transfer_write %flat, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @shape_cast_flat_identity
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NEXT: annotation.mark %[[UPSTREAM]] {reached_mask_ops_idx = 0 : i32}
// CHECK: %[[CAST:.*]] = vector.shape_cast
// CHECK-NEXT: annotation.mark %[[CAST]] {reached_mask_ops_idx = 0 : i32}
func.func @shape_cast_flat_identity(%dst: memref<32xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [32] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<2x32xf32>
  %upstream = arith.addf %one, %one : vector<2x32xf32>
  %flat = vector.shape_cast %upstream : vector<2x32xf32> to vector<64xf32>
  vector.transfer_write %flat, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

// -----

// The first 16 result lanes require source lanes 32..47, not 0..15.
// CHECK-LABEL: func.func @extract_strided_slice_offset
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @extract_strided_slice_offset(%dst: memref<16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [16] : vector<32xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<64xf32>
  %upstream = arith.addf %one, %one : vector<64xf32>
  %slice = vector.extract_strided_slice %upstream {offsets = [32], sizes = [32], strides = [1]} : vector<64xf32> to vector<32xf32>
  vector.transfer_write %slice, %dst[%c0], %mask {in_bounds = [true]} : vector<32xf32>, memref<16xf32, #hivm.address_space<ub>>
  return
}

// -----

// Exercise the multi-input cast guard without a deinterleave hiding it.
// CHECK-LABEL: func.func @cast_concat
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @cast_concat(%dst: memref<16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [16] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<32xf32>
  %upstream = arith.addf %one, %one : vector<32xf32>
  %concat = builtin.unrealized_conversion_cast %upstream, %upstream : vector<32xf32>, vector<32xf32> to vector<64xf32>
  vector.transfer_write %concat, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
  return
}

// -----

// A single operand alone is insufficient if the flat element count changes.
// CHECK-LABEL: func.func @cast_element_count_change
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @cast_element_count_change(%dst: memref<16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [16] : vector<32xi1>
  %c0 = arith.constant 0 : index
  %one = arith.constant dense<1.0> : vector<64xf32>
  %upstream = arith.addf %one, %one : vector<64xf32>
  %cast = builtin.unrealized_conversion_cast %upstream : vector<64xf32> to vector<32xf32>
  vector.transfer_write %cast, %dst[%c0], %mask {in_bounds = [true]} : vector<32xf32>, memref<16xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @extract_offset(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @extract_offset(
    %input: vector<2x32xf32>, %dst: memref<16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [16] : vector<32xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<2x32xf32>
  %mapped = vector.extract %upstream[1] : vector<32xf32> from vector<2x32xf32>
  vector.transfer_write %mapped, %dst[%c0], %mask {in_bounds = [true]} : vector<32xf32>, memref<16xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @insert_offset(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @insert_offset(
    %input: vector<32xf32>, %base: vector<2x32xf32>, %dst: memref<2x16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2, 16] : vector<2x32xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<32xf32>
  %mapped = vector.insert %upstream, %base[1] : vector<32xf32> into vector<2x32xf32>
  vector.transfer_write %mapped, %dst[%c0, %c0], %mask {in_bounds = [true, true]} : vector<2x32xf32>, memref<2x16xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @insert_strided_slice_offset(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @insert_strided_slice_offset(
    %input: vector<32xf32>, %base: vector<64xf32>, %dst: memref<32xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [32] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<32xf32>
  %mapped = vector.insert_strided_slice %upstream, %base {offsets = [16], strides = [1]} : vector<32xf32> into vector<64xf32>
  vector.transfer_write %mapped, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @broadcast(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @broadcast(
    %input: vector<1xf32>, %dst: memref<32xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [32] : vector<64xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<1xf32>
  %mapped = vector.broadcast %upstream : vector<1xf32> to vector<64xf32>
  vector.transfer_write %mapped, %dst[%c0], %mask {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @transpose(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @transpose(
    %input: vector<2x4xf32>, %dst: memref<4x1xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [4, 1] : vector<4x2xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<2x4xf32>
  %mapped = vector.transpose %upstream, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  vector.transfer_write %mapped, %dst[%c0, %c0], %mask {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x1xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @splat(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @splat(
    %input: f32, %dst: memref<2xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2] : vector<4xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : f32
  %mapped = vector.splat %upstream : vector<4xf32>
  vector.transfer_write %mapped, %dst[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}

// -----

// Lane offsets, replication and transposition also stop mask propagation.
// CHECK-LABEL: func.func @from_elements(
// CHECK: %[[UPSTREAM:.*]] = arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.transfer_write
func.func @from_elements(
    %input: f32, %dst: memref<2xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2] : vector<4xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : f32
  %mapped = vector.from_elements %upstream, %upstream, %upstream, %upstream : vector<4xf32>
  vector.transfer_write %mapped, %dst[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}

// -----

// Each output lane requires every source lane along the reduction dimension.
// CHECK-LABEL: func.func @multi_dim_reduction(
// CHECK: arith.addf
// CHECK-NOT: annotation.mark
// CHECK: %[[FINAL:.*]] = arith.addf
// CHECK-NEXT: annotation.mark %[[FINAL]]
// CHECK: vector.transfer_write
func.func @multi_dim_reduction(
    %input: vector<4x8xf32>, %acc: vector<4xf32>,
    %dst: memref<2xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2] : vector<4xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<4x8xf32>
  %reduced = vector.multi_reduction <add>, %upstream, %acc [1] : vector<4x8xf32> to vector<4xf32>
  %final = arith.addf %reduced, %acc : vector<4xf32>
  vector.transfer_write %final, %dst[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}

// -----

// The scalar-to-one-lane cast permits traversal to the extraction itself,
// independently of the existing splat/broadcast barriers.
// CHECK-LABEL: func.func @extract_element_dynamic(
// CHECK: arith.addf
// CHECK-NOT: annotation.mark
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: annotation.mark %[[CAST]]
// CHECK: vector.transfer_write
func.func @extract_element_dynamic(
    %input: vector<4xf32>, %position: index,
    %dst: memref<1xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [1] : vector<1xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<4xf32>
  %element = vector.extractelement %upstream[%position : index] : vector<4xf32>
  %cast = builtin.unrealized_conversion_cast %element : f32 to vector<1xf32>
  vector.transfer_write %cast, %dst[%c0], %mask {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #hivm.address_space<ub>>
  return
}

// -----

// A vector mask cannot be propagated identically to the destination vector,
// inserted scalar and dynamic position operands.
// CHECK-LABEL: func.func @insert_element_dynamic(
// CHECK: arith.addf
// CHECK-NOT: annotation.mark
// CHECK: arith.addf
// CHECK-NOT: annotation.mark
// CHECK: vector.insertelement
// CHECK-NOT: annotation.mark
// CHECK: %[[FINAL:.*]] = arith.addf
// CHECK-NEXT: annotation.mark %[[FINAL]]
// CHECK: vector.transfer_write
func.func @insert_element_dynamic(
    %input: vector<4xf32>, %scalar: f32, %position: index,
    %dst: memref<2xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %mask = vector.constant_mask [2] : vector<4xi1>
  %c0 = arith.constant 0 : index
  %upstream = arith.addf %input, %input : vector<4xf32>
  %element = arith.addf %scalar, %scalar : f32
  %inserted = vector.insertelement %element, %upstream[%position : index] : vector<4xf32>
  %final = arith.addf %inserted, %input : vector<4xf32>
  vector.transfer_write %final, %dst[%c0], %mask {in_bounds = [true]} : vector<4xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}
