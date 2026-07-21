// RUN: bishengir-opt %s -hfusion-simplify-ops -split-input-file | FileCheck %s
//
//==============================================================================
// TEST SPECIFICATION: MergeConsecutiveCopiesPattern
//==============================================================================
//
// PURPOSE: Verify that MergeConsecutiveCopiesPattern recognizes two memref.copy
// ops from the same source with stride-2 reinterpret_casts and consecutive
// offsets (X, X+1), replacing them with a single contiguous copy followed by
// hfusion::DeinterleaveOp (channel 0/1).
//
// NOTE: Offsets must be dynamic (function arguments) to prevent the
// canonicalizer from folding `arith.addi %offset, 1` into a constant, which
// would break the pattern's structural check for offset+1.
//
// NOTE: Each test case is wrapped in `module attributes {hacc.target = ...}`
// declaring an Ascend950 target device, because MergeConsecutiveCopiesPattern
// produces hfusion::DeinterleaveOp which is an Ascend950-specific operation and
// the pattern is guarded by `hacc::utils::isAscend950(moduleOp)`.
//
// COVERAGE:
//   - basic_merge: Same source, stride 2, offsets X and X+1 (dynamic)
//   - with_scf_if_fill: Copies with scf.if fill blocks
//   - non_consecutive_offsets: Offsets X and X+2 (negative: no match)
//   - different_source: Different source memrefs (negative: no match)
//   - different_sizes: Different static sizes (negative: no match)
//   - single_copy_only: Only one copy (negative: no match)
//==============================================================================

// TEST_SPEC: basic_merge | Shape: 2048xf16 → 512xf16 each | Dynamic offset | Stride 2
//   Two stride-2 copies from same source with offsets %off, %off+1.
//   EXPECTED: Merged into one contiguous copy + hfusion.deinterleave channel<0>/<1>
// CHECK-LABEL: func @basic_merge(
// CHECK: memref.alloc
// CHECK-SAME: memref<1024xf16>
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strides: [1]
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: hfusion.deinterleave {{%.*}} channel<0> : tensor<1024xf16> -> tensor<512xf16>
// CHECK: hfusion.deinterleave {{%.*}} channel<1> : tensor<1024xf16> -> tensor<512xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @basic_merge(%arg0: memref<2048xf16>, %off: index) -> (tensor<512xf16>, tensor<512xf16>) {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index

    // First copy: offset %off, stride 2
    %rc0 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [512], strides: [2] : memref<2048xf16> to memref<512xf16, strided<[2], offset: ?>>
    %alloc0 = memref.alloc() : memref<512xf16>
    %sv0_src = memref.subview %rc0[0] [%c512] [1] : memref<512xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv0_dst = memref.subview %alloc0[0] [%c512] [1] : memref<512xf16> to memref<?xf16>
    memref.copy %sv0_src, %sv0_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<512xf16>

    // Second copy: offset %off + 1, stride 2
    %off_plus_1 = arith.addi %off, %c1 : index
    %rc1 = memref.reinterpret_cast %arg0 to offset: [%off_plus_1], sizes: [512], strides: [2] : memref<2048xf16> to memref<512xf16, strided<[2], offset: ?>>
    %alloc1 = memref.alloc() : memref<512xf16>
    %sv1_src = memref.subview %rc1[0] [%c512] [1] : memref<512xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv1_dst = memref.subview %alloc1[0] [%c512] [1] : memref<512xf16> to memref<?xf16>
    memref.copy %sv1_src, %sv1_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<512xf16>

    return %t0, %t1 : tensor<512xf16>, tensor<512xf16>
  }
}

// -----

// TEST_SPEC: with_scf_if_fill | Shape: 2048xf16 -> 512xf16 each | Dynamic offset | scf.if + linalg.fill
//   Copies with optional scf.if fill blocks. The fill should be replicated for
//   the merged alloc.
// EXPECTED: Merged into one copy + deinterleave, fill replicated in scf.if
// CHECK-LABEL: func @with_scf_if_fill(
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: hfusion.deinterleave {{%.*}} channel<0>
// CHECK: hfusion.deinterleave {{%.*}} channel<1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @with_scf_if_fill(%arg0: memref<2048xf16>, %off: index, %cond: i1) -> (tensor<512xf16>, tensor<512xf16>) {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %cst_0 = arith.constant 0.000000e+00 : f16

    %rc0 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [512], strides: [2] : memref<2048xf16> to memref<512xf16, strided<[2], offset: ?>>
    %alloc0 = memref.alloc() : memref<512xf16>
    scf.if %cond {
      linalg.fill ins(%cst_0 : f16) outs(%alloc0 : memref<512xf16>)
    }
    %sv0_src = memref.subview %rc0[0] [%c512] [1] : memref<512xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv0_dst = memref.subview %alloc0[0] [%c512] [1] : memref<512xf16> to memref<?xf16>
    memref.copy %sv0_src, %sv0_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<512xf16>

    %off_plus_1 = arith.addi %off, %c1 : index
    %rc1 = memref.reinterpret_cast %arg0 to offset: [%off_plus_1], sizes: [512], strides: [2] : memref<2048xf16> to memref<512xf16, strided<[2], offset: ?>>
    %alloc1 = memref.alloc() : memref<512xf16>
    scf.if %cond {
      linalg.fill ins(%cst_0 : f16) outs(%alloc1 : memref<512xf16>)
    }
    %sv1_src = memref.subview %rc1[0] [%c512] [1] : memref<512xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv1_dst = memref.subview %alloc1[0] [%c512] [1] : memref<512xf16> to memref<?xf16>
    memref.copy %sv1_src, %sv1_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<512xf16>

    return %t0, %t1 : tensor<512xf16>, tensor<512xf16>
  }
}

// -----

// TEST_SPEC: non_consecutive_offsets | Shape: 1024xf32 → 256xf32 each | Offsets X and X+2
//   Stride-2 copies from same source but gap=2 (not 1), so the addi doesn't
//   match the `const_val == 1` check.
// EXPECTED: Pattern does NOT match — no deinterleave created.
// CHECK-LABEL: func @non_consecutive_offsets(
// CHECK: memref.reinterpret_cast {{%.*}} to offset: {{.*}} sizes: [256], strides: [2]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @non_consecutive_offsets(%arg0: memref<1024xf32>, %off: index) -> (tensor<256xf32>, tensor<256xf32>) {
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index

    %rc0 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [256], strides: [2] : memref<1024xf32> to memref<256xf32, strided<[2], offset: ?>>
    %alloc0 = memref.alloc() : memref<256xf32>
    %sv0_src = memref.subview %rc0[0] [%c256] [1] : memref<256xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[2], offset: ?>>
    %sv0_dst = memref.subview %alloc0[0] [%c256] [1] : memref<256xf32> to memref<?xf32>
    memref.copy %sv0_src, %sv0_dst : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<256xf32>

    %off_plus_2 = arith.addi %off, %c2 : index
    %rc1 = memref.reinterpret_cast %arg0 to offset: [%off_plus_2], sizes: [256], strides: [2] : memref<1024xf32> to memref<256xf32, strided<[2], offset: ?>>
    %alloc1 = memref.alloc() : memref<256xf32>
    %sv1_src = memref.subview %rc1[0] [%c256] [1] : memref<256xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[2], offset: ?>>
    %sv1_dst = memref.subview %alloc1[0] [%c256] [1] : memref<256xf32> to memref<?xf32>
    memref.copy %sv1_src, %sv1_dst : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<256xf32>

    return %t0, %t1 : tensor<256xf32>, tensor<256xf32>
  }
}

// -----

// TEST_SPEC: different_source | Shape: 512xf16 -> 128xf16 each | Different base memrefs
//   Each copy uses a different source memref.
// EXPECTED: Pattern does NOT match - no deinterleave created.
// CHECK-LABEL: func @different_source(
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strides: [2]
// CHECK-NOT: hfusion.deinterleave
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @different_source(%arg0: memref<512xf16>, %arg1: memref<512xf16>, %off: index) -> (tensor<128xf16>, tensor<128xf16>) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index

    %rc0 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [128], strides: [2] : memref<512xf16> to memref<128xf16, strided<[2], offset: ?>>
    %alloc0 = memref.alloc() : memref<128xf16>
    %sv0_src = memref.subview %rc0[0] [%c128] [1] : memref<128xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv0_dst = memref.subview %alloc0[0] [%c128] [1] : memref<128xf16> to memref<?xf16>
    memref.copy %sv0_src, %sv0_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<128xf16>

    %off_plus_1 = arith.addi %off, %c1 : index
    %rc1 = memref.reinterpret_cast %arg1 to offset: [%off_plus_1], sizes: [128], strides: [2] : memref<512xf16> to memref<128xf16, strided<[2], offset: ?>>
    %alloc1 = memref.alloc() : memref<128xf16>
    %sv1_src = memref.subview %rc1[0] [%c128] [1] : memref<128xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv1_dst = memref.subview %alloc1[0] [%c128] [1] : memref<128xf16> to memref<?xf16>
    memref.copy %sv1_src, %sv1_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<128xf16>

    return %t0, %t1 : tensor<128xf16>, tensor<128xf16>
  }
}

// -----

// TEST_SPEC: different_sizes | Shape: 2048xf16 -> 512 and 256 | Stride 2, different N
//   Both stride-2 copies but with different output sizes (512 vs 256).
// EXPECTED: Pattern does NOT match - sizes must be equal.
// CHECK-LABEL: func @different_sizes(
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strides: [2]
// CHECK-NOT: hfusion.deinterleave
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @different_sizes(%arg0: memref<2048xf16>, %off: index) -> (tensor<512xf16>, tensor<256xf16>) {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index

    %rc0 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [512], strides: [2] : memref<2048xf16> to memref<512xf16, strided<[2], offset: ?>>
    %alloc0 = memref.alloc() : memref<512xf16>
    %sv0_src = memref.subview %rc0[0] [%c512] [1] : memref<512xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv0_dst = memref.subview %alloc0[0] [%c512] [1] : memref<512xf16> to memref<?xf16>
    memref.copy %sv0_src, %sv0_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<512xf16>

    %off_plus_1 = arith.addi %off, %c1 : index
    %rc1 = memref.reinterpret_cast %arg0 to offset: [%off_plus_1], sizes: [256], strides: [2] : memref<2048xf16> to memref<256xf16, strided<[2], offset: ?>>
    %alloc1 = memref.alloc() : memref<256xf16>
    %sv1_src = memref.subview %rc1[0] [%c256] [1] : memref<256xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv1_dst = memref.subview %alloc1[0] [%c256] [1] : memref<256xf16> to memref<?xf16>
    memref.copy %sv1_src, %sv1_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<256xf16>

    return %t0, %t1 : tensor<512xf16>, tensor<256xf16>
  }
}

// -----

// TEST_SPEC: single_copy_only | Shape: 256xf16 -> 32xf16 | Only one copy
//   Just a single stride-2 copy with no sibling.
// EXPECTED: Pattern does NOT match - no sibling found.
// CHECK-LABEL: func @single_copy_only(
// CHECK: memref.copy
// CHECK-NOT: hfusion.deinterleave
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @single_copy_only(%arg0: memref<256xf16>, %off: index) -> tensor<32xf16> {
    %c32 = arith.constant 32 : index

    %rc0 = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [32], strides: [2] : memref<256xf16> to memref<32xf16, strided<[2], offset: ?>>
    %alloc0 = memref.alloc() : memref<32xf16>
    %sv0_src = memref.subview %rc0[0] [%c32] [1] : memref<32xf16, strided<[2], offset: ?>> to memref<?xf16, strided<[2], offset: ?>>
    %sv0_dst = memref.subview %alloc0[0] [%c32] [1] : memref<32xf16> to memref<?xf16>
    memref.copy %sv0_src, %sv0_dst : memref<?xf16, strided<[2], offset: ?>> to memref<?xf16>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32xf16>

    return %t0 : tensor<32xf16>
  }
}
