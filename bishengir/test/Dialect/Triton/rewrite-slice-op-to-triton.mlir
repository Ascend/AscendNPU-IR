// RUN: bishengir-opt -split-input-file -verify-diagnostics %s -rewrite-slice-op-to-triton | FileCheck %s

// 2D, axis 0, S = 1, r = 0 -> m = 0 -> bit_0 = 0, bit_1 = 0  (LHS twice).
// CHECK-LABEL: @axis0_s1_r0
// CHECK-SAME: (%[[SRC:.*]]: tensor<4x64xbf16>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[SRC]] : tensor<4x64xbf16> -> tensor<4x1x64xbf16>
// CHECK-NEXT: %[[T:.*]] = tt.trans %[[R0]] {order = array<i32: 1, 2, 0>} : tensor<4x1x64xbf16> -> tensor<1x64x4xbf16>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T]] : tensor<1x64x4xbf16> -> tensor<1x64x2x2xbf16>
// CHECK-NEXT: %[[S0L:.*]], %{{.*}} = tt.split %[[R1]] : tensor<1x64x2x2xbf16> -> tensor<1x64x2xbf16>
// CHECK-NEXT: %[[S1L:.*]], %{{.*}} = tt.split %[[S0L]] : tensor<1x64x2xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT: return %[[S1L]]
func.func @axis0_s1_r0(%src: tensor<4x64xbf16>) -> tensor<1x64xbf16> {
  %0 = tensor.extract_slice %src[0, 0] [1, 64] [1, 1] : tensor<4x64xbf16> to tensor<1x64xbf16>
  return %0 : tensor<1x64xbf16>
}

// -----

// 2D, axis 0, S = 2, r = 4 -> m = r/S = 2 = 0b10 -> bit_0 = 0 (LHS), bit_1 = 1 (RHS).
// N/S = 4 -> 2 splits.
// CHECK-LABEL: @axis0_s2_r4
// CHECK-SAME: (%[[SRC:.*]]: tensor<8x16xbf16>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[SRC]] : tensor<8x16xbf16> -> tensor<4x2x16xbf16>
// CHECK-NEXT: %[[T:.*]] = tt.trans %[[R0]] {order = array<i32: 1, 2, 0>} : tensor<4x2x16xbf16> -> tensor<2x16x4xbf16>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T]] : tensor<2x16x4xbf16> -> tensor<2x16x2x2xbf16>
// CHECK-NEXT: %[[S0L:.*]], %{{.*}} = tt.split %[[R1]] : tensor<2x16x2x2xbf16> -> tensor<2x16x2xbf16>
// CHECK-NEXT: %{{.*}}, %[[S1R:.*]] = tt.split %[[S0L]] : tensor<2x16x2xbf16> -> tensor<2x16xbf16>
// CHECK-NEXT: return %[[S1R]]
func.func @axis0_s2_r4(%src: tensor<8x16xbf16>) -> tensor<2x16xbf16> {
  %0 = tensor.extract_slice %src[4, 0] [2, 16] [1, 1] : tensor<8x16xbf16> to tensor<2x16xbf16>
  return %0 : tensor<2x16xbf16>
}

// -----

// 2D, axis 1, S = 4, r = 8 -> m = 2 = 0b10.  N/S = 4 -> 2 splits.
// CHECK-LABEL: @axis1_s4_r8
// CHECK-SAME: (%[[SRC:.*]]: tensor<8x16xf32>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[SRC]] : tensor<8x16xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[T:.*]] = tt.trans %[[R0]] {order = array<i32: 0, 2, 1>} : tensor<8x4x4xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T]] : tensor<8x4x4xf32> -> tensor<8x4x2x2xf32>
// CHECK-NEXT: %[[S0L:.*]], %{{.*}} = tt.split %[[R1]] : tensor<8x4x2x2xf32> -> tensor<8x4x2xf32>
// CHECK-NEXT: %{{.*}}, %[[S1R:.*]] = tt.split %[[S0L]] : tensor<8x4x2xf32> -> tensor<8x4xf32>
// CHECK-NEXT: return %[[S1R]]
func.func @axis1_s4_r8(%src: tensor<8x16xf32>) -> tensor<8x4xf32> {
  %0 = tensor.extract_slice %src[0, 8] [8, 4] [1, 1] : tensor<8x16xf32> to tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

// 1D, axis 0, S = 1, r = 2.
// CHECK-LABEL: @rank1_s1
// CHECK-SAME: (%[[SRC:.*]]: tensor<4xbf16>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[SRC]] : tensor<4xbf16> -> tensor<4x1xbf16>
// CHECK-NEXT: %[[T:.*]] = tt.trans %[[R0]] {order = array<i32: 1, 0>} : tensor<4x1xbf16> -> tensor<1x4xbf16>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T]] : tensor<1x4xbf16> -> tensor<1x2x2xbf16>
// CHECK-NEXT: %[[S0L:.*]], %{{.*}} = tt.split %[[R1]] : tensor<1x2x2xbf16> -> tensor<1x2xbf16>
// CHECK-NEXT: %{{.*}}, %[[S1R:.*]] = tt.split %[[S0L]] : tensor<1x2xbf16> -> tensor<1xbf16>
// CHECK-NEXT: return %[[S1R]]
func.func @rank1_s1(%src: tensor<4xbf16>) -> tensor<1xbf16> {
  %0 = tensor.extract_slice %src[2] [1] [1] : tensor<4xbf16> to tensor<1xbf16>
  return %0 : tensor<1xbf16>
}

// -----

// 1D, axis 0, S = 2, r = 4.  m = 2 = 0b10 -> 2 splits.
// CHECK-LABEL: @rank1_s2
// CHECK-SAME: (%[[SRC:.*]]: tensor<8xbf16>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[SRC]] : tensor<8xbf16> -> tensor<4x2xbf16>
// CHECK-NEXT: %[[T:.*]] = tt.trans %[[R0]] {order = array<i32: 1, 0>} : tensor<4x2xbf16> -> tensor<2x4xbf16>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T]] : tensor<2x4xbf16> -> tensor<2x2x2xbf16>
// CHECK-NEXT: %[[S0L:.*]], %{{.*}} = tt.split %[[R1]] : tensor<2x2x2xbf16> -> tensor<2x2xbf16>
// CHECK-NEXT: %{{.*}}, %[[S1R:.*]] = tt.split %[[S0L]] : tensor<2x2xbf16> -> tensor<2xbf16>
// CHECK-NEXT: return %[[S1R]]
func.func @rank1_s2(%src: tensor<8xbf16>) -> tensor<2xbf16> {
  %0 = tensor.extract_slice %src[4] [2] [1] : tensor<8xbf16> to tensor<2xbf16>
  return %0 : tensor<2xbf16>
}

// -----

// 3D, axis 1 (middle), S = 2, r = 2 -> m = 1 -> bit_0 = 1 (RHS).  N/S = 2 -> 1 split.
// Step 3's reshape would be <8x2x16x2> -> <8x2x16x2> (identity), so it is
// folded away by tt.reshape's no-op folder.
// CHECK-LABEL: @rank3_axis1_s2
// CHECK-SAME: (%[[SRC:.*]]: tensor<8x4x16xf32>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[SRC]] : tensor<8x4x16xf32> -> tensor<8x2x2x16xf32>
// CHECK-NEXT: %[[T:.*]] = tt.trans %[[R0]] {order = array<i32: 0, 2, 3, 1>} : tensor<8x2x2x16xf32> -> tensor<8x2x16x2xf32>
// CHECK-NEXT: %{{.*}}, %[[S0R:.*]] = tt.split %[[T]] : tensor<8x2x16x2xf32> -> tensor<8x2x16xf32>
// CHECK-NEXT: return %[[S0R]]
func.func @rank3_axis1_s2(%src: tensor<8x4x16xf32>) -> tensor<8x2x16xf32> {
  %0 = tensor.extract_slice %src[0, 2, 0] [8, 2, 16] [1, 1, 1] : tensor<8x4x16xf32> to tensor<8x2x16xf32>
  return %0 : tensor<8x2x16xf32>
}

// -----

// Source/result shapes match (S == src.dim at the would-be axis) -> no
// differing axis -> forward source.
// CHECK-LABEL: @noop
// CHECK-SAME: (%[[SRC:.*]]: tensor<1x4x128xbf16>)
// CHECK-NOT:  tt.trans
// CHECK-NOT:  tt.split
// CHECK:      return %[[SRC]]
func.func @noop(%src: tensor<1x4x128xbf16>) -> tensor<1x4x128xbf16> {
  %0 = tensor.extract_slice %src[0, 0, 0] [1, 4, 128] [1, 1, 1] : tensor<1x4x128xbf16> to tensor<1x4x128xbf16>
  return %0 : tensor<1x4x128xbf16>
}

// -----

// Multi-axis slicing must error out.
func.func @bad_multi_axis(%src: tensor<4x4x16xbf16>) -> tensor<1x1x16xbf16> {
  // expected-error @+2 {{only single-axis slicing is supported}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.extract_slice %src[1, 2, 0] [1, 1, 16] [1, 1, 1] : tensor<4x4x16xbf16> to tensor<1x1x16xbf16>
  return %0 : tensor<1x1x16xbf16>
}

// -----

// Non-power-of-two source dim must error out.
func.func @bad_non_pow2(%src: tensor<3x64xbf16>) -> tensor<1x64xbf16> {
  // expected-error @+2 {{must be a power of two}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.extract_slice %src[0, 0] [1, 64] [1, 1] : tensor<3x64xbf16> to tensor<1x64xbf16>
  return %0 : tensor<1x64xbf16>
}

// -----

// Non-power-of-two slice size must error out.  Source <8x16> -> result
// <3x16> means S = 3, not a power of two.
func.func @bad_non_pow2_size(%src: tensor<8x16xbf16>) -> tensor<3x16xbf16> {
  // expected-error @+2 {{indexed-axis size on the small tensor must be a power of two}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.extract_slice %src[0, 0] [3, 16] [1, 1] : tensor<8x16xbf16> to tensor<3x16xbf16>
  return %0 : tensor<3x16xbf16>
}

// -----

// Offset not a multiple of size must error out.  S = 2 but r = 3.
func.func @bad_unaligned_offset(%src: tensor<8x16xbf16>) -> tensor<2x16xbf16> {
  // expected-error @+2 {{must be a multiple of size 2}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.extract_slice %src[3, 0] [2, 16] [1, 1] : tensor<8x16xbf16> to tensor<2x16xbf16>
  return %0 : tensor<2x16xbf16>
}

// -----

// Non-zero offset on a non-indexed axis must error out.
func.func @bad_offset_other_axis(%src: tensor<4x64xbf16>) -> tensor<1x64xbf16> {
  // expected-error @+2 {{must be 0}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.extract_slice %src[0, 4] [1, 64] [1, 1] : tensor<4x64xbf16> to tensor<1x64xbf16>
  return %0 : tensor<1x64xbf16>
}

// -----

//===----------------------------------------------------------------------===//
// insert_slice cases
//===----------------------------------------------------------------------===//

// 2D, axis 0, S = 2, r = 4 -> m = 2 = 0b10.  Split chain runs identically to
// the matching extract_slice case; the leaf chunk is discarded and the
// source operand takes its place before the joins reassemble the dest.
// CHECK-LABEL: @insert_axis0_s2_r4
// CHECK-SAME: (%[[SRC:.*]]: tensor<2x16xbf16>, %[[DST:.*]]: tensor<8x16xbf16>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[DST]] : tensor<8x16xbf16> -> tensor<4x2x16xbf16>
// CHECK-NEXT: %[[T0:.*]] = tt.trans %[[R0]] {order = array<i32: 1, 2, 0>} : tensor<4x2x16xbf16> -> tensor<2x16x4xbf16>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T0]] : tensor<2x16x4xbf16> -> tensor<2x16x2x2xbf16>
// CHECK-NEXT: %[[S0L:.*]], %[[S0R:.*]] = tt.split %[[R1]] : tensor<2x16x2x2xbf16> -> tensor<2x16x2xbf16>
// CHECK-NEXT: %[[S1L:.*]], %{{.*}} = tt.split %[[S0L]] : tensor<2x16x2xbf16> -> tensor<2x16xbf16>
//   Joins: bit_1 = 1 -> cur was RHS, join(other=S1L, src); bit_0 = 0 -> cur was LHS, join(cur, other=S0R).
// CHECK-NEXT: %[[J0:.*]] = tt.join %[[S1L]], %[[SRC]] : tensor<2x16xbf16> -> tensor<2x16x2xbf16>
// CHECK-NEXT: %[[J1:.*]] = tt.join %[[J0]], %[[S0R]] : tensor<2x16x2xbf16> -> tensor<2x16x2x2xbf16>
// CHECK-NEXT: %[[R2:.*]] = tt.reshape %[[J1]] : tensor<2x16x2x2xbf16> -> tensor<2x16x4xbf16>
// CHECK-NEXT: %[[T1:.*]] = tt.trans %[[R2]] {order = array<i32: 2, 0, 1>} : tensor<2x16x4xbf16> -> tensor<4x2x16xbf16>
// CHECK-NEXT: %[[R3:.*]] = tt.reshape %[[T1]] : tensor<4x2x16xbf16> -> tensor<8x16xbf16>
// CHECK-NEXT: return %[[R3]]
func.func @insert_axis0_s2_r4(%src: tensor<2x16xbf16>, %dst: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
  %0 = tensor.insert_slice %src into %dst[4, 0] [2, 16] [1, 1] : tensor<2x16xbf16> into tensor<8x16xbf16>
  return %0 : tensor<8x16xbf16>
}

// -----

// 2D, axis 1 (innermost), S = 4, r = 8 -> m = 2 = 0b10.
// CHECK-LABEL: @insert_axis1_s4_r8
// CHECK-SAME: (%[[SRC:.*]]: tensor<8x4xf32>, %[[DST:.*]]: tensor<8x16xf32>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[DST]] : tensor<8x16xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[T0:.*]] = tt.trans %[[R0]] {order = array<i32: 0, 2, 1>} : tensor<8x4x4xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[R1:.*]] = tt.reshape %[[T0]] : tensor<8x4x4xf32> -> tensor<8x4x2x2xf32>
// CHECK-NEXT: %[[S0L:.*]], %[[S0R:.*]] = tt.split %[[R1]] : tensor<8x4x2x2xf32> -> tensor<8x4x2xf32>
// CHECK-NEXT: %[[S1L:.*]], %{{.*}} = tt.split %[[S0L]] : tensor<8x4x2xf32> -> tensor<8x4xf32>
// CHECK-NEXT: %[[J0:.*]] = tt.join %[[S1L]], %[[SRC]] : tensor<8x4xf32> -> tensor<8x4x2xf32>
// CHECK-NEXT: %[[J1:.*]] = tt.join %[[J0]], %[[S0R]] : tensor<8x4x2xf32> -> tensor<8x4x2x2xf32>
// CHECK-NEXT: %[[R2:.*]] = tt.reshape %[[J1]] : tensor<8x4x2x2xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[T1:.*]] = tt.trans %[[R2]] {order = array<i32: 0, 2, 1>} : tensor<8x4x4xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[R3:.*]] = tt.reshape %[[T1]] : tensor<8x4x4xf32> -> tensor<8x16xf32>
// CHECK-NEXT: return %[[R3]]
func.func @insert_axis1_s4_r8(%src: tensor<8x4xf32>, %dst: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = tensor.insert_slice %src into %dst[0, 8] [8, 4] [1, 1] : tensor<8x4xf32> into tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// 3D, axis 1 (middle), S = 2, r = 2 -> m = 1 (one split + one join).
// Both the forward and the inverse step-3 reshapes are <8x2x16x2> ->
// <8x2x16x2> identities and are folded away by tt.reshape's no-op folder.
// CHECK-LABEL: @insert_rank3_axis1_s2
// CHECK-SAME: (%[[SRC:.*]]: tensor<8x2x16xf32>, %[[DST:.*]]: tensor<8x4x16xf32>)
// CHECK:      %[[R0:.*]] = tt.reshape %[[DST]] : tensor<8x4x16xf32> -> tensor<8x2x2x16xf32>
// CHECK-NEXT: %[[T0:.*]] = tt.trans %[[R0]] {order = array<i32: 0, 2, 3, 1>} : tensor<8x2x2x16xf32> -> tensor<8x2x16x2xf32>
// CHECK-NEXT: %[[S0L:.*]], %{{.*}} = tt.split %[[T0]] : tensor<8x2x16x2xf32> -> tensor<8x2x16xf32>
//   bit_0 = 1 -> cur was RHS, join(other=S0L, src).
// CHECK-NEXT: %[[J0:.*]] = tt.join %[[S0L]], %[[SRC]] : tensor<8x2x16xf32> -> tensor<8x2x16x2xf32>
// CHECK-NEXT: %[[T1:.*]] = tt.trans %[[J0]] {order = array<i32: 0, 3, 1, 2>} : tensor<8x2x16x2xf32> -> tensor<8x2x2x16xf32>
// CHECK-NEXT: %[[R3:.*]] = tt.reshape %[[T1]] : tensor<8x2x2x16xf32> -> tensor<8x4x16xf32>
// CHECK-NEXT: return %[[R3]]
func.func @insert_rank3_axis1_s2(%src: tensor<8x2x16xf32>, %dst: tensor<8x4x16xf32>) -> tensor<8x4x16xf32> {
  %0 = tensor.insert_slice %src into %dst[0, 2, 0] [8, 2, 16] [1, 1, 1] : tensor<8x2x16xf32> into tensor<8x4x16xf32>
  return %0 : tensor<8x4x16xf32>
}

// -----

// Source/dest shapes match -> no differing axis -> result = source.
// CHECK-LABEL: @insert_noop
// CHECK-SAME: (%[[SRC:.*]]: tensor<1x4x128xbf16>, %{{.*}}: tensor<1x4x128xbf16>)
// CHECK-NOT:  tt.split
// CHECK-NOT:  tt.join
// CHECK:      return %[[SRC]]
func.func @insert_noop(%src: tensor<1x4x128xbf16>, %dst: tensor<1x4x128xbf16>) -> tensor<1x4x128xbf16> {
  %0 = tensor.insert_slice %src into %dst[0, 0, 0] [1, 4, 128] [1, 1, 1] : tensor<1x4x128xbf16> into tensor<1x4x128xbf16>
  return %0 : tensor<1x4x128xbf16>
}

// -----

// Multi-axis insert must error out.
func.func @insert_bad_multi_axis(%src: tensor<1x1x16xbf16>, %dst: tensor<4x4x16xbf16>) -> tensor<4x4x16xbf16> {
  // expected-error @+2 {{only single-axis slicing is supported}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.insert_slice %src into %dst[1, 2, 0] [1, 1, 16] [1, 1, 1] : tensor<1x1x16xbf16> into tensor<4x4x16xbf16>
  return %0 : tensor<4x4x16xbf16>
}

// -----

// Unaligned offset must error out.  S = 2 but r = 3.
func.func @insert_bad_unaligned_offset(%src: tensor<2x16xbf16>, %dst: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
  // expected-error @+2 {{must be a multiple of size 2}}
  // expected-error @+1 {{Unsupported tensor slicing operations found in the SIMT kernel}}
  %0 = tensor.insert_slice %src into %dst[3, 0] [2, 16] [1, 1] : tensor<2x16xbf16> into tensor<8x16xbf16>
  return %0 : tensor<8x16xbf16>
}

// -----

// 1-D dynamic offset.
// CHECK-LABEL: @rank1_dyn_offset
// CHECK-SAME:  (%[[SRC:.*]]: tensor<8xf32>, %[[OFF:.*]]: index)
// CHECK:       %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NEXT:  %[[OFFI32:.*]] = arith.index_cast %[[OFF]] : index to i32
// CHECK-NEXT:  %[[SPLAT:.*]] = tt.splat %[[OFFI32]] : i32 -> tensor<2xi32>
// CHECK-NEXT:  %[[IDX:.*]] = arith.addi %[[RANGE]], %[[SPLAT]] : tensor<2xi32>
// CHECK-NEXT:  %[[G:.*]] = tt.gather %[[SRC]][%[[IDX]]] {axis = 0 : i32} : (tensor<8xf32>, tensor<2xi32>) -> tensor<2xf32>
// CHECK-NEXT:  return %[[G]]
func.func @rank1_dyn_offset(%src: tensor<8xf32>, %off: index) -> tensor<2xf32> {
  %0 = tensor.extract_slice %src[%off] [2] [1] : tensor<8xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// 2-D, axis 0 dynamic.  Index tensor is reshaped to <2x1> and broadcast to
// <2x16>; the non-sliced axis passes through (full size, zero offset).
// CHECK-LABEL: @rank2_axis0_dyn_offset
// CHECK-SAME:  (%[[SRC:.*]]: tensor<8x16xf32>, %[[OFF:.*]]: index)
// CHECK:       %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NEXT:  %[[OFFI32:.*]] = arith.index_cast %[[OFF]] : index to i32
// CHECK-NEXT:  %[[SPLAT:.*]] = tt.splat %[[OFFI32]] : i32 -> tensor<2xi32>
// CHECK-NEXT:  %[[IDX1D:.*]] = arith.addi %[[RANGE]], %[[SPLAT]] : tensor<2xi32>
// CHECK-NEXT:  %[[IDX2D:.*]] = tt.reshape %[[IDX1D]] : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-NEXT:  %[[BCAST:.*]] = tt.broadcast %[[IDX2D]] : tensor<2x1xi32> -> tensor<2x16xi32>
// CHECK-NEXT:  %[[G:.*]] = tt.gather %[[SRC]][%[[BCAST]]] {axis = 0 : i32} : (tensor<8x16xf32>, tensor<2x16xi32>) -> tensor<2x16xf32>
// CHECK-NEXT:  return %[[G]]
func.func @rank2_axis0_dyn_offset(%src: tensor<8x16xf32>, %off: index) -> tensor<2x16xf32> {
  %0 = tensor.extract_slice %src[%off, 0] [2, 16] [1, 1] : tensor<8x16xf32> to tensor<2x16xf32>
  return %0 : tensor<2x16xf32>
}

// -----

// 2-D, axis 1 (innermost) dynamic.  Reshape to <1x4>, broadcast to <8x4>.
// CHECK-LABEL: @rank2_axis1_dyn_offset
// CHECK-SAME:  (%[[SRC:.*]]: tensor<8x16xf32>, %[[OFF:.*]]: index)
// CHECK:       %[[RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NEXT:  %[[OFFI32:.*]] = arith.index_cast %[[OFF]] : index to i32
// CHECK-NEXT:  %[[SPLAT:.*]] = tt.splat %[[OFFI32]] : i32 -> tensor<4xi32>
// CHECK-NEXT:  %[[IDX1D:.*]] = arith.addi %[[RANGE]], %[[SPLAT]] : tensor<4xi32>
// CHECK-NEXT:  %[[IDX2D:.*]] = tt.reshape %[[IDX1D]] : tensor<4xi32> -> tensor<1x4xi32>
// CHECK-NEXT:  %[[BCAST:.*]] = tt.broadcast %[[IDX2D]] : tensor<1x4xi32> -> tensor<8x4xi32>
// CHECK-NEXT:  %[[G:.*]] = tt.gather %[[SRC]][%[[BCAST]]] {axis = 1 : i32} : (tensor<8x16xf32>, tensor<8x4xi32>) -> tensor<8x4xf32>
// CHECK-NEXT:  return %[[G]]
func.func @rank2_axis1_dyn_offset(%src: tensor<8x16xf32>, %off: index) -> tensor<8x4xf32> {
  %0 = tensor.extract_slice %src[0, %off] [8, 4] [1, 1] : tensor<8x16xf32> to tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

