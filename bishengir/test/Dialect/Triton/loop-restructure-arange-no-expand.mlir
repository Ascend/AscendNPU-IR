// RUN: bishengir-opt -split-input-file %s -loop-restructure-arange-optimization | FileCheck %s

// A mask bound is only an access width when it constrains the same axis that
// the pass will rewrite. Here 16 bounds the row range, while the rewritten
// access width would be the last dimension of size 4. Ignore the row bound
// instead of comparing it with the column width. Changing only the reshape
// result from 32x4 to 32x16 would create an invalid 128-to-512-element
// reshape. The independent last-axis shrink is also left unchanged because a
// pattern without a proven embedding size makes the function ineligible.

// CHECK-LABEL: tt.func public @do_not_expand_flattened_range
// CHECK-NOT:   group_id
// CHECK:       %[[RANGE:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:  %{{.*}} = tt.reshape %[[RANGE]] : tensor<128xi32> -> tensor<32x4xi32>
// CHECK-NOT:   group_id
// CHECK:       %[[RANGE32:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:  %{{.*}} = tt.expand_dims %[[RANGE32]] {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NOT:   group_id
// CHECK:       tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK-NOT:   group_id
// CHECK:       tt.return

module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @do_not_expand_flattened_range(%src: !tt.ptr<f32>,
                                                %dst: !tt.ptr<f32>) {
    %c16 = arith.constant dense<16> : tensor<32xi32>
    %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %row_2d = tt.reshape %row : tensor<32xi32> -> tensor<32x1xi32>
    %rows = tt.broadcast %row_2d : tensor<32x1xi32> -> tensor<32x4xi32>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %col_2d = tt.reshape %col : tensor<4xi32> -> tensor<1x4xi32>
    %cols = tt.broadcast %col_2d : tensor<1x4xi32> -> tensor<32x4xi32>
    %offsets = arith.addi %rows, %cols : tensor<32x4xi32>
    %mask_1d = arith.cmpi slt, %row, %c16 : tensor<32xi32>
    %mask_col = tt.reshape %mask_1d : tensor<32xi1> -> tensor<32x1xi1>
    %mask = tt.broadcast %mask_col : tensor<32x1xi1> -> tensor<32x4xi1>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %range_2d = tt.reshape %range : tensor<128xi32> -> tensor<32x4xi32>
    %src_splat = tt.splat %src : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_splat, %range_2d
        : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
    %value = tt.load %src_ptrs, %mask : tensor<32x4x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %offsets
        : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
    tt.store %dst_ptrs, %value, %mask : tensor<32x4x!tt.ptr<f32>>

    // This independent chain could shrink 32 -> 8, but must remain in place
    // because the preceding group has no proven last-axis bound and the pass
    // currently uses a function-atomic fallback.
    %c8 = arith.constant dense<8> : tensor<1x32xi32>
    %range32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range32_2d = tt.expand_dims %range32 {axis = 0 : i32}
        : tensor<32xi32> -> tensor<1x32xi32>
    %indices32 = tt.broadcast %range32_2d
        : tensor<1x32xi32> -> tensor<16x32xi32>
    %mask32_1d = arith.cmpi slt, %range32_2d, %c8
        : tensor<1x32xi32>
    %mask32 = tt.broadcast %mask32_1d
        : tensor<1x32xi1> -> tensor<16x32xi1>
    %src32 = tt.splat %src
        : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %src_ptrs32 = tt.addptr %src32, %indices32
        : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
    %value32 = tt.load %src_ptrs32, %mask32
        : tensor<16x32x!tt.ptr<f32>>
    %dst32 = tt.splat %dst
        : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %dst_ptrs32 = tt.addptr %dst32, %indices32
        : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
    tt.store %dst_ptrs32, %value32, %mask32
        : tensor<16x32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A wrong-axis bound is unsafe in both numeric directions. In particular, a
// row bound smaller than the column width must not be interpreted as a valid
// last-axis shrink.

// CHECK-LABEL: tt.func public @do_not_shrink_from_row_bound
// CHECK-NOT:   group_id
// CHECK:       %[[COL:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:       %[[MASK:.*]] = tt.broadcast %{{.*}} : tensor<32x1xi1> -> tensor<32x4xi1>
// CHECK:       tt.store %{{.*}}, %{{.*}}, %[[MASK]] : tensor<32x4x!tt.ptr<f32>>
// CHECK-NOT:   group_id
// CHECK:       tt.return

module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @do_not_shrink_from_row_bound(
      %src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c2 = arith.constant dense<2> : tensor<32xi32>
    %row = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %row_2d = tt.reshape %row : tensor<32xi32> -> tensor<32x1xi32>
    %rows = tt.broadcast %row_2d
        : tensor<32x1xi32> -> tensor<32x4xi32>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %col_2d = tt.reshape %col : tensor<4xi32> -> tensor<1x4xi32>
    %cols = tt.broadcast %col_2d : tensor<1x4xi32> -> tensor<32x4xi32>
    %offsets = arith.addi %rows, %cols : tensor<32x4xi32>
    %row_mask_1d = arith.cmpi slt, %row, %c2 : tensor<32xi32>
    %row_mask_2d = tt.reshape %row_mask_1d
        : tensor<32xi1> -> tensor<32x1xi1>
    %mask = tt.broadcast %row_mask_2d
        : tensor<32x1xi1> -> tensor<32x4xi1>
    %src_splat = tt.splat %src
        : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_splat, %offsets
        : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
    %value = tt.load %src_ptrs, %mask : tensor<32x4x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst
        : !tt.ptr<f32> -> tensor<32x4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %offsets
        : tensor<32x4x!tt.ptr<f32>>, tensor<32x4xi32>
    tt.store %dst_ptrs, %value, %mask : tensor<32x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Equality is a no-op, not an unsupported expansion. Preserve the historical
// behavior in which an equal-width group can coexist with an independently
// shrinkable, axis-aligned group.

// CHECK-LABEL: tt.func public @shrink_with_equal_width_group
// CHECK-DAG:   tt.make_range {end = 64 : i32, group_id = 0 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-DAG:   tt.make_range {end = 8 : i32, group_id = 1 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-DAG:   tt.store %{{.*}}, %{{.*}}, %{{.*}} {group_id = 1 : i32} : tensor<16x8x!tt.ptr<f32>>

module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @shrink_with_equal_width_group(
      %src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %range64_2d = tt.expand_dims %range64 {axis = 0 : i32}
        : tensor<64xi32> -> tensor<1x64xi32>
    %indices64 = tt.broadcast %range64_2d
        : tensor<1x64xi32> -> tensor<16x64xi32>
    %mask64_1d = arith.cmpi slt, %range64_2d, %c64
        : tensor<1x64xi32>
    %mask64 = tt.broadcast %mask64_1d
        : tensor<1x64xi1> -> tensor<16x64xi1>
    %src64 = tt.splat %src
        : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
    %src_ptrs64 = tt.addptr %src64, %indices64
        : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    %value64 = tt.load %src_ptrs64, %mask64
        : tensor<16x64x!tt.ptr<f32>>
    %dst64 = tt.splat %dst
        : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
    %dst_ptrs64 = tt.addptr %dst64, %indices64
        : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    tt.store %dst_ptrs64, %value64, %mask64
        : tensor<16x64x!tt.ptr<f32>>

    %c8 = arith.constant dense<8> : tensor<1x32xi32>
    %range32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range32_2d = tt.expand_dims %range32 {axis = 0 : i32}
        : tensor<32xi32> -> tensor<1x32xi32>
    %indices32 = tt.broadcast %range32_2d
        : tensor<1x32xi32> -> tensor<16x32xi32>
    %mask32_1d = arith.cmpi slt, %range32_2d, %c8
        : tensor<1x32xi32>
    %mask32 = tt.broadcast %mask32_1d
        : tensor<1x32xi1> -> tensor<16x32xi1>
    %src32 = tt.splat %src
        : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %src_ptrs32 = tt.addptr %src32, %indices32
        : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
    %value32 = tt.load %src_ptrs32, %mask32
        : tensor<16x32x!tt.ptr<f32>>
    %dst32 = tt.splat %dst
        : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %dst_ptrs32 = tt.addptr %dst32, %indices32
        : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
    tt.store %dst_ptrs32, %value32, %mask32
        : tensor<16x32x!tt.ptr<f32>>
    tt.return
  }
}
