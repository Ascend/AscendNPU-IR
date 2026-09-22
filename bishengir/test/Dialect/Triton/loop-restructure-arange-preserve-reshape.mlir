// RUN: bishengir-opt %s -loop-restructure-arange-optimization | FileCheck %s
// RUN: bishengir-opt %s -loop-restructure-arange-optimization | FileCheck %s --check-prefix=PRESERVE

// PRESERVE-LABEL: tt.func public @resize_flattened_range_before_reshape
// PRESERVE-NOT: group_id
// PRESERVE-LABEL: tt.func public @optimize_with_unrelated_reshape

// CHECK-LABEL: tt.func public @resize_flattened_range_before_reshape
// CHECK: %[[FLAT_RANGE:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK: tt.reshape %[[FLAT_RANGE]] : tensor<16xi32> -> tensor<2x8xi32>
// CHECK: tt.return

// CHECK-LABEL: tt.func public @optimize_with_unrelated_reshape
// Safe stores are rewritten in place, before the unrelated reshape.
// CHECK: tt.store {{.*}} {group_id = 0 : i32} : tensor<2x4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}} {group_id = 1 : i32} : tensor<2x2x!tt.ptr<i32>>
// CHECK: tt.reshape {{.*}} : tensor<4xi32> -> tensor<2x2xi32>
// CHECK: tt.return

module attributes {"ttg.simt-optimization-mode" = 200000 : i32} {
  tt.func public @resize_flattened_range_before_reshape(
      %src: !tt.ptr<i32>, %dst: !tt.ptr<i32>) {
    %flat_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %reshaped = tt.reshape %flat_range : tensor<16xi32> -> tensor<2x8xi32>
    %column_range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %column_expanded = tt.expand_dims %column_range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %columns = tt.broadcast %column_expanded : tensor<1x8xi32> -> tensor<2x8xi32>
    %indices = arith.addi %reshaped, %columns : tensor<2x8xi32>

    %src_splat = tt.splat %src : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %src_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded = tt.load %src_ptrs : tensor<2x8x!tt.ptr<i32>>

    %four = arith.constant dense<4> : tensor<2x8xi32>
    %store_mask = arith.cmpi slt, %indices, %four : tensor<2x8xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dst_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst_ptrs, %loaded, %store_mask : tensor<2x8x!tt.ptr<i32>>
    tt.return
  }

  tt.func public @optimize_with_unrelated_reshape(
      %src0: !tt.ptr<i32>, %dst0: !tt.ptr<i32>,
      %src1: !tt.ptr<i32>, %dst1: !tt.ptr<i32>) {
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %indices = tt.broadcast %expanded : tensor<1x8xi32> -> tensor<2x8xi32>

    %src0_splat = tt.splat %src0 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src0_ptrs = tt.addptr %src0_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded0 = tt.load %src0_ptrs : tensor<2x8x!tt.ptr<i32>>
    %four = arith.constant dense<4> : tensor<2x8xi32>
    %mask0 = arith.cmpi slt, %indices, %four : tensor<2x8xi32>
    %dst0_splat = tt.splat %dst0 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst0_ptrs = tt.addptr %dst0_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst0_ptrs, %loaded0, %mask0 : tensor<2x8x!tt.ptr<i32>>

    %src1_splat = tt.splat %src1 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src1_ptrs = tt.addptr %src1_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded1 = tt.load %src1_ptrs : tensor<2x8x!tt.ptr<i32>>
    %two = arith.constant dense<2> : tensor<2x8xi32>
    %mask1 = arith.cmpi slt, %indices, %two : tensor<2x8xi32>
    %dst1_splat = tt.splat %dst1 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst1_ptrs = tt.addptr %dst1_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst1_ptrs, %loaded1, %mask1 : tensor<2x8x!tt.ptr<i32>>

    %flat_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %unused = tt.reshape %flat_range : tensor<4xi32> -> tensor<2x2xi32>
    tt.return
  }
}
