// RUN: bishengir-opt %s -loop-restructure-arange-optimization -canonicalize | FileCheck %s
// RUN: bishengir-opt %s -canonicalize -loop-restructure-arange-optimization -canonicalize | FileCheck %s

// Shrink columns from 64 to 8 without changing the two source row offsets.
// Before cloning, canonicalization may fold the constant's expand_dims; both
// tensor<2xi32> and tensor<2x1xi32> must retain all their original values.
// CHECK-LABEL: tt.func public @row_offsets
// CHECK-DAG: %[[ROWS:.*]] = arith.constant {{.*}}dense<{{\[\[0\], \[64\]\]}}> : tensor<2x1xi32>
// CHECK-DAG: %[[RANGE:.*]] = tt.make_range {{.*}}end = 8 : i32{{.*}} : tensor<8xi32>
// CHECK: %[[COLS1:.*]] = tt.expand_dims %[[RANGE]] {{.*}} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK: %[[COLS:.*]] = tt.broadcast %[[COLS1]] {{.*}} : tensor<1x8xi32> -> tensor<2x8xi32>
// CHECK: %[[ROW_OFFSETS:.*]] = tt.broadcast %[[ROWS]] {{.*}} : tensor<2x1xi32> -> tensor<2x8xi32>
// CHECK: %[[OFFSETS:.*]] = arith.addi %[[ROW_OFFSETS]], %[[COLS]] {{.*}} : tensor<2x8xi32>
// CHECK: %[[SRC:.*]] = tt.splat %arg0 {{.*}} : !tt.ptr<f32> -> tensor<2x8x!tt.ptr<f32>>
// CHECK: %[[PTRS:.*]] = tt.addptr %[[SRC]], %[[OFFSETS]] {{.*}} : tensor<2x8x!tt.ptr<f32>>, tensor<2x8xi32>
// CHECK: %[[VALUES:.*]] = tt.load %[[PTRS]], {{.*}} : tensor<2x8x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %[[VALUES]], {{.*}} : tensor<2x8x!tt.ptr<f32>>
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @row_offsets(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %m = arith.andi %full, %lt8 : tensor<1x64xi1>
    %mask = tt.broadcast %m : tensor<1x64xi1> -> tensor<2x64xi1>
    %cols = tt.broadcast %r2 : tensor<1x64xi32> -> tensor<2x64xi32>
    %rows = arith.constant dense<[0, 64]> : tensor<2xi32>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %row_offsets = tt.broadcast %rows2 : tensor<2x1xi32> -> tensor<2x64xi32>
    %offsets = arith.addi %row_offsets, %cols : tensor<2x64xi32>
    %sp = tt.splat %src : !tt.ptr<f32> -> tensor<2x64x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %sp, %offsets : tensor<2x64x!tt.ptr<f32>>, tensor<2x64xi32>
    %v = tt.load %src_ptrs, %mask : tensor<2x64x!tt.ptr<f32>>
    %dr = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %stride = arith.constant dense<64> : tensor<2xi32>
    %drs = arith.muli %dr, %stride : tensor<2xi32>
    %dr2 = tt.expand_dims %drs {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %drows = tt.broadcast %dr2 : tensor<2x1xi32> -> tensor<2x64xi32>
    %doff = arith.addi %drows, %cols : tensor<2x64xi32>
    %dp = tt.splat %dst : !tt.ptr<f32> -> tensor<2x64x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dp, %doff : tensor<2x64x!tt.ptr<f32>>, tensor<2x64xi32>
    tt.store %dst_ptrs, %v, %mask : tensor<2x64x!tt.ptr<f32>>
    tt.return
  }
}
