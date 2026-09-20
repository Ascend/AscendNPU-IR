// RUN: bishengir-opt -split-input-file %s -loop-restructure-arange-optimization | FileCheck %s

// Truncation wraps: both [0, 8) and [256, 264) are live stores.
// CHECK-LABEL: tt.func public @truncated_range
// CHECK-NOT: group_id
// CHECK: tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
// CHECK-NOT: group_id
// CHECK: arith.trunci {{.*}} : tensor<1x512xi32> to tensor<1x512xi8>
// CHECK-NOT: group_id
// CHECK: tt.store {{.*}} : tensor<1x512x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @truncated_range(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c512 = arith.constant dense<512> : tensor<1x512xi32>
    %c8 = arith.constant dense<8> : tensor<1x512xi8>
    %range = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32>
    %range2 = tt.expand_dims %range {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %narrow = arith.trunci %range2 : tensor<1x512xi32> to tensor<1x512xi8>
    %full = arith.cmpi ult, %range2, %c512 : tensor<1x512xi32>
    %periodic = arith.cmpi ult, %narrow, %c8 : tensor<1x512xi8>
    %mask = arith.andi %full, %periodic : tensor<1x512xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x512x!tt.ptr<f32>>, tensor<1x512xi32>
    %v = tt.load %sp, %mask : tensor<1x512x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x512x!tt.ptr<f32>>, tensor<1x512xi32>
    tt.store %dp, %v, %mask : tensor<1x512x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Adding a leading unit dimension preserves the sliced last axis.
// CHECK-LABEL: tt.func public @safe_unit_reshape
// CHECK: tt.reshape {{.*}} {group_id = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK: tt.store {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @safe_unit_reshape(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi8>
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %reshaped = tt.reshape %range : tensor<64xi32> -> tensor<1x64xi32>
    %range2 = tt.broadcast %reshaped : tensor<1x64xi32> -> tensor<1x64xi32>
    %narrow = arith.trunci %range2 : tensor<1x64xi32> to tensor<1x64xi8>
    %full = arith.cmpi ult, %range2, %c64 : tensor<1x64xi32>
    %periodic = arith.cmpi ult, %narrow, %c8 : tensor<1x64xi8>
    %mask = arith.andi %full, %periodic : tensor<1x64xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %v = tt.load %sp, %mask : tensor<1x64x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %v, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A trailing unit dimension makes the row range unrelated to the last axis,
// whether introduced by expand_dims or reshape.
// CHECK-LABEL: tt.func public @reshape_axis_collision
// CHECK-NOT: group_id
// CHECK: tt.reshape {{.*}} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NOT: group_id
// CHECK: tt.store {{.*}} : tensor<32x32x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @reshape_axis_collision(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c32 = arith.constant dense<32> : tensor<32xi32>
    %c8 = arith.constant dense<8> : tensor<32xi32>
    %row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %row2 = tt.reshape %row : tensor<32xi32> -> tensor<32x1xi32>
    %rows = tt.broadcast %row2 : tensor<32x1xi32> -> tensor<32x32xi32>
    %col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %col2 = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %cols = tt.broadcast %col2 : tensor<1x32xi32> -> tensor<32x32xi32>
    %stride = arith.constant dense<32> : tensor<32x32xi32>
    %row_offsets = arith.muli %rows, %stride : tensor<32x32xi32>
    %offsets = arith.addi %row_offsets, %cols : tensor<32x32xi32>
    %rm = arith.cmpi slt, %row, %c32 : tensor<32xi32>
    %rm2 = tt.reshape %rm : tensor<32xi1> -> tensor<32x1xi1>
    %rmb = tt.broadcast %rm2 : tensor<32x1xi1> -> tensor<32x32xi1>
    %cm = arith.cmpi slt, %col, %c8 : tensor<32xi32>
    %cm2 = tt.expand_dims %cm {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %cmb = tt.broadcast %cm2 : tensor<1x32xi1> -> tensor<32x32xi1>
    %mask = arith.andi %rmb, %cmb : tensor<32x32xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %sp = tt.addptr %s, %offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %v = tt.load %sp, %mask : tensor<32x32x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %dp = tt.addptr %d, %offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %dp, %v, %mask : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// [0, 256) fits unsigned i8 even though it does not fit signed i8.
// CHECK-LABEL: tt.func public @unsigned_truncated_range
// CHECK: tt.make_range {end = 8 : i32, group_id = 0 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: tt.store {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @unsigned_truncated_range(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c256 = arith.constant dense<256> : tensor<1x256xi32>
    %c8 = arith.constant dense<8> : tensor<1x256xi8>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %range2 = tt.expand_dims %range {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %narrow = arith.trunci %range2 : tensor<1x256xi32> to tensor<1x256xi8>
    %full = arith.cmpi ult, %range2, %c256 : tensor<1x256xi32>
    %periodic = arith.cmpi ult, %narrow, %c8 : tensor<1x256xi8>
    %mask = arith.andi %full, %periodic : tensor<1x256xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
    %v = tt.load %sp, %mask : tensor<1x256x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
    tt.store %dp, %v, %mask : tensor<1x256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Widening after truncation does not restore values if it sign-extends them.
// CHECK-LABEL: tt.func public @sign_extended_range
// CHECK-NOT: group_id
// CHECK: arith.extsi {{.*}} : tensor<1x256xi8> to tensor<1x256xi32>
// CHECK-NOT: group_id
// CHECK: tt.store {{.*}} : tensor<1x256x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @sign_extended_range(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c256 = arith.constant dense<256> : tensor<1x256xi32>
    %c8 = arith.constant dense<8> : tensor<1x256xi8>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %range2 = tt.expand_dims %range {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %narrow = arith.trunci %range2 : tensor<1x256xi32> to tensor<1x256xi8>
    %full = arith.cmpi ult, %range2, %c256 : tensor<1x256xi32>
    %extended = arith.extsi %narrow : tensor<1x256xi8> to tensor<1x256xi32>
    %bound = arith.constant dense<8> : tensor<1x256xi32>
    %periodic = arith.cmpi slt, %extended, %bound : tensor<1x256xi32>
    %mask = arith.andi %full, %periodic : tensor<1x256xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
    %v = tt.load %sp, %mask : tensor<1x256x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
    tt.store %dp, %v, %mask : tensor<1x256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Equal row/column widths must not make the cloner shrink the row axis.
// CHECK-LABEL: tt.func public @axis_collision
// CHECK-NOT: group_id
// CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NOT: group_id
// CHECK: tt.store {{.*}} : tensor<32x32x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @axis_collision(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c32 = arith.constant dense<32> : tensor<32xi32>
    %c8 = arith.constant dense<8> : tensor<32xi32>
    %row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %rows = tt.broadcast %row2 : tensor<32x1xi32> -> tensor<32x32xi32>
    %col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %col2 = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %cols = tt.broadcast %col2 : tensor<1x32xi32> -> tensor<32x32xi32>
    %stride = arith.constant dense<32> : tensor<32x32xi32>
    %row_offsets = arith.muli %rows, %stride : tensor<32x32xi32>
    %offsets = arith.addi %row_offsets, %cols : tensor<32x32xi32>
    %rm = arith.cmpi slt, %row, %c32 : tensor<32xi32>
    %rm2 = tt.expand_dims %rm {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
    %rmb = tt.broadcast %rm2 : tensor<32x1xi1> -> tensor<32x32xi1>
    %cm = arith.cmpi slt, %col, %c8 : tensor<32xi32>
    %cm2 = tt.expand_dims %cm {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %cmb = tt.broadcast %cm2 : tensor<1x32xi1> -> tensor<32x32xi1>
    %mask = arith.andi %rmb, %cmb : tensor<32x32xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %sp = tt.addptr %s, %offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %v = tt.load %sp, %mask : tensor<32x32x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %dp = tt.addptr %d, %offsets : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %dp, %v, %mask : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A provably value-preserving truncation should still allow the shrink.
// CHECK-LABEL: tt.func public @safe_truncated_range
// CHECK: tt.make_range {end = 8 : i32, group_id = 0 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: arith.trunci {{.*}} : tensor<1x8xi32> to tensor<1x8xi8>
// CHECK: tt.store {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @safe_truncated_range(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi8>
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %range2 = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %narrow = arith.trunci %range2 : tensor<1x64xi32> to tensor<1x64xi8>
    %full = arith.cmpi ult, %range2, %c64 : tensor<1x64xi32>
    %periodic = arith.cmpi ult, %narrow, %c8 : tensor<1x64xi8>
    %mask = arith.andi %full, %periodic : tensor<1x64xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %v = tt.load %sp, %mask : tensor<1x64x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %v, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Although unsigned i8 can represent [0, 256), a signed comparison observes
// [128, 256) as negative. It is not a prefix bound on the original range.
// CHECK-LABEL: tt.func public @signed_truncated_range
// CHECK-NOT: group_id
// CHECK: tt.store {{.*}} : tensor<1x256x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @signed_truncated_range(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c256 = arith.constant dense<256> : tensor<1x256xi32>
    %c8 = arith.constant dense<8> : tensor<1x256xi8>
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %range2 = tt.expand_dims %range {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %narrow = arith.trunci %range2 : tensor<1x256xi32> to tensor<1x256xi8>
    %full = arith.cmpi ult, %range2, %c256 : tensor<1x256xi32>
    %periodic = arith.cmpi slt, %narrow, %c8 : tensor<1x256xi8>
    %mask = arith.andi %full, %periodic : tensor<1x256xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
    %v = tt.load %sp, %mask : tensor<1x256x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x256x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x256x!tt.ptr<f32>>, tensor<1x256xi32>
    tt.store %dp, %v, %mask : tensor<1x256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Index casts can narrow too; do not infer their width without a target proof.
// CHECK-LABEL: tt.func public @index_cast_range
// CHECK-NOT: group_id
// CHECK: arith.index_cast {{.*}} : tensor<1x512xindex> to tensor<1x512xi8>
// CHECK-NOT: group_id
// CHECK: tt.store {{.*}} : tensor<1x512x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @index_cast_range(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c512 = arith.constant dense<512> : tensor<1x512xi32>
    %c8 = arith.constant dense<8> : tensor<1x512xi8>
    %range = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32>
    %range2 = tt.expand_dims %range {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %indices = arith.index_cast %range2 : tensor<1x512xi32> to tensor<1x512xindex>
    %narrow = arith.index_cast %indices : tensor<1x512xindex> to tensor<1x512xi8>
    %full = arith.cmpi ult, %range2, %c512 : tensor<1x512xi32>
    %periodic = arith.cmpi ult, %narrow, %c8 : tensor<1x512xi8>
    %mask = arith.andi %full, %periodic : tensor<1x512xi1>
    %s = tt.splat %src : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x512x!tt.ptr<f32>>, tensor<1x512xi32>
    %v = tt.load %sp, %mask : tensor<1x512x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x512x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x512x!tt.ptr<f32>>, tensor<1x512xi32>
    tt.store %dp, %v, %mask : tensor<1x512x!tt.ptr<f32>>
    tt.return
  }
}
