// RUN: bishengir-opt %s -split-input-file -loop-restructure-arange-optimization -canonicalize | FileCheck %s

// An intervening unselected store aliases both the selected load and store.
// The selected store must write the value read BEFORE that intervening write,
// just as the input does. This is safe to shrink, not a reason to reject it.
// CHECK-LABEL: tt.func public @foreign_alias_write
// CHECK: %[[VALUE:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK-NEXT: tt.store {{.*}} : tensor<1x64x!tt.ptr<f32>>
// CHECK-NEXT: tt.store {{.*}}, %[[VALUE]], {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @foreign_alias_write(%base: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %mask = arith.andi %full, %lt8 : tensor<1x64xi1>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr = tt.addptr %s, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %before = tt.load %ptr, %mask : tensor<1x64x!tt.ptr<f32>>
    %zeros = arith.constant dense<0.0> : tensor<1x64xf32>
    tt.store %ptr, %zeros, %full : tensor<1x64x!tt.ptr<f32>>
    tt.store %ptr, %before, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The volatile read is not the pattern's loadOp: it is a scalar dependency of
// the address. Preflight must still reject copying it, despite unchanged type.
// CHECK-LABEL: tt.func public @secondary_volatile
// CHECK-NOT: group_id
// CHECK: tt.load {{.*}} {isVolatile = true} : !tt.ptr<i32>
// CHECK-NOT: group_id
// CHECK: tt.load {{.*}} : tensor<1x64x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @secondary_volatile(%base: !tt.ptr<f32>, %index: !tt.ptr<i32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %mask = arith.andi %full, %lt8 : tensor<1x64xi1>
    %shift = tt.load %index {isVolatile = true} : !tt.ptr<i32>
    %shifts = tt.splat %shift : i32 -> tensor<1x64xi32>
    %offsets = arith.addi %r2, %shifts : tensor<1x64xi32>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr = tt.addptr %s, %offsets : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %before = tt.load %ptr, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.store %ptr, %before, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}
