// RUN: bishengir-opt %s -split-input-file -loop-restructure-arange-optimization | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -loop-restructure-arange-optimization -canonicalize -cse -o /dev/null

// Valid SSA does not require textual block order to follow dominance. The
// producer block is printed after its consumer but executes before it. Clone
// construction must visit producers first without moving copies between blocks.
// CHECK-LABEL: tt.func public @cfg_order
// CHECK: cf.br ^bb2
// CHECK: ^bb1:
// CHECK: %[[LOAD:.*]] = tt.load %[[PTR:[^,]+]], %[[MASK:[^ ]+]] {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store %[[PTR]], %[[LOAD]], %[[MASK]] {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.return
// CHECK: ^bb2:
// CHECK: tt.make_range {end = 8 : i32, group_id = 0 : i32, start = 0 : i32}
// CHECK: %[[MASK]] = arith.andi
// CHECK: %[[PTR]] = tt.addptr
// CHECK: cf.br ^bb1
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @cfg_order(%base: !tt.ptr<f32>) {
    cf.br ^producer
  ^consumer:
    %before = tt.load %ptr, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.store %ptr, %before, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  ^producer:
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %mask = arith.andi %full, %lt8 : tensor<1x64xi1>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr = tt.addptr %s, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    cf.br ^consumer
  }
}

// -----

// Building clones in dependency order must not schedule memory operations in
// that order. Preserve an intervening unselected write to the exact same pointer:
// the final store writes the value read before the zeroing store.
// CHECK-LABEL: tt.func public @cfg_foreign_alias_write
// CHECK: cf.br ^bb2
// CHECK: ^bb1:
// CHECK: %[[VALUE:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<1x64x!tt.ptr<f32>>
// CHECK-NEXT: tt.store {{.*}}, %[[VALUE]], {{.*}} {group_id = 0 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return
// CHECK: ^bb2:
// CHECK: tt.make_range {end = 8 : i32, group_id = 0 : i32, start = 0 : i32}
// CHECK: cf.br ^bb1
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @cfg_foreign_alias_write(%base: !tt.ptr<f32>) {
    cf.br ^producer
  ^consumer:
    %before = tt.load %ptr, %mask : tensor<1x64x!tt.ptr<f32>>
    %zeros = arith.constant dense<0.0> : tensor<1x64xf32>
    tt.store %ptr, %zeros, %full : tensor<1x64x!tt.ptr<f32>>
    tt.store %ptr, %before, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  ^producer:
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %mask = arith.andi %full, %lt8 : tensor<1x64xi1>
    %s = tt.splat %base : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr = tt.addptr %s, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    cf.br ^consumer
  }
}
