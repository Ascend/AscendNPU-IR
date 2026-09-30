// RUN: bishengir-opt -split-input-file %s -loop-restructure-arange-optimization | FileCheck %s

// Both loads must precede both stores, even if any of the arguments alias.
// Group IDs may be width-sorted; actual memory operations must not be.
// CHECK-LABEL: tt.func public @alias_order
// CHECK: %[[A:.*]] = tt.load {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: %[[B:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[A]], {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[B]], {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @alias_order(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %as = tt.splat %a : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ap = tt.addptr %as, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// An unchanged tensor argument may need a no-op broadcast. Creating it must
// not move the caller's clone before the definitions of its other operands.
// CHECK-LABEL: tt.func public @equal_width_tensor_argument
// CHECK: %[[A:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[B:.*]] = tt.load {{.*}} {group_id = 1 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[A]], {{.*}} {group_id = 0 : i32} : tensor<1x64x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[B]], {{.*}} {group_id = 1 : i32} : tensor<1x16x!tt.ptr<f32>>
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @equal_width_tensor_argument(%a: tensor<1x64x!tt.ptr<f32>>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<64> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %ap = tt.addptr %a, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// No copied load may cross a write outside the selected store/load patterns.
// CHECK-LABEL: tt.func public @unselected_store
// CHECK: %[[A:.*]] = tt.load {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store {{.*}} : tensor<1x64x!tt.ptr<f32>>
// CHECK: %[[B:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[A]], {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[B]], {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @unselected_store(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %as = tt.splat %a : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ap = tt.addptr %as, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %zeros = arith.constant dense<0.0> : tensor<1x64xf32>
    tt.store %dp, %zeros, %full : tensor<1x64x!tt.ptr<f32>>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Volatile reads must not be duplicated, even if their type could be shrunk.
// CHECK-LABEL: tt.func public @volatile_dependency
// CHECK-NOT: group_id
// CHECK: tt.load {{.*}} {isVolatile = true} : tensor<1x64x!tt.ptr<f32>>
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @volatile_dependency(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %as = tt.splat %a : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ap = tt.addptr %as, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 {isVolatile = true} : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A read after a potentially aliasing write must remain after that write.
// CHECK-LABEL: tt.func public @read_after_write
// CHECK: %[[A:.*]] = tt.load {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[A]], {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: %[[B:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[B]], {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @read_after_write(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %as = tt.splat %a : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ap = tt.addptr %as, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Keep the original loop and per-iteration memory ordering.
// CHECK-LABEL: tt.func public @loop_alias_order
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: %[[A:.*]] = tt.load {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: %[[B:.*]] = tt.load {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[A]], {{.*}} {group_id = 1 : i32} : tensor<1x8x!tt.ptr<f32>>
// CHECK: tt.store {{.*}}, %[[B]], {{.*}} {group_id = 0 : i32} : tensor<1x16x!tt.ptr<f32>>
// CHECK-NOT: scf.for
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @loop_alias_order(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    scf.for %iv = %zero to %two step %one {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %as = tt.splat %a : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ap = tt.addptr %as, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// Scalar result types do not imply that regionless cloning is legal.
// CHECK-LABEL: tt.func public @scalar_region
// CHECK-NOT: group_id
// CHECK: scf.if
// CHECK-NOT: group_id
// CHECK: scf.yield
// CHECK-NOT: group_id
// CHECK: } else {
// CHECK-NOT: group_id
// CHECK: scf.yield
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @scalar_region(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %pred: i1) {
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<8> : tensor<1x64xi8>
    %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %reshaped = tt.reshape %range : tensor<64xi32> -> tensor<1x64xi32>
    %range2 = tt.broadcast %reshaped : tensor<1x64xi32> -> tensor<1x64xi32>
    %narrow = arith.trunci %range2 : tensor<1x64xi32> to tensor<1x64xi8>
    %full = arith.cmpi ult, %range2, %c64 : tensor<1x64xi32>
    %periodic = arith.cmpi ult, %narrow, %c8 : tensor<1x64xi8>
    %mask = arith.andi %full, %periodic : tensor<1x64xi1>
    %selected = scf.if %pred -> (!tt.ptr<f32>) {
      scf.yield %src : !tt.ptr<f32>
    } else {
      scf.yield %dst : !tt.ptr<f32>
    }
    %s = tt.splat %selected : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %sp = tt.addptr %s, %range2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %v = tt.load %sp, %mask : tensor<1x64x!tt.ptr<f32>>
    %d = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %d, %range2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %v, %mask : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}


// -----

// Equal-width groups are also cloned when a different group shrinks.
// CHECK-LABEL: tt.func public @region_in_equal_width_group
// CHECK-NOT: group_id
// CHECK: scf.if
// CHECK-NOT: group_id
// CHECK: scf.yield
// CHECK-NOT: group_id
// CHECK: } else {
// CHECK-NOT: group_id
// CHECK: scf.yield
// CHECK-NOT: group_id
// CHECK: tt.return
module attributes {"ttg.simt-optimization-mode" = 900101 : i32} {
  tt.func public @region_in_equal_width_group(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %pred: i1) {
    %r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %r2 = tt.expand_dims %r {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %c64 = arith.constant dense<64> : tensor<1x64xi32>
    %c8 = arith.constant dense<64> : tensor<1x64xi32>
    %c16 = arith.constant dense<16> : tensor<1x64xi32>
    %full = arith.cmpi ult, %r2, %c64 : tensor<1x64xi32>
    %lt8 = arith.cmpi ult, %r2, %c8 : tensor<1x64xi32>
    %lt16 = arith.cmpi ult, %r2, %c16 : tensor<1x64xi32>
    %m8 = arith.andi %full, %lt8 : tensor<1x64xi1>
    %m16 = arith.andi %full, %lt16 : tensor<1x64xi1>
    %selected = scf.if %pred -> (!tt.ptr<f32>) {
      scf.yield %a : !tt.ptr<f32>
    } else {
      scf.yield %b : !tt.ptr<f32>
    }
    %as = tt.splat %selected : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ap = tt.addptr %as, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %av = tt.load %ap, %m8 : tensor<1x64x!tt.ptr<f32>>
    %bs = tt.splat %b : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %bp = tt.addptr %bs, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %bv = tt.load %bp, %m16 : tensor<1x64x!tt.ptr<f32>>
    %ds = tt.splat %dst : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %dp = tt.addptr %ds, %r2 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    tt.store %dp, %av, %m8 : tensor<1x64x!tt.ptr<f32>>
    tt.store %dp, %bv, %m16 : tensor<1x64x!tt.ptr<f32>>
    tt.return
  }
}
