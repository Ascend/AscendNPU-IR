// RUN: bishengir-opt -convert-shared-ptr-to-memdesc %s -split-input-file | FileCheck %s

// ---- Test 1: All accesses in the same block ---------------------------------
//
// The nearest common dominator of all accesses is the function entry block.
// The alloc is placed just before the first access (the envelope store), AFTER
// the offset-computation IR and constants — NOT at the very start of the block.
//
// Input: an envelope store (tileSize=8) + two per-tile stores (tileSize=4),
//        all three directly in the function entry block.

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: tt.func @all_same_block(
// CHECK:           ttg.local_alloc
// CHECK-NEXT:      ttg.local_store
// CHECK:           ttg.memdesc_subslice
// CHECK:           ttg.local_store
// CHECK:           ttg.memdesc_subslice
// CHECK:           ttg.local_store
// CHECK-NOT:       tt.store
// CHECK-NOT:       tt.splat
// CHECK-NOT:       tt.addptr
module {
  tt.func @all_same_block(%arg0: !tt.ptr<f32, 6> {bishengir.scratch_shm}) {
    %cst0 = arith.constant dense<0.0> : tensor<8x8xf32>
    %cst1 = arith.constant dense<1.0> : tensor<8x4xf32>
    %cst2 = arith.constant dense<2.0> : tensor<8x4xf32>
    // Envelope store: full 8x8 (tileSize=8 = envSize)
    %sp0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x8x!tt.ptr<f32, 6>>
    %r0  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e0  = tt.expand_dims %r0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %b0  = tt.broadcast %e0 : tensor<1x8xi32> -> tensor<8x8xi32>
    %r1  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e1  = tt.expand_dims %r1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %b1  = tt.broadcast %e1 : tensor<8x1xi32> -> tensor<8x8xi32>
    %off0 = arith.addi %b0, %b1 : tensor<8x8xi32>
    %ap0 = tt.addptr %sp0, %off0 : tensor<8x8x!tt.ptr<f32, 6>>, tensor<8x8xi32>
    tt.store %ap0, %cst0 : tensor<8x8x!tt.ptr<f32, 6>>
    // Per-tile store 1: 8x4 at tile offset 0
    %sp1 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x4x!tt.ptr<f32, 6>>
    %r2  = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %e2  = tt.expand_dims %r2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %b2  = tt.broadcast %e2 : tensor<1x4xi32> -> tensor<8x4xi32>
    %r3  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e3  = tt.expand_dims %r3 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %b3  = tt.broadcast %e3 : tensor<8x1xi32> -> tensor<8x4xi32>
    %off1 = arith.addi %b2, %b3 : tensor<8x4xi32>
    %ap1 = tt.addptr %sp1, %off1 : tensor<8x4x!tt.ptr<f32, 6>>, tensor<8x4xi32>
    tt.store %ap1, %cst1 : tensor<8x4x!tt.ptr<f32, 6>>
    // Per-tile store 2: 8x4 at tile offset 4
    %sp2 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x4x!tt.ptr<f32, 6>>
    %r4  = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c4  = arith.constant dense<4> : tensor<4xi32>
    %r4_off = arith.addi %r4, %c4 : tensor<4xi32>
    %e4  = tt.expand_dims %r4_off {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %b4  = tt.broadcast %e4 : tensor<1x4xi32> -> tensor<8x4xi32>
    %r5  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e5  = tt.expand_dims %r5 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %b5  = tt.broadcast %e5 : tensor<8x1xi32> -> tensor<8x4xi32>
    %off2 = arith.addi %b4, %b5 : tensor<8x4xi32>
    %ap2 = tt.addptr %sp2, %off2 : tensor<8x4x!tt.ptr<f32, 6>>, tensor<8x4xi32>
    tt.store %ap2, %cst2 : tensor<8x4x!tt.ptr<f32, 6>>
    tt.return
  }
}

// -----
//
// Test 2: All accesses inside a single scf.for loop body
//
// The nearest common dominator is the loop body block, so the alloc is placed
// INSIDE the loop, just before the first access there.  This is the key
// feature: the alloc is NOT hoisted to the function entry block.
//
// Input: an envelope store + a per-tile store, both inside scf.for.

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: tt.func @loop_body(
// CHECK:           scf.for
// CHECK:             ttg.local_alloc
// CHECK:             ttg.local_store
// CHECK:             ttg.memdesc_subslice
// CHECK:             ttg.local_store
// CHECK-NOT:       tt.store
// CHECK-NOT:       tt.splat
// CHECK-NOT:       tt.addptr
module {
  tt.func @loop_body(%arg0: !tt.ptr<f32, 6> {bishengir.scratch_shm}) {
    %cst0 = arith.constant dense<0.0> : tensor<8x8xf32>
    %cst1 = arith.constant dense<1.0> : tensor<8x4xf32>
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c4_idx = arith.constant 4 : index
    %res = scf.for %iv = %c0_idx to %c4_idx step %c1_idx iter_args(%acc = %cst0) -> (tensor<8x8xf32>) {
      // Envelope store (tileSize=8)
      %sp0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x8x!tt.ptr<f32, 6>>
      %r0  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e0  = tt.expand_dims %r0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
      %b0  = tt.broadcast %e0 : tensor<1x8xi32> -> tensor<8x8xi32>
      %r1  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e1  = tt.expand_dims %r1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %b1  = tt.broadcast %e1 : tensor<8x1xi32> -> tensor<8x8xi32>
      %off0 = arith.addi %b0, %b1 : tensor<8x8xi32>
      %ap0 = tt.addptr %sp0, %off0 : tensor<8x8x!tt.ptr<f32, 6>>, tensor<8x8xi32>
      tt.store %ap0, %cst0 : tensor<8x8x!tt.ptr<f32, 6>>
      // Per-tile store at tile 0 (tileSize=4)
      %sp1 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x4x!tt.ptr<f32, 6>>
      %r2  = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %e2  = tt.expand_dims %r2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %b2  = tt.broadcast %e2 : tensor<1x4xi32> -> tensor<8x4xi32>
      %r3  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e3  = tt.expand_dims %r3 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %b3  = tt.broadcast %e3 : tensor<8x1xi32> -> tensor<8x4xi32>
      %off1 = arith.addi %b2, %b3 : tensor<8x4xi32>
      %ap1 = tt.addptr %sp1, %off1 : tensor<8x4x!tt.ptr<f32, 6>>, tensor<8x4xi32>
      tt.store %ap1, %cst1 : tensor<8x4x!tt.ptr<f32, 6>>
      scf.yield %cst0 : tensor<8x8xf32>
    }
    tt.return
  }
}

// -----
//
// Test 3: Accesses in two different scf.for loops
//
// The nearest common dominator is the function entry block (the parent of both
// loops).  The alloc is placed BEFORE the first loop, not inside either one.

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: tt.func @cross_loop(
// CHECK:           ttg.local_alloc
// CHECK:           scf.for
// CHECK:             ttg.local_store
// CHECK:           scf.for
// CHECK:             ttg.memdesc_subslice
// CHECK:             ttg.local_store
// CHECK-NOT:       tt.store
// CHECK-NOT:       tt.splat
// CHECK-NOT:       tt.addptr
module {
  tt.func @cross_loop(%arg0: !tt.ptr<f32, 6> {bishengir.scratch_shm}) {
    %cst0 = arith.constant dense<0.0> : tensor<8x8xf32>
    %cst1 = arith.constant dense<1.0> : tensor<8x4xf32>
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c4_idx = arith.constant 4 : index
    // First loop: envelope store
    %r1 = scf.for %iv1 = %c0_idx to %c4_idx step %c1_idx iter_args(%acc1 = %cst0) -> (tensor<8x8xf32>) {
      %sp0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x8x!tt.ptr<f32, 6>>
      %r0  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e0  = tt.expand_dims %r0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
      %b0  = tt.broadcast %e0 : tensor<1x8xi32> -> tensor<8x8xi32>
      %r1_ = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e1  = tt.expand_dims %r1_ {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %b1  = tt.broadcast %e1 : tensor<8x1xi32> -> tensor<8x8xi32>
      %off0 = arith.addi %b0, %b1 : tensor<8x8xi32>
      %ap0 = tt.addptr %sp0, %off0 : tensor<8x8x!tt.ptr<f32, 6>>, tensor<8x8xi32>
      tt.store %ap0, %cst0 : tensor<8x8x!tt.ptr<f32, 6>>
      scf.yield %cst0 : tensor<8x8xf32>
    }
    // Second loop: per-tile store at tile 0
    %r2 = scf.for %iv2 = %c0_idx to %c4_idx step %c1_idx iter_args(%acc2 = %cst1) -> (tensor<8x4xf32>) {
      %sp1 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x4x!tt.ptr<f32, 6>>
      %r2_ = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %e2  = tt.expand_dims %r2_ {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %b2  = tt.broadcast %e2 : tensor<1x4xi32> -> tensor<8x4xi32>
      %r3  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e3  = tt.expand_dims %r3 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %b3  = tt.broadcast %e3 : tensor<8x1xi32> -> tensor<8x4xi32>
      %off1 = arith.addi %b2, %b3 : tensor<8x4xi32>
      %ap1 = tt.addptr %sp1, %off1 : tensor<8x4x!tt.ptr<f32, 6>>, tensor<8x4xi32>
      tt.store %ap1, %cst1 : tensor<8x4x!tt.ptr<f32, 6>>
      scf.yield %cst1 : tensor<8x4xf32>
    }
    tt.return
  }
}

// -----
//
// Test 4: One access before scf.for + one access inside scf.for
//
// The nearest common dominator is the function entry block (the parent of
// both).  The alloc is placed before the first access (which is before the
// loop), not inside the loop.

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: tt.func @mixed(
// CHECK:           ttg.local_alloc
// CHECK:           ttg.memdesc_subslice
// CHECK:           ttg.local_store
// CHECK:           scf.for
// CHECK:             ttg.local_store
// CHECK-NOT:       tt.store
// CHECK-NOT:       tt.splat
// CHECK-NOT:       tt.addptr
module {
  tt.func @mixed(%arg0: !tt.ptr<f32, 6> {bishengir.scratch_shm}) {
    %cst0 = arith.constant dense<0.0> : tensor<8x8xf32>
    %cst1 = arith.constant dense<1.0> : tensor<8x4xf32>
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c4_idx = arith.constant 4 : index
    // Access BEFORE the loop: per-tile store at tile 0 (tileSize=4)
    %sp1 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x4x!tt.ptr<f32, 6>>
    %r2  = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %e2  = tt.expand_dims %r2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %b2  = tt.broadcast %e2 : tensor<1x4xi32> -> tensor<8x4xi32>
    %r3  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e3  = tt.expand_dims %r3 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %b3  = tt.broadcast %e3 : tensor<8x1xi32> -> tensor<8x4xi32>
    %off1 = arith.addi %b2, %b3 : tensor<8x4xi32>
    %ap1 = tt.addptr %sp1, %off1 : tensor<8x4x!tt.ptr<f32, 6>>, tensor<8x4xi32>
    tt.store %ap1, %cst1 : tensor<8x4x!tt.ptr<f32, 6>>
    // Access INSIDE the loop: envelope store (tileSize=8)
    %res = scf.for %iv = %c0_idx to %c4_idx step %c1_idx iter_args(%acc = %cst0) -> (tensor<8x8xf32>) {
      %sp0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x8x!tt.ptr<f32, 6>>
      %r0  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e0  = tt.expand_dims %r0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
      %b0  = tt.broadcast %e0 : tensor<1x8xi32> -> tensor<8x8xi32>
      %r1  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e1  = tt.expand_dims %r1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %b1  = tt.broadcast %e1 : tensor<8x1xi32> -> tensor<8x8xi32>
      %off0 = arith.addi %b0, %b1 : tensor<8x8xi32>
      %ap0 = tt.addptr %sp0, %off0 : tensor<8x8x!tt.ptr<f32, 6>>, tensor<8x8xi32>
      tt.store %ap0, %cst0 : tensor<8x8x!tt.ptr<f32, 6>>
      scf.yield %cst0 : tensor<8x8xf32>
    }
    tt.return
  }
}

// -----
//
// Test 5: First access inside scf.for, second access after scf.for
//
// The nearest common dominator is the function entry block (parent of both the
// loop body and the continuation).  The alloc is placed BEFORE the first loop
// (the earliest access site in the entry block), so it dominates both regions.

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: tt.func @first_in_loop(
// CHECK:           ttg.local_alloc
// CHECK:           scf.for
// CHECK:             ttg.local_store
// CHECK:           ttg.memdesc_subslice
// CHECK:           ttg.local_store
// CHECK-NOT:       tt.store
// CHECK-NOT:       tt.splat
// CHECK-NOT:       tt.addptr
module {
  tt.func @first_in_loop(%arg0: !tt.ptr<f32, 6> {bishengir.scratch_shm}) {
    %cst0 = arith.constant dense<0.0> : tensor<8x8xf32>
    %cst1 = arith.constant dense<1.0> : tensor<8x4xf32>
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c4_idx = arith.constant 4 : index
    // First access: INSIDE THE LOOP (envelope store, tileSize=8)
    %res = scf.for %iv = %c0_idx to %c4_idx step %c1_idx iter_args(%acc = %cst0) -> (tensor<8x8xf32>) {
      %sp0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x8x!tt.ptr<f32, 6>>
      %r0  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e0  = tt.expand_dims %r0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
      %b0  = tt.broadcast %e0 : tensor<1x8xi32> -> tensor<8x8xi32>
      %r1  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %e1  = tt.expand_dims %r1 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
      %b1  = tt.broadcast %e1 : tensor<8x1xi32> -> tensor<8x8xi32>
      %off0 = arith.addi %b0, %b1 : tensor<8x8xi32>
      %ap0 = tt.addptr %sp0, %off0 : tensor<8x8x!tt.ptr<f32, 6>>, tensor<8x8xi32>
      tt.store %ap0, %cst0 : tensor<8x8x!tt.ptr<f32, 6>>
      scf.yield %cst0 : tensor<8x8xf32>
    }
    // Second access: AFTER THE LOOP (per-tile store, tileSize=4)
    %sp1 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x4x!tt.ptr<f32, 6>>
    %r2  = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %e2  = tt.expand_dims %r2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %b2  = tt.broadcast %e2 : tensor<1x4xi32> -> tensor<8x4xi32>
    %r3  = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %e3  = tt.expand_dims %r3 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %b3  = tt.broadcast %e3 : tensor<8x1xi32> -> tensor<8x4xi32>
    %off1 = arith.addi %b2, %b3 : tensor<8x4xi32>
    %ap1 = tt.addptr %sp1, %off1 : tensor<8x4x!tt.ptr<f32, 6>>, tensor<8x4xi32>
    tt.store %ap1, %cst1 : tensor<8x4x!tt.ptr<f32, 6>>
    tt.return
  }
}
