// RUN: bishengir-opt %s -hfusion-uplift-while-to-for -split-input-file | FileCheck %s

// Thin downstream wrapper around upstream
// scf::populateUpliftWhileToForPatterns (see
// third-party/llvm-project/.../UpliftWhileToFor.cpp and
// mlir/test/Dialect/SCF/uplift-while.mlir). These tests verify the
// downstream pass binding (`hfusion-uplift-while-to-for`) hits both the
// pattern-matching uplift case AND the non-matching fallback case
// (data-driven exit, must stay as scf.while so the downstream
// MultiBufferLoopAdapter alloca-based counter path can still kick in).

// -----
// Canonical for-shaped while: `before` block is a single arith.cmpi slt
// feeding scf.condition, `after` block contains a linear arith.addi. The
// pattern should fire and rewrite to scf.for.
//
// CHECK-LABEL: func.func @uplift_for_shaped_while_slt
//  CHECK-SAME: (%[[BEGIN:.*]]: index, %[[END:.*]]: index, %[[STEP:.*]]: index) -> index
//   CHECK-NOT: scf.while
//       CHECK: scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]] {
//       CHECK:   "test.body"(%[[I]]) : (index) -> ()
//       CHECK: }
func.func @uplift_for_shaped_while_slt(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.body"(%arg3) : (index) -> ()
    %added = arith.addi %arg3, %arg2 : index
    scf.yield %added : index
  }
  return %0 : index
}

// -----
// Same shape with arith.cmpi sgt (commuted comparison). Should also uplift.
//
// CHECK-LABEL: func.func @uplift_for_shaped_while_sgt
//   CHECK-NOT: scf.while
//       CHECK: scf.for
func.func @uplift_for_shaped_while_sgt(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi sgt, %arg1, %arg3 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.body"(%arg3) : (index) -> ()
    %added = arith.addi %arg3, %arg2 : index
    scf.yield %added : index
  }
  return %0 : index
}

// -----
// `arith.addi` with operands swapped (step + iv instead of iv + step).
// Upstream pattern handles commutativity so this should still uplift.
//
// CHECK-LABEL: func.func @uplift_for_shaped_while_addi_commuted
//   CHECK-NOT: scf.while
//       CHECK: scf.for
func.func @uplift_for_shaped_while_addi_commuted(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.body"(%arg3) : (index) -> ()
    %added = arith.addi %arg2, %arg3 : index
    scf.yield %added : index
  }
  return %0 : index
}

// -----
// While with additional non-IV iter-args (i32 / f32). The pattern should
// still uplift the IV part and keep the unrelated iter-args as scf.for
// iter_args.
//
// CHECK-LABEL: func.func @uplift_for_shaped_while_extra_iter_args
//   CHECK-NOT: scf.while
//       CHECK: %{{.*}}:2 = scf.for {{.*}} iter_args({{.*}}) -> (i32, f32)
func.func @uplift_for_shaped_while_extra_iter_args(%arg0: index, %arg1: index, %arg2: index) -> (i32, f32) {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2.0 : f32
  %0:3 = scf.while (%arg4 = %c1, %arg3 = %arg0, %arg5 = %c2) : (i32, index, f32) -> (i32, index, f32) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg4, %arg3, %arg5 : i32, index, f32
  } do {
  ^bb0(%arg4: i32, %arg3: index, %arg5: f32):
    %1 = "test.transform_i32"(%arg4) : (i32) -> i32
    %added = arith.addi %arg3, %arg2 : index
    %2 = "test.transform_f32"(%arg5) : (f32) -> f32
    scf.yield %1, %added, %2 : i32, index, f32
  }
  return %0#0, %0#2 : i32, f32
}

// -----
// While whose IV type is i64 (not index). Upstream pattern handles
// integer IVs too.
//
// CHECK-LABEL: func.func @uplift_for_shaped_while_i64
//   CHECK-NOT: scf.while
//       CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i64
func.func @uplift_for_shaped_while_i64(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
  %0 = scf.while (%arg3 = %arg0) : (i64) -> (i64) {
    %1 = arith.cmpi slt, %arg3, %arg1 : i64
    scf.condition(%1) %arg3 : i64
  } do {
  ^bb0(%arg3: i64):
    "test.body"(%arg3) : (i64) -> ()
    %added = arith.addi %arg3, %arg2 : i64
    scf.yield %added : i64
  }
  return %0 : i64
}

// -----
// Data-driven exit: the `before` block reads from memory (the exit
// condition isn't derived from a monotonic IV). The pattern must NOT
// fire so the loop stays as scf.while - the downstream MultiBufferLoopAdapter
// alloca-based counter is the right tool for these.
//
// CHECK-LABEL: func.func @keep_data_driven_while
//       CHECK: scf.while
//   CHECK-NOT: scf.for
func.func @keep_data_driven_while(%arg0: memref<1xi1>, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg1) : (index) -> (index) {
    %c0 = arith.constant 0 : index
    %cond = memref.load %arg0[%c0] : memref<1xi1>
    scf.condition(%cond) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.body"(%arg3) : (index) -> ()
    %added = arith.addi %arg3, %arg2 : index
    scf.yield %added : index
  }
  return %0 : index
}

// -----
// While whose `after` block lacks an arith.addi on the IV (in-loop
// updates the IV via a non-linear op). Must NOT uplift.
//
// CHECK-LABEL: func.func @keep_non_linear_while
//       CHECK: scf.while
//   CHECK-NOT: scf.for
func.func @keep_non_linear_while(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.body"(%arg3) : (index) -> ()
    %next = arith.muli %arg3, %arg2 : index
    scf.yield %next : index
  }
  return %0 : index
}
