// RUN: bishengir-opt %s -create-preload -verify-each | FileCheck %s

// Identical address lists are not identical state: the carried pointer selects
// the previous iteration's slot, while the producer writes the current slot.
// The guarded result also preserves the initial/final slot for empty/drain ticks.
// CHECK-LABEL: func.func @carried_ring
// CHECK: %[[INIT:.*]] = hivm.hir.pointer_cast
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[PREVIOUS:.*]] = %[[INIT]])
// CHECK: %[[CURRENT:.*]] = hivm.hir.pointer_cast
// CHECK: %[[DELAYED:.*]] = hivm.hir.pointer_cast
// CHECK: %[[NEXT:.*]] = scf.if {{.*}} -> (memref<1xf32, #hivm.address_space<ub>>) {
// CHECK: %[[OLD:.*]] = memref.load %[[PREVIOUS]]
// CHECK: %[[NEW:.*]] = arith.addf %[[OLD]],
// CHECK: memref.store %[[NEW]], %[[CURRENT]]
// CHECK: scf.yield %[[CURRENT]]
// CHECK: } else {
// CHECK: scf.yield %[[PREVIOUS]]
// CHECK: scf.if
// CHECK: memref.load %[[DELAYED]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: %[[FINAL:.*]] = memref.load %[[LOOP]]
// CHECK: return %[[FINAL]]
func.func @carried_ring(%ub: index, %seen: memref<?xf32, #hivm.address_space<gm>>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a0 = arith.constant 0 : i64
  %a1 = arith.constant 64 : i64
  %one = arith.constant 1.0 : f32
  %init = hivm.hir.pointer_cast(%a0, %a1) : memref<1xf32, #hivm.address_space<ub>>
  %result = scf.for %i = %c0 to %ub step %c1
      iter_args(%previous = %init) -> memref<1xf32, #hivm.address_space<ub>> {
    %current = hivm.hir.pointer_cast(%a0, %a1) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %current {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
        : memref<1xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %old = memref.load %previous[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %new = arith.addf %old, %one : f32
      memref.store %new, %current[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %value = memref.load %current[%c0] : memref<1xf32, #hivm.address_space<ub>>
      memref.store %value, %seen[%i] : memref<?xf32, #hivm.address_space<gm>>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %current : memref<1xf32, #hivm.address_space<ub>>
  }
  %final = memref.load %result[%c0] : memref<1xf32, #hivm.address_space<ub>>
  return %final : f32
}

// A returned view must carry the guarded state, while other stage consumers
// keep views of their own ring slot.
// CHECK-LABEL: func.func @carried_ring_view
// CHECK: %[[VLOOP:.*]] = scf.for {{.*}} iter_args(%[[VPREV:.*]] = {{.*}})
// CHECK: %[[VNEXT:.*]] = scf.if {{.*}} -> (memref<1xf32, #hivm.address_space<ub>>) {
// CHECK: memref.load %[[VPREV]]
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield %[[VPREV]]
// CHECK: scf.yield %[[VNEXT]]
// CHECK: memref.load %[[VLOOP]]
func.func @carried_ring_view(%ub: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a0 = arith.constant 0 : i64
  %a1 = arith.constant 64 : i64
  %initial = hivm.hir.pointer_cast(%a0, %a1) : memref<2xf32, #hivm.address_space<ub>>
  %init = memref.subview %initial[0] [1] [1]
      : memref<2xf32, #hivm.address_space<ub>> to memref<1xf32, #hivm.address_space<ub>>
  %result = scf.for %i = %c0 to %ub step %c1
      iter_args(%previous = %init) -> memref<1xf32, #hivm.address_space<ub>> {
    %current = hivm.hir.pointer_cast(%a0, %a1) : memref<2xf32, #hivm.address_space<ub>>
    annotation.mark %current {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
        : memref<2xf32, #hivm.address_space<ub>>
    %next = scope.scope : () -> memref<1xf32, #hivm.address_space<ub>> {
      %view = memref.subview %current[0] [1] [1]
          : memref<2xf32, #hivm.address_space<ub>> to memref<1xf32, #hivm.address_space<ub>>
      %old = memref.load %previous[%c0] : memref<1xf32, #hivm.address_space<ub>>
      memref.store %old, %view[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return %view : memref<1xf32, #hivm.address_space<ub>>
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    scf.yield %next : memref<1xf32, #hivm.address_space<ub>>
  }
  %final = memref.load %result[%c0] : memref<1xf32, #hivm.address_space<ub>>
  return %final : f32
}
