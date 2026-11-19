// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-plan-memory | FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_950z --hivm-plan-memory | FileCheck %s --check-prefix=A5

!ub = memref<64xf32, #hivm.address_space<ub>>
!gm = memref<64xf32, #hivm.address_space<gm>>

// Starting at 5, two additions of an initialized 2 must produce 9. Using a
// single slot for the broadcast and incoming value instead produces 4.
// CHECK-LABEL: func.func @initialized
// CHECK-DAG: %[[FRESH_ADDR:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[INIT_ADDR:.*]] = arith.constant 256 : i64
// CHECK: %[[INIT:.*]] = hivm.hir.pointer_cast(%[[INIT_ADDR]])
// CHECK: iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[FRESH:.*]] = hivm.hir.pointer_cast(%[[FRESH_ADDR]])
// CHECK: hivm.hir.vbrc {{.*}} outs(%[[FRESH]]
// CHECK: hivm.hir.vadd ins(%[[PREV]], %[[FRESH]]
// CHECK: %[[CARRY:.*]] = hivm.hir.pointer_cast(%[[INIT_ADDR]])
// CHECK: hivm.hir.copy ins(%[[FRESH]] {{.*}} outs(%[[CARRY]]
// CHECK: scf.yield %[[CARRY]]
func.func @initialized(%out: !gm) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %two = arith.constant 2.0 : f32
  %five = arith.constant 5.0 : f32
  %initial = memref.alloc() : !ub
  hivm.hir.vbrc ins(%five : f32) outs(%initial : !ub)
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%prev = %initial) -> (!ub) {
    %fresh = memref.alloc() : !ub
    scope.scope : () -> () {
      hivm.hir.vbrc ins(%two : f32) outs(%fresh : !ub)
      scope.return
    } {no_inline}
    hivm.hir.vadd ins(%prev, %fresh : !ub, !ub) outs(%fresh : !ub)
    scf.yield %fresh : !ub
  }
  hivm.hir.store ins(%result : !ub) outs(%out : !gm)
  return
}

// A zero-trip loop returns the exact input. With positive trips, an alias
// retained by the caller still observes the original storage.
// CHECK-LABEL: func.func @external_init(
// CHECK-SAME: %[[INIT:[^:]+]]: memref
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[FRESH:.*]] = hivm.hir.pointer_cast
// CHECK: hivm.hir.vadd ins(%[[PREV]], %[[FRESH]]
// CHECK: %[[BACK:.*]] = hivm.hir.pointer_cast
// CHECK: hivm.hir.copy ins(%[[FRESH]] {{.*}} outs(%[[BACK]]
// CHECK: scf.yield %[[BACK]]
// CHECK: return %[[LOOP]]
func.func @external_init(%initial: !ub, %n: index, %two: f32) -> !ub {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %initial) -> (!ub) {
    %new = memref.alloc() : !ub
    hivm.hir.vbrc ins(%two : f32) outs(%new : !ub)
    hivm.hir.vadd ins(%prev, %new : !ub, !ub) outs(%new : !ub)
    scf.yield %new : !ub
  }
  return %result : !ub
}

// The single update can already consume its previous value in place.
// CHECK-LABEL: func.func @single_update
// CHECK: hivm.hir.vadd
// CHECK-NEXT: scf.yield
// A5-LABEL: func.func @single_update
// A5: hivm.hir.vadd
// A5-NEXT: scf.yield
func.func @single_update(%initial: !ub, %n: index) -> !ub {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %initial) -> (!ub) {
    %new = memref.alloc() : !ub
    hivm.hir.vadd ins(%prev, %prev : !ub, !ub) outs(%new : !ub)
    scf.yield %new : !ub
  }
  return %result : !ub
}

// CHECK-LABEL: func.func @scalar_writer
// CHECK: memref.store {{.*}}, %[[NEW:.*]][
// CHECK: memref.load %[[PREV:.*]][
// CHECK: memref.store {{.*}}, %[[NEW]][
// CHECK: hivm.hir.copy ins(%[[NEW]] {{.*}} outs(%[[BACK:[^ ]+]]
// CHECK: scf.yield %[[BACK]]
func.func @scalar_writer(%index: index, %initial: !ub, %n: index, %value: f32) -> !ub {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %initial) -> (!ub) {
    %new = memref.alloc() : !ub
    memref.store %value, %new[%index] : !ub
    %a = memref.load %prev[%index] : !ub
    %b = memref.load %new[%index] : !ub
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %new[%index] : !ub
    scf.yield %new : !ub
  }
  return %result : !ub
}
