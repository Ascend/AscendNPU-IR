// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-plan-memory | FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-plan-memory --hivm-lower-to-loops --hivm-decompose-op --canonicalize --convert-hivm-to-std -o /dev/null

!ub = memref<f32, #hivm.address_space<ub>>

// CHECK-LABEL: func.func @rank_zero
// CHECK: %[[SUM:.*]] = arith.addf
// CHECK-NEXT: memref.store %[[SUM]], %[[NEW:.*]][]
// CHECK-NEXT: %[[BACK:.*]] = hivm.hir.pointer_cast
// CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[NEW]][]
// CHECK-NEXT: memref.store %[[VALUE]], %[[BACK]][]
// CHECK-NEXT: scf.yield %[[BACK]]
func.func @rank_zero() -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %two = arith.constant 2.0 : f32
  %five = arith.constant 5.0 : f32
  %initial = memref.alloc() : !ub
  memref.store %five, %initial[] : !ub
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%prev = %initial) -> (!ub) {
    %new = memref.alloc() : !ub
    memref.store %two, %new[] : !ub
    %a = memref.load %prev[] : !ub
    %b = memref.load %new[] : !ub
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %new[] : !ub
    scf.yield %new : !ub
  }
  %value = memref.load %result[] : !ub
  return %value : f32
}
