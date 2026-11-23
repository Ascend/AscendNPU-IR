// RUN: bishengir-opt %s -create-preload -verify-each | FileCheck %s

#extent = affine_map<()[s0, s1] -> (s0 - s1)>
func.func private @write(memref<?xf32>)
func.func private @read(memref<?xf32>)

// The returned view uses the active tile's index in the producer and the
// preceding tile's index in the consumer.
// CHECK-LABEL: func.func @return_partial_tile(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<?xf32>, %[[LIMIT:[^:]+]]: index
// CHECK: scf.for %[[IV:.*]] =
// CHECK: %[[PREV:.*]] = arith.subi %[[IV]],
// CHECK: %[[PRODUCER_SIZE:.*]] = affine.apply {{.*}}[%[[LIMIT]], %[[IV]]]
// CHECK: %[[PRODUCER_VIEW:.*]] = memref.subview %[[BUFFER]][0] [%[[PRODUCER_SIZE]]] [1]
// CHECK: call @write(%[[PRODUCER_VIEW]])
// CHECK: %[[CONSUMER_SIZE:.*]] = affine.apply {{.*}}[%[[LIMIT]], %[[PREV]]]
// CHECK: %[[CONSUMER_VIEW:.*]] = memref.subview %[[BUFFER]][0] [%[[CONSUMER_SIZE]]] [1]
// CHECK: call @read(%[[CONSUMER_VIEW]])
func.func @return_partial_tile(%buffer: memref<?xf32>, %limit: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %limit step %c1 {
    %tile = scope.scope : () -> memref<?xf32> {
      %size = affine.apply #extent()[%limit, %iv]
      %view = memref.subview %buffer[0] [%size] [1]
        : memref<?xf32> to memref<?xf32>
      func.call @write(%view) : (memref<?xf32>) -> ()
      scope.return %view : memref<?xf32>
    } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}
    scope.scope : () -> () {
      func.call @read(%tile) : (memref<?xf32>) -> ()
      scope.return
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
  }
  return
}
