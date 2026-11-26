// RUN: bishengir-opt %s --hivm-vectorize=phase=prepare-buffers --verify-each \
// RUN:   | FileCheck %s

// Vector functions must not allocate. The prepare-buffers phase outlines
// static allocations to the caller and passes them as extra arguments.

module {
  func.func private @vf(%src: memref<8xf32>, %dst: memref<8xf32>)
      attributes {hivm.vector_function} {
    %scratch = memref.alloc() : memref<8xf32>
    %c0 = arith.constant 0 : index
    %v = memref.load %src[%c0] : memref<8xf32>
    memref.store %v, %scratch[%c0] : memref<8xf32>
    memref.copy %scratch, %dst : memref<8xf32> to memref<8xf32>
    return
  }

  func.func @caller(%src: memref<8xf32>, %dst: memref<8xf32>) {
    call @vf(%src, %dst) : (memref<8xf32>, memref<8xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @vf(
// CHECK-SAME: %[[SRC:.*]]: memref<8xf32>, %[[DST:.*]]: memref<8xf32>,
// CHECK-SAME: %[[SCRATCH:.*]]: memref<8xf32>)
// CHECK-NOT: memref.alloc
// CHECK: memref.store %{{.*}}, %[[SCRATCH]]
// CHECK: memref.copy %[[SCRATCH]], %[[DST]]

// CHECK-LABEL: func.func @caller(
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32>
// CHECK: call @vf(%{{.*}}, %{{.*}}, %[[ALLOC]])
