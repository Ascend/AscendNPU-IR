// RUN: bishengir-opt %s -hivm-enable-stride-align | FileCheck %s

// Regression test for a rank-reducing subview whose non-unit subview stride
// introduces a result stride that is absent from the source memref layout.
// getDroppedDims must fall back to position-based matching instead of
// asserting when stride-based matching cannot identify a unique dropped dim.

// CHECK-LABEL: func.func @rank_reducing_subview
// CHECK: %[[SUBVIEW:.*]] = memref.subview %{{.*}}[0, 0, 0, 0] [1, 1, 2, 1] [2, 1, 1, 1]
// CHECK-SAME: memref<2x1x2x1xi32> to memref<1x2x1xi32, strided<[4, 1, 1]>>
// CHECK: memref.load %[[SUBVIEW]]

module {
  func.func @rank_reducing_subview() -> i32 {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x1x2x1xi32>
    %subview = memref.subview %alloc[0, 0, 0, 0] [1, 1, 2, 1] [2, 1, 1, 1] : memref<2x1x2x1xi32> to memref<1x2x1xi32, strided<[4, 1, 1]>>
    annotation.mark %subview {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x2x1xi32, strided<[4, 1, 1]>>
    %value = memref.load %subview[%c0, %c0, %c0] : memref<1x2x1xi32, strided<[4, 1, 1]>>
    return %value : i32
  }
}
