// RUN: bishengir-opt %s --hivm-vectorize=phase=tail --verify-each \
// RUN:   | FileCheck %s

// The tail phase runs backend legalization followed by arithmetic/vector
// normalization and mask analysis. Mask analysis annotates the constant mask
// and the values that reach the same mask, so later lowering can reuse it.

module {
  func.func @masked(%src: memref<64xf32>, %dst: memref<64xf32>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %mask = vector.constant_mask [32] : vector<64xi1>
    %v = vector.transfer_read %src[%c0], %cst, %mask {in_bounds = [true]}
        : memref<64xf32>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0], %mask {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32>
    return
  }
}

// CHECK-LABEL: func.func @masked(
// CHECK: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CHECK-NEXT: annotation.mark %[[MASK]] {mask_op_idx = 0 : i32}
// CHECK: %[[VALUE:.*]] = vector.transfer_read %{{.*}}, %[[MASK]]
// CHECK-NEXT: annotation.mark %[[VALUE]] {reached_mask_ops_idx = 0 : i32}
// CHECK: vector.transfer_write %[[VALUE]], %{{.*}}, %[[MASK]]
