// RUN: bishengir-opt %s --hivm-vectorize=phase=buffers --verify-each \
// RUN:   | FileCheck %s --implicit-check-not=hivm.hir.vinterleave \
// RUN:   --implicit-check-not=hivm.hir.vdeinterleave \
// RUN:   --implicit-check-not=hivm.hir.vtranspose

// Buffer-movement matrix. Transpose, interleave and deinterleave are outlined
// into separate vector functions, lifted to linalg and vectorized.

module {
  func.func @move(%src: memref<8x16xf32>, %dst: memref<16x8xf32>,
                  %even: memref<64xf32>, %odd: memref<64xf32>,
                  %joined: memref<128xf32>) {
    hivm.hir.vtranspose ins(%src : memref<8x16xf32>)
        outs(%dst : memref<16x8xf32>) permutation = [1, 0]
    hivm.hir.vinterleave
        ins(%even, %odd : memref<64xf32>, memref<64xf32>)
        outs(%joined : memref<128xf32>) interleave_channel_nums = 2
    hivm.hir.vdeinterleave ins(%joined : memref<128xf32>)
        outs(%even, %odd : memref<64xf32>, memref<64xf32>)
    return
  }
}

// CHECK-LABEL: func.func private @move_fused_0(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: scf.for
// CHECK: memref.store
// CHECK-LABEL: func.func private @move_fused_1(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: scf.for
// CHECK: memref.store
// CHECK-LABEL: func.func private @move_fused_2(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: vector.transfer_read {{.*}} permutation_map = #{{.*}}
// CHECK: vector.transfer_write
// CHECK-LABEL: func.func @move(
// CHECK: call @move_fused_2
// CHECK-NEXT: call @move_fused_1
// CHECK-NEXT: call @move_fused_0
