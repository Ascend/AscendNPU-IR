// RUN: bishengir-opt %s --hivm-vectorize=phase=buffers --verify-each \
// RUN:   | FileCheck %s --implicit-check-not=hivm.hir.vbrc

// The buffers phase vectorizes buffer movement that appears after tensor
// vectorization. The movement is outlined into a vector function, lifted and
// scheduled with the same engines as the tensor path.

module {
  func.func @fill(%value: f32, %dst: memref<64xf32>) {
    hivm.hir.vbrc ins(%value : f32) outs(%dst : memref<64xf32>)
    return
  }
}

// CHECK-LABEL: func.func private @fill_fused_0(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: vector.broadcast %{{.*}} : f32 to vector<64xf32>
// CHECK: vector.transfer_write
// CHECK-LABEL: func.func @fill(
// CHECK: call @fill_fused_0
// CHECK-SAME: {hivm.vector_function, no_inline}
