// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true" -split-input-file | FileCheck %s

// -----
// The source of the tail reshaping op is produced by an operation the dimension
// analyzer does not model (arith.select on tensors), so it never receives a
// collapse group. The flattener must leave such a tail operand alone instead of
// asking for a collapse group that does not exist.

// CHECK-LABEL: func.func @tail_expand_shape_uncollapsed_source
// CHECK: %[[SEL:.*]] = arith.select %{{.*}} : tensor<256xf32>
// CHECK: tensor.expand_shape %[[SEL]] {{\[\[}}0, 1]] output_shape [16, 16] : tensor<256xf32> into tensor<16x16xf32>
// CHECK: linalg.transpose
// CHECK: bufferization.materialize_in_destination
func.func @tail_expand_shape_uncollapsed_source(%arg0: memref<?xf32>, %arg1: i1) {
  %empty = tensor.empty() : tensor<16x16xf32>
  %a = arith.constant dense<1.0> : tensor<256xf32>
  %b = arith.constant dense<2.0> : tensor<256xf32>
  %sel = arith.select %arg1, %a, %b : tensor<256xf32>
  %exp = tensor.expand_shape %sel [[0, 1]] output_shape [16, 16] : tensor<256xf32> into tensor<16x16xf32>
  %tr = linalg.transpose ins(%exp : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) permutation = [1, 0]
  %col = tensor.collapse_shape %tr [[0, 1]] : tensor<16x16xf32> into tensor<256xf32>
  %out = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1]>>
  bufferization.materialize_in_destination %col in writable %out : (tensor<256xf32>, memref<256xf32, strided<[1]>>) -> ()
  return
}
