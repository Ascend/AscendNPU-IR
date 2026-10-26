// RUN: bishengir-opt -one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s

// The select OpOperands always bufferize in-place, so no copy is inserted for
// them. The conflict between the write through the select result and the read
// of %a is resolved on the destination of the writing op instead.
// CHECK-LABEL: func.func @select_operand_conflict
// CHECK: %[[SEL:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} :
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.copy %[[SEL]], %[[ALLOC]]
// CHECK: %[[SV:.*]] = memref.subview %[[ALLOC]]
// CHECK: memref.copy %{{.*}}, %[[SV]]
func.func @select_operand_conflict(%c: i1, %a: tensor<4xf32>, %b: tensor<4xf32>,
                                   %v: tensor<2xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %s = arith.select %c, %a, %b : tensor<4xf32>
  %w = tensor.insert_slice %v into %s[0] [2] [1]
      : tensor<2xf32> into tensor<4xf32>
  return %w, %a : tensor<4xf32>, tensor<4xf32>
}
