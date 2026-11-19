// RUN: bishengir-opt %s --hivm-vectorize --verify-each | FileCheck %s
// RUN: bishengir-opt %s --hivm-vectorize --one-shot-bufferize=bufferize-function-boundaries --verify-each -o /dev/null

module {
  // CHECK-LABEL: func.func @sliced
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice
  // CHECK: %[[COPY:.*]] = bufferization.alloc_tensor() copy(%[[SLICE]])
  // CHECK: call @sliced_fused_0(%[[COPY]],
  func.func @sliced(%src: tensor<8xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
    %slice = tensor.extract_slice %src[2] [4] [1]
        : tensor<8xf32> to tensor<4xf32>
    %result = hivm.hir.vmul ins(%slice, %slice : tensor<4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
