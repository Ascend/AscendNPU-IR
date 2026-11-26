// RUN: bishengir-opt %s --hfusion-select-hivm-vectorize | FileCheck %s --implicit-check-not=hivm.defer_vectorize

module {
  func.func @select_add(%a: tensor<16xf32>, %b: tensor<16xf32>,
                        %out: tensor<16xf32>) -> tensor<16xf32> {
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
        outs(%out : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }

  func.func @skip_unsupported(%a: tensor<16xf32>, %b: tensor<16xf32>,
                              %out: tensor<16xf32>) -> tensor<16xf32> {
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
        ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
        outs(%out : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}

// CHECK-LABEL: func.func @select_add
// CHECK: linalg.elemwise_binary
// CHECK: return
// CHECK-LABEL: func.func @skip_unsupported
// CHECK-NOT: hivm.defer_vectorize
// CHECK: return
