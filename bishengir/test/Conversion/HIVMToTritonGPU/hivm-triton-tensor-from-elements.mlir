// RUN: bishengir-opt %s --convert-hivm-to-tritongpu | FileCheck %s --check-prefix=STAGE2

// STAGE2-LABEL: @tensor_from_elements
// STAGE2: tensor.from_elements
// STAGE2-NOT: func.func

module {
  func.func @tensor_from_elements(%out: memref<1xf32>, %scalar: f32) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %from_elements = tensor.from_elements %scalar : tensor<1xf32>
    hivm.hir.local_store ins(%out : memref<1xf32>, %from_elements : tensor<1xf32>)
    return
  }
}