// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-mem-scope -split-input-file -verify-diagnostics

// -----
module {
  func.func @test_infer_mem_scope_mmadmx_tensor_error() {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %c = tensor.empty() : tensor<16x16xf32>
    // expected-error@+1 {{Run infer memory scope after bufferization}}
    %mad = hivm.hir.mmadmxL1
      ins(%a, %b, %sa, %sb, %true, %c16, %c16, %c16 :
          tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
          tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
      outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}

// -----
// Bias is a function arg (no memref.alloc root) → error path.
module {
  func.func @test_infer_mem_scope_mmadmx_missing_bias_alloc(
      %bias: memref<1x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %a = memref.alloc() : memref<16x16xf8E5M2>
    %b = memref.alloc() : memref<16x16xf8E5M2>
    %scaleA = memref.alloc() : memref<1xui8>
    %scaleB = memref.alloc() : memref<1xui8>
    %c = memref.alloc() : memref<16x16xf32>
    // expected-error@+1 {{Cannot find root memref.alloc for bias of this op}}
    hivm.hir.mmadmxL1
      ins(%a, %b, %scaleA, %scaleB, %true, %c16, %c16, %c16, %bias :
          memref<16x16xf8E5M2>, memref<16x16xf8E5M2>,
          memref<1xui8>, memref<1xui8>, i1, index, index, index,
          memref<1x16xf32>)
      outs(%c : memref<16x16xf32>)
    return
  }
}
