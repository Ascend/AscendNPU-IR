// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// -----
module {
  // CHECK-LABEL: func.func @test_mmadmx_post_per_channel_add_splitk
  // Bias tensor must be defined before the for (dominance check uses hookOp).
  func.func @test_mmadmx_post_per_channel_add_splitk(%bias_mem: memref<1x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %bias = bufferization.to_tensor %bias_mem restrict writable : memref<1x16xf32>
    %init = tensor.empty() : tensor<16x16xf32>
    // Split-K style loop yielding mmadmx accumulator, then post-loop per-channel add.
    %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg = %init) -> (tensor<16x16xf32>) {
      %mm = hivm.hir.mmadmxL1
        ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
            tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
            tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
        outs(%arg : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mm : tensor<16x16xf32>
    }
    %brc_out = tensor.empty() : tensor<16x16xf32>
    %brc = hivm.hir.vbrc ins(%bias : tensor<1x16xf32>)
      outs(%brc_out : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
    %add_out = tensor.empty() : tensor<16x16xf32>
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: normalized_init_or_bias
    // CHECK-SAME: tensor<1x16xf32>
    // CHECK-NOT: hivm.hir.vadd
    %add = hivm.hir.vadd ins(%acc, %brc : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%add_out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %add : tensor<16x16xf32>
  }
}

// -----
module {
  // CHECK-LABEL: func.func @test_mmadmx_mm_init_per_channel_add_splitk
  func.func @test_mmadmx_mm_init_per_channel_add_splitk(%bias: tensor<1x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %brc_out = tensor.empty() : tensor<16x16xf32>
    %brc = hivm.hir.vbrc ins(%bias : tensor<1x16xf32>)
      outs(%brc_out : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
    // Loop init is per-channel broadcast; mmadmx accumulates inside and folds bias.
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: normalized_init_or_bias
    // CHECK-SAME: tensor<1x16xf32>
    %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg = %brc) -> (tensor<16x16xf32>) {
      %mm = hivm.hir.mmadmxL1
        ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
            tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
            tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
        outs(%arg : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %mm : tensor<16x16xf32>
    }
    return %acc : tensor<16x16xf32>
  }
}

// -----
module attributes {hfusion.disableHfusionVectorize} {
  // CHECK-LABEL: func.func @test_mmadmx_skip_affinity_disable_hfusion_vectorize
  func.func @test_mmadmx_skip_affinity_disable_hfusion_vectorize() -> tensor<16x16xf32> {
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %c = tensor.empty() : tensor<16x16xf32>
    // Affinity / L0C normalization should be skipped under disableHfusionVectorize.
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-NOT: normalized_in_L0C
    %mm = hivm.hir.mmadmxL1
      ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
          tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
          tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
      outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mm : tensor<16x16xf32>
  }
}
