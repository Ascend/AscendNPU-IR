// RUN: bishengir-opt -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// With disableHfusionVectorize, the CCF Strategy path is skipped so the legacy
// DecomposeMatmulWithBiasPattern helpers (PerChannel / PostPerChannel / MMInit /
// Elementwise) are exercised for mmadmx.

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  // CHECK-LABEL: func.func @test_mmadmx_legacy_per_channel_add
  func.func @test_mmadmx_legacy_per_channel_add(%bias: tensor<1x16xf32>) -> tensor<4x16xf32> {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<4x8xf8E5M2>
    %b = tensor.empty() : tensor<8x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %brc_out = tensor.empty() : tensor<4x16xf32>
    %brc = hivm.hir.vbrc ins(%bias : tensor<1x16xf32>)
      outs(%brc_out : tensor<4x16xf32>) broadcast_dims = [0] -> tensor<4x16xf32>
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: %[[TRUE]]
    // CHECK-SAME: %[[BIAS:.*]]{{.*}}tensor<1x16xf32>
    // CHECK-NOT: hivm.hir.vbrc
    %mad = hivm.hir.mmadmxL1
      ins(%a, %b, %sa, %sb, %false, %c4, %c8, %c16 :
          tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>,
          tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
      outs(%brc : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %mad : tensor<4x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  // CHECK-LABEL: func.func @test_mmadmx_legacy_post_per_channel_splitk
  // Bias tensor must be defined before the for (dominance check uses hookOp).
  func.func @test_mmadmx_legacy_post_per_channel_splitk(%bias_mem: memref<1x16xf32>) -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16 = arith.constant 16 : index
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    %bias = bufferization.to_tensor %bias_mem restrict writable : memref<1x16xf32>
    %init = tensor.empty() : tensor<16x16xf32>
    // Init condition must be (iv == lb) for PostPerChannelSplitK recognition.
    %acc = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg = %init) -> (tensor<16x16xf32>) : i32 {
      %isFirst = arith.cmpi eq, %i, %c0_i32 : i32
      %mm = hivm.hir.mmadmxL1
        ins(%a, %b, %sa, %sb, %isFirst, %c16, %c16, %c16 :
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
    // CHECK-SAME: tensor<1x16xf32>
    // CHECK-NOT: hivm.hir.vadd
    %add = hivm.hir.vadd ins(%acc, %brc : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%add_out : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %add : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  // CHECK-LABEL: func.func @test_mmadmx_legacy_mm_init_per_channel_splitk
  func.func @test_mmadmx_legacy_mm_init_per_channel_splitk(%bias_mem: memref<1x16xf16>) -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    // MMInit pattern: load/cast/expand/vbrc feeds the for-iter init.
    %bias_t = bufferization.to_tensor %bias_mem restrict writable : memref<1x16xf16>
    %cast_out = tensor.empty() : tensor<1x16xf32>
    %casted = hivm.hir.vcast ins(%bias_t : tensor<1x16xf16>) outs(%cast_out : tensor<1x16xf32>) -> tensor<1x16xf32>
    %brc_out = tensor.empty() : tensor<16x16xf32>
    %brc = hivm.hir.vbrc ins(%casted : tensor<1x16xf32>)
      outs(%brc_out : tensor<16x16xf32>) broadcast_dims = [0] -> tensor<16x16xf32>
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: tensor<1x16xf16>
    %acc = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg = %brc) -> (tensor<16x16xf32>) : i32 {
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
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  // CHECK-LABEL: func.func @test_mmadmx_legacy_elementwise_add
  func.func @test_mmadmx_legacy_elementwise_add(%bias: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    // Full C bias + init=false → elementwise decompose (mmad + vadd).
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: hivm.hir.mmadmxL1
    // CHECK-SAME: %[[TRUE]]
    // CHECK: hivm.hir.vadd
    %mad = hivm.hir.mmadmxL1
      ins(%a, %b, %sa, %sb, %false, %c16, %c16, %c16 :
          tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
          tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
      outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mad : tensor<16x16xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">, hfusion.disableHfusionVectorize} {
  // CHECK-LABEL: func.func @test_mmadmx_legacy_elemwise_dynamic_init
  // Elementwise bias + dynamic init → conditional scf.if + vadd decompose.
  func.func @test_mmadmx_legacy_elemwise_dynamic_init(
      %bias: tensor<16x16xf32>, %iv: index) -> tensor<16x16xf32> {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %init = arith.cmpi eq, %iv, %c0 : index
    %a = tensor.empty() : tensor<16x16xf8E5M2>
    %b = tensor.empty() : tensor<16x16xf8E5M2>
    %sa = tensor.empty() : tensor<1xui8>
    %sb = tensor.empty() : tensor<1xui8>
    // CHECK: hivm.hir.mmadmxL1
    // CHECK: scf.if
    // CHECK: hivm.hir.vadd
    %mad = hivm.hir.mmadmxL1
      ins(%a, %b, %sa, %sb, %init, %c16, %c16, %c16 :
          tensor<16x16xf8E5M2>, tensor<16x16xf8E5M2>,
          tensor<1xui8>, tensor<1xui8>, i1, index, index, index)
      outs(%bias : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %mad : tensor<16x16xf32>
  }
}
