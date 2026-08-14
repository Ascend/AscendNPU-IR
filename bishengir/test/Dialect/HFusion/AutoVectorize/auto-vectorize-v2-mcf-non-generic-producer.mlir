// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" 2>&1 | FileCheck %s

// Verify that `linalg.map` does not crash vectorization in the multi-consumer
// fusion case. The pass should complete successfully; the exact fusion shape
// is not checked.
//
// Upstream `replaceForWithNewSignature` only supports `linalg.generic` for
// the required loop yield. If a non-generic op like `linalg.map` is allowed
// into the multi-consumer fusion path without being generalized first, it
// leaves the untiled original alive alongside the tiled clone, triggering
// a vectorizer abort.
//
// The reduce -> max -> broadcast chain keeps the two consumers in separate
// loops, forcing the multi-consumer path.

// CHECK-NOT: Attempted to vectorize, but failed
// CHECK-NOT: AutoVectorizeV2 failed;

module {
  func.func @map_producer_not_fused_as_multiple_consumer(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64x128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = tensor.empty() : tensor<64x128xf32>
    %mapped = linalg.map { arith.mulf } ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>)
    %reduced = linalg.reduce ins(%mapped : tensor<64x128xf32>) outs(%0 : tensor<64xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    %2 = linalg.max ins(%arg2, %reduced : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%1 : tensor<64x128xf32>) dimensions = [1]
    %3 = linalg.sub ins(%mapped, %broadcasted : tensor<64x128xf32>, tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
    return %2, %3 : tensor<64xf32>, tensor<64x128xf32>
  }
}
