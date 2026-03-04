// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=40 enable-deterministic-computing=false" -split-input-file | FileCheck %s

// CHECK: multicore_reduce_sum
// CHECK: mapping = [#hivm.block]
// CHECK: mapping = [#hivm.block]

func.func @multicore_reduce_sum(%arg0: tensor<?x3xf32>) -> tensor<3xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %0 = tensor.empty() : tensor<3xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<?x3xf32>) outs(%0 : tensor<3xf32>) dimensions = [0] 
    (%in: f32, %init: f32) {
      %1 = arith.addf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<3xf32>
}

// CHECK: multicore_reduce_max
// CHECK: mapping = [#hivm.block]
// CHECK: mapping = [#hivm.block]
func.func @multicore_reduce_max(%arg0: tensor<?x3xf32>) -> tensor<3xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %0 = tensor.empty() : tensor<3xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<?x3xf32>) outs(%0 : tensor<3xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %1 = arith.maximumf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<3xf32>
}

// CHECK: multicore_reduce_min
// CHECK: mapping = [#hivm.block]
// CHECK: mapping = [#hivm.block]
func.func @multicore_reduce_min(%arg0: tensor<?x3xf32>) -> tensor<3xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %0 = tensor.empty() : tensor<3xf32>
  %reduced = linalg.reduce ins(%arg0 : tensor<?x3xf32>) outs(%0 : tensor<3xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %1 = arith.minimumf %in, %init : f32
      linalg.yield %1 : f32
    }
  return %reduced : tensor<3xf32>
}