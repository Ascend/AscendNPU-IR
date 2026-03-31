// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion --hfusion-auto-vectorize-v2 -split-input-file | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" -split-input-file | FileCheck %s --check-prefix=CHECK-ENABLE

// CHECK-DEFAULT-LABEL: func.func @enable_multiple_comsumer_fusion
// CHECK-DEFAULT: {"outlined-loop-target-4"}
// CHECK-DEFAULT: {"outlined-loop-target-3"}
// CHECK-DEFAULT: {"outlined-loop-target-2"}
// CHECK-DEFAULT: {"outlined-loop-target-1"}

// CHECK-ENABLE-LABEL: func.func @enable_multiple_comsumer_fusion
// CHECK-ENABLE-NOT: {"outlined-loop-target-3"}
// CHECK-ENABLE: {"outlined-loop-target-3"}
// CHECK-ENABLE: {"outlined-loop-target-2"}
// CHECK-ENABLE: {"outlined-loop-target-1"}

func.func @enable_multiple_comsumer_fusion(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64x128xf32>) 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = tensor.empty() : tensor<64x128xf32>
  // Default: linalg.mul will not be fused into linalg.reduce and linalg.sub loop, because
  // its two consumers cannot be fused into the same loop
  // Enable: linalg.mul will be cloned and fused into linalg.reduce and linalg.sub loop 
  %mul = linalg.mul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  
  %reduced = linalg.reduce ins(%mul : tensor<64x128xf32>) outs(%0 : tensor<64xf32>) dimensions = [1] 
    (%in: f32, %init: f32) {
    %7 = arith.addf %in, %init : f32
    linalg.yield %7 : f32
  }
  %max = linalg.max ins(%arg2, %reduced : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %brc = linalg.broadcast ins(%max : tensor<64xf32>) outs(%1 : tensor<64x128xf32>) dimensions = [1]
  %sub = linalg.sub ins(%mul, %brc : tensor<64x128xf32>, tensor<64x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %max, %sub : tensor<64xf32>, tensor<64x128xf32>
}
