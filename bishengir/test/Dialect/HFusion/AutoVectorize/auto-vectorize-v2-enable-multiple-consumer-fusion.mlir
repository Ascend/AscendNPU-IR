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

// -----

// CHECK-DEFAULT-LABEL: func.func @vgather_blocks_multiple_consumer_fusion
// CHECK-DEFAULT-DAG: outlined-loop-target
// CHECK-DEFAULT-DAG: outlined-loop-target
// CHECK-DEFAULT-DAG: hivm.hir.vgather

// CHECK-ENABLE-LABEL: func.func @vgather_blocks_multiple_consumer_fusion
// CHECK-ENABLE: outlined-loop-target
// CHECK-ENABLE-NOT: outlined-loop-target
// CHECK-ENABLE: hivm.hir.vgather

#map = affine_map<(d0) -> (d0)>

func.func @vgather_blocks_multiple_consumer_fusion(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %idx: tensor<128xi32>) -> (tensor<128xf32>, tensor<128xf32>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %producer_init = tensor.empty() : tensor<128xf32>
  %producer = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xf32>) outs(%producer_init : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %res = math.exp %in : f32
    linalg.yield %res : f32
  } -> tensor<128xf32>

  %consumer_init = tensor.empty() : tensor<128xf32>
  %consumer = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%producer, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%consumer_init : tensor<128xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %res = arith.addf %lhs, %rhs : f32
    linalg.yield %res : f32
  } -> tensor<128xf32>

  %gather_init = tensor.empty() : tensor<128xf32>
  %gather = hivm.hir.vgather ins(%producer : tensor<128xf32>)
                             indices(%idx : tensor<128xi32>)
                             outs(%gather_init : tensor<128xf32>)
                             -> tensor<128xf32>
  return %consumer, %gather : tensor<128xf32>, tensor<128xf32>
}
