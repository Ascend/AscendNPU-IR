// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=40 enable-count-buffer-dma-opt" -split-input-file | FileCheck %s


// CHECK-LABEL: reduction_tile_multiple_results
// CHECK: %[[GENERIC:.*]]:2 = linalg.generic
// CHECK: %[[REDUCE:.*]]:2 = linalg.reduce
func.func @reduction_tile_multiple_results(%arg0: tensor<1024x10240xf32>, %arg1: tensor<1024x10240xf32>, %arg2: tensor<1024x10240xf32>,
                                           %out0: tensor<1024xf32>, %out1: tensor<1024xf32>, %out2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>}
{
  %red:2 = linalg.reduce
   ins(%arg0, %arg1 : tensor<1024x10240xf32>, tensor<1024x10240xf32>)
   outs(%out0, %out1: tensor<1024xf32>, tensor<1024xf32>) dimensions = [1]
   (%in0: f32, %in1: f32, %init0: f32, %init1: f32) {
      %1 = arith.addf %in0, %init0 : f32
      %2 = arith.addf %in1, %init1 : f32
      linalg.yield %1, %2 : f32, f32
    }
  
  %red1 = linalg.reduce
   ins(%arg2: tensor<1024x10240xf32>)
   outs(%out2: tensor<1024xf32>) dimensions = [1]
   (%in0: f32, %init0: f32) {
      %1 = arith.addf %in0, %init0 : f32
      linalg.yield %1 : f32
    }
  return %red#0, %red#1, %red1 : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
}

// -----

// CHECK-LABEL: @test_align_reduction_dim_tiling_func_tiling_function
// CHECK-DAG:     %[[TILING_KEY:.*]] = arith.constant 0 : i64
// CHECK-DAG:     %[[DIM0_TILE:.*]] = arith.constant 10 : i64
// CHECK-DAG:     %[[DIM1_TILE:.*]] = arith.constant 32 : i64
// CHECK-DAG:     %[[DIM2_TILE:.*]] = arith.constant 16 : i64
// CHECK:         return %[[TILING_KEY]], %[[DIM0_TILE]], %[[DIM1_TILE]], %[[DIM2_TILE]]
module {
  func.func @test_align_reduction_dim_tiling_func(%arg0: tensor<24x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<24x384xf32>, %arg3: tensor<24x32xf32>, %arg4: tensor<24x32xf32>) -> (tensor<24x32xf32>, tensor<24x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = tensor.empty() : tensor<24x32x12xf32>
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [24, 32, 12] : tensor<24x384xf32> into tensor<24x32x12xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1]] output_shape [32, 12] : tensor<384xf32> into tensor<32x12xf32>
    %broadcasted = linalg.broadcast ins(%expanded_0 : tensor<32x12xf32>) outs(%0 : tensor<24x32x12xf32>) dimensions = [0]
    %expanded_1 = tensor.expand_shape %arg2 [[0], [1, 2]] output_shape [24, 32, 12] : tensor<24x384xf32> into tensor<24x32x12xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %broadcasted : tensor<24x32x12xf32>, tensor<24x32x12xf32>) outs(%0 : tensor<24x32x12xf32>) -> tensor<24x32x12xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_1, %broadcasted : tensor<24x32x12xf32>, tensor<24x32x12xf32>) outs(%0 : tensor<24x32x12xf32>) -> tensor<24x32x12xf32>
    %reduced:2 = linalg.reduce ins(%1, %2 : tensor<24x32x12xf32>, tensor<24x32x12xf32>) outs(%arg3, %arg4 : tensor<24x32xf32>, tensor<24x32xf32>) dimensions = [2]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_2: f32, %init: f32, %init_3: f32) {
        %3 = arith.addf %in, %init : f32
        %4 = arith.addf %in_2, %init_3 : f32
        linalg.yield %3, %4 : f32, f32
      }
    return %reduced#0, %reduced#1 : tensor<24x32xf32>, tensor<24x32xf32>
  }
}
