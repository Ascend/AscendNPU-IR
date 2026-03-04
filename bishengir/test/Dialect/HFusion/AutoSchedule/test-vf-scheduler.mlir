// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: @tile_parallel_axis_0(
// CHECK: linalg.elemwise_binary {{.*}} __tile_sizes__ = [1, 1, 1, 64]
// CHECK: linalg.transpose {{.*}} __tile_sizes__ = [1, 1, 1, 64]
module {
  func.func @tile_parallel_axis_0(%arg0: tensor<256x8x16x64xf32>) -> (tensor<256x8x16x64xf32>, tensor<256x16x8x64xf32>) 
  attributes {hivm.vector_function, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<256x16x8x64xf32>
    %1 = tensor.empty() : tensor<256x8x16x64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<256x8x16x64xf32>, tensor<256x8x16x64xf32>) outs(%1 : tensor<256x8x16x64xf32>) -> tensor<256x8x16x64xf32>
    %transposed = linalg.transpose ins(%2 : tensor<256x8x16x64xf32>) outs(%0 : tensor<256x16x8x64xf32>) permutation = [0, 2, 1, 3] 
    return %2, %transposed : tensor<256x8x16x64xf32>, tensor<256x16x8x64xf32>
  }
}

// -----

// CHECK-LABEL: @tile_reduction_axis_0(
// CHECK: linalg.reduce ins({{.*}} : tensor<1x64xf32>) outs({{.*}} : tensor<1xf32>)
// TODO: find a way to mark tile sizes for tiled reduction ops
module {
  func.func @tile_reduction_axis_0(%arg0: tensor<256x168xf32>) -> tensor<256xf32> 
  attributes {hivm.vector_function, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %3 = tensor.empty() : tensor<256xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<256x168xf32>) outs(%3 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    return %reduced : tensor<256xf32>
  }
}

// -----

// CHECK-LABEL: @transpose_and_tile_reduction_axis_0(
// CHECK: linalg.elemwise_binary {{.*}} __tile_sizes__ = [1, 1, 1, 64]
// CHECK: linalg.transpose {{.*}} __tile_sizes__ = [1, 1, 1, 64]
// TODO: find a way to mark tile sizes for tiled reduction ops
module {
  func.func @transpose_and_tile_reduction_axis_0(%arg0: tensor<256x8x16x64xf32>) -> (tensor<256x16x8x64xf32>, tensor<256x16x8xf32>) 
  attributes {hivm.vector_function, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<256x16x8x64xf32>
    %1 = tensor.empty() : tensor<256x8x16x64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<256x8x16x64xf32>, tensor<256x8x16x64xf32>) outs(%1 : tensor<256x8x16x64xf32>) -> tensor<256x8x16x64xf32>
    %transposed = linalg.transpose ins(%2 : tensor<256x8x16x64xf32>) outs(%0 : tensor<256x16x8x64xf32>) permutation = [0, 2, 1, 3] 
    %3 = tensor.empty() : tensor<256x16x8xf32>
    %reduced = linalg.reduce ins(%transposed : tensor<256x16x8x64xf32>) outs(%3 : tensor<256x16x8xf32>) dimensions = [3] 
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    return %transposed, %reduced : tensor<256x16x8x64xf32>, tensor<256x16x8xf32>
  }
}
