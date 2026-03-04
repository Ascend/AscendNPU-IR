// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=40" -split-input-file | FileCheck %s

// CHECK-LABEL: reduce_broadcast_reduce_0
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.reduce ins({{.*}} : tensor<1x2560xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
// CHECK: linalg.broadcast ins({{.*}} : tensor<1xf32>) outs({{.*}} : tensor<1x2560xf32>) dimensions = [1]
// CHECK: linalg.reduce ins({{.*}} : tensor<1x2560xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
module {
  func.func @reduce_broadcast_reduce_0(%arg0: tensor<2048x2560xbf16>) -> tensor<2048xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048x2560xf32>
    %1 = tensor.empty() : tensor<2048xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2048xf32>) -> tensor<2048xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<2048x2560xbf16>) outs(%0 : tensor<2048x2560xf32>) -> tensor<2048x2560xf32>
    %reduced = linalg.reduce ins(%3 : tensor<2048x2560xf32>) outs(%2 : tensor<2048xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    %broadcasted = linalg.broadcast ins(%reduced : tensor<2048xf32>) outs(%0 : tensor<2048x2560xf32>) dimensions = [1] 
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted : tensor<2048x2560xf32>, tensor<2048x2560xf32>) outs(%0 : tensor<2048x2560xf32>) -> tensor<2048x2560xf32>
    %reduced_0 = linalg.reduce ins(%4 : tensor<2048x2560xf32>) outs(%2 : tensor<2048xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    return %reduced_0 : tensor<2048xf32>
  }
}

// -----

// CHECK-LABEL: reduce_broadcast_reduce_1
// CHECK: scf.for
// CHECK: linalg.reduce ins({{.*}} : tensor<1x1x?xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1, 2]
// CHECK: scf.for
// CHECK: linalg.reduce ins({{.*}} : tensor<1x1x?xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1, 2]
module {
  func.func @reduce_broadcast_reduce_1(%arg0: tensor<2048x2560x5120xf32>) -> tensor<2048xf32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048x2560x5120xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<2048x2560x5120xf32>) outs(%0 : tensor<2048x2560x5120xf32>) -> tensor<2048x2560x5120xf32>
    %2 = tensor.empty() : tensor<2048xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2048xf32>) -> tensor<2048xf32>
    %reduced = linalg.reduce ins(%1 : tensor<2048x2560x5120xf32>) outs(%3 : tensor<2048xf32>) dimensions = [1, 2] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    %broadcasted = linalg.broadcast ins(%reduced : tensor<2048xf32>) outs(%0 : tensor<2048x2560x5120xf32>) dimensions = [1, 2] 
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted : tensor<2048x2560x5120xf32>, tensor<2048x2560x5120xf32>) outs(%0 : tensor<2048x2560x5120xf32>) -> tensor<2048x2560x5120xf32>
    %reduced_0 = linalg.reduce ins(%4 : tensor<2048x2560x5120xf32>) outs(%3 : tensor<2048xf32>) dimensions = [1, 2] 
      (%in: f32, %init: f32) {
        %5 = arith.addf %in, %init : f32
        linalg.yield %5 : f32
      }
    return %reduced_0 : tensor<2048xf32>
  }
}

// -----

// This test case validates a specific scenario where:
// 1. Tile multiple reduction axis.
// 2. Output consumer (i.e., store op) may have `implicit reduction axe`s that don't originate directly from a reduce op, 
//    but rather from a broadcast op where the broadcast axis matches a previously reduced axis.
// 3. When multiple output consumers share the same implicit reduction axes, they must be fused together to ensure:
//    - a single fused loop is available, otherwise there will be "cannot fuse into" problem caused by multiple loops to fuse into

// CHECK-LABEL: output_consumer_with_multiple_fusible_tile_reduction_loops_0
// CHECK: scf.for
// CHECK: linalg.reduce ins({{.*}} : tensor<1x1x1x?xf32>)
// CHECK: scf.for
// CHECK: linalg.reduce ins({{.*}} : tensor<1x1x1x?xf32>)
// CHECK: scf.for
// CHECK: hfusion.store {{.*}} ins({{.*}} : tensor<1x1x1x?xbf16>)
module {
  func.func @output_consumer_with_multiple_fusible_tile_reduction_loops_0(%arg0: tensor<16x32x4096x10240xf32>, %arg1: tensor<16x32x4096x10240xf32>) -> (tensor<16x16x2048x5120xbf16>, tensor<16x16x2048x5120xf32>) 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %cst = arith.constant 2.560000e+03 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16x2048x5120xbf16>
    %1 = tensor.empty() : tensor<16x16x2048x5120xf32>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [16, 16, 2048, 5120] [1, 1, 1, 1] : tensor<16x32x4096x10240xf32> to tensor<16x16x2048x5120xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, 0] [16, 16, 2048, 5120] [1, 1, 1, 1] : tensor<16x32x4096x10240xf32> to tensor<16x16x2048x5120xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_1 : tensor<16x16x2048x5120xf32>, tensor<16x16x2048x5120xf32>) outs(%1 : tensor<16x16x2048x5120xf32>) -> tensor<16x16x2048x5120xf32>
    %5 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<16x16x2048x5120xf32>) outs(%0 : tensor<16x16x2048x5120xbf16>) -> tensor<16x16x2048x5120xbf16>
    %reduced = linalg.reduce ins(%4 : tensor<16x16x2048x5120xf32>) outs(%3 : tensor<16x16xf32>) dimensions = [2, 3] 
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%reduced, %cst : tensor<16x16xf32>, f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %broadcasted = linalg.broadcast ins(%6 : tensor<16x16xf32>) outs(%1 : tensor<16x16x2048x5120xf32>) dimensions = [2, 3] 
    %reduced_2 = linalg.reduce ins(%broadcasted : tensor<16x16x2048x5120xf32>) outs(%3 : tensor<16x16xf32>) dimensions = [2, 3] 
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %broadcasted_3 = linalg.broadcast ins(%reduced_2 : tensor<16x16xf32>) outs(%1 : tensor<16x16x2048x5120xf32>) dimensions = [2, 3] 
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_3, %4 : tensor<16x16x2048x5120xf32>, tensor<16x16x2048x5120xf32>) outs(%1 : tensor<16x16x2048x5120xf32>) -> tensor<16x16x2048x5120xf32>
    return %5, %7 : tensor<16x16x2048x5120xbf16>, tensor<16x16x2048x5120xf32>
  }
}