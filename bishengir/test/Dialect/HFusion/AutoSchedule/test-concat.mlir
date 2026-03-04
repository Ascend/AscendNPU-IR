// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-CONCAT

// CHECK-LABEL: @test_concat_producer(
// CHECK: scf.for
// CHECK-CONCAT-LABEL: @test_concat_producer(
// CHECK-CONCAT: %[[concat:.*]] = tensor.concat dim(0) {{.*}}, {{.*}} {__intermediate_producer__}
module {
  func.func @test_concat_producer(%arg0: tensor<1x2047xi64>) -> tensor<2047x2047xi32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<2047xi32>
    %1 = tensor.empty() : tensor<2047x2047xi32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2047xi64> into tensor<2047xi64>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<2047xi64>) outs(%0 : tensor<2047xi32>) -> tensor<2047xi32>
    %extracted_slice = tensor.extract_slice %2[2046] [1] [1] : tensor<2047xi32> to tensor<1xi32>
    %concat = tensor.concat dim(0) %2, %extracted_slice : (tensor<2047xi32>, tensor<1xi32>) -> tensor<2048xi32>
    %extracted_slice_1 = tensor.extract_slice %concat[1] [2047] [1] : tensor<2048xi32> to tensor<2047xi32>
    %broadcasted = linalg.broadcast ins(%extracted_slice_1 : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [1] 
    %broadcasted_1 = linalg.broadcast ins(%2 : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [0] 
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %broadcasted_1 : tensor<2047x2047xi32>, tensor<2047x2047xi32>) outs(%1 : tensor<2047x2047xi32>) -> tensor<2047x2047xi32>
    return %3 : tensor<2047x2047xi32>
  }
}

// -----

// CHECK-LABEL: @test_concat_store(
// CHECK: scf.for
// CHECK-CONCAT-LABEL: @test_concat_store(
// CHECK-CONCAT: %[[concat:.*]] = tensor.concat dim(0) {{.*}}, {{.*}} {__intermediate_producer__}
module {
  func.func @test_concat_store(%arg0: tensor<1x2047xi64>) -> tensor<2048xi32> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<2047xi32>
    %1 = tensor.empty() : tensor<2047x2047xi32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2047xi64> into tensor<2047xi64>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<2047xi64>) outs(%0 : tensor<2047xi32>) -> tensor<2047xi32>
    %extracted_slice = tensor.extract_slice %2[2046] [1] [1] : tensor<2047xi32> to tensor<1xi32>
    %concat = tensor.concat dim(0) %2, %extracted_slice : (tensor<2047xi32>, tensor<1xi32>) -> tensor<2048xi32>
    return %concat : tensor<2048xi32>
  }
}

// -----

// CHECK-LABEL: @test_concat_fuse_tile_reduction_loop(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: %[[slice0:.*]] = tensor.extract_slice
// CHECK: %[[load0:.*]] = hfusion.load {{{.*}}, __reduction0_axis1_fusible_producer__} ins(%[[slice0]]
// CHECK: %[[brc:.*]] = linalg.broadcast ins(%[[load0]] : tensor<?xf32>) outs({{.*}}) dimensions = [0]  {{{.*}}, __reduction0_axis1_fusible_producer__}
// CHECK: tensor.concat dim(0) %[[brc]], {{.*}} {{{.*}}, __reduction0_axis1_fusible_producer__} : ({{.*}}) -> tensor<1x?xf32>
module {
  func.func @test_concat_fuse_tile_reduction_loop(%arg0: tensor<40960xf32>, %arg1: tensor<40960xf32>) -> (tensor<32xf32>) 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<16x40960xf32>
    %1 = tensor.empty() : tensor<16x40960xf32>
    %2 = tensor.empty() : tensor<32x40960xf32>
    %3 = tensor.empty() : tensor<32xf32>
    %broadcasted_0 = linalg.broadcast ins(%arg0 : tensor<40960xf32>) outs(%0 : tensor<16x40960xf32>) dimensions = [0] 
    %broadcasted_1 = linalg.broadcast ins(%arg1 : tensor<40960xf32>) outs(%1 : tensor<16x40960xf32>) dimensions = [0] 
    %concat = tensor.concat dim(0) %broadcasted_0, %broadcasted_1 : (tensor<16x40960xf32>, tensor<16x40960xf32>) -> tensor<32x40960xf32>
    %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%concat, %concat : tensor<32x40960xf32>, tensor<32x40960xf32>) outs(%2 : tensor<32x40960xf32>) -> tensor<32x40960xf32>
    %reduced_0 = linalg.reduce ins(%mul : tensor<32x40960xf32>) outs(%3 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    return %reduced_0 : tensor<32xf32>
  }
}

// -----

module {
  // CHECK-NOT: tensor.concat
  // CHECK: scf.for
  // CHECK: tensor.concat
  func.func @test_dyn_concat(%arg0: tensor<16x?xf32>, %arg1: tensor<16x?xf32>) -> (tensor<32x?xf32>) 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<16x?xf32>, tensor<16x?xf32>) -> tensor<32x?xf32>
    return %concat : tensor<32x?xf32>
  }
}

// -----

// CHECK-LABEL: @test_tile_non_concat_dim_with_size_one(
// CHECK: tensor.concat dim(0) {{.*}} : (tensor<?x1x?xf32>, tensor<?x1x?xf32>) -> tensor<1x1x?xf32>
module {
  func.func @test_tile_non_concat_dim_with_size_one(%arg0: tensor<2x25600xf32>) -> tensor<4x1024x25600xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<4x1024x25600xf32>
    %1 = tensor.empty() : tensor<2x1024x25600xf32>
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<2x25600xf32>) outs(%1 : tensor<2x1024x25600xf32>) dimensions = [1] 
    %broadcasted_0 = linalg.broadcast ins(%arg0 : tensor<2x25600xf32>) outs(%1 : tensor<2x1024x25600xf32>) dimensions = [1] 
    %concat = tensor.concat dim(0) %broadcasted, %broadcasted_0 : (tensor<2x1024x25600xf32>, tensor<2x1024x25600xf32>) -> tensor<4x1024x25600xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%concat, %concat : tensor<4x1024x25600xf32>, tensor<4x1024x25600xf32>) outs(%0 : tensor<4x1024x25600xf32>) -> tensor<4x1024x25600xf32>
    return %2 : tensor<4x1024x25600xf32>
  }
}
