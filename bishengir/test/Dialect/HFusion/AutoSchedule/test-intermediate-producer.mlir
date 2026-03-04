// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-auto-schedule="block-dim=20" -split-input-file -debug-only="hfusion-auto-schedule" 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG

// -----
// CHECK-LABEL: @model_0
// CHECK: hfusion.load {__intermediate_producer__
// CHECK: hfusion.cast {__intermediate_producer__
// CHECK: hfusion.load {__intermediate_producer__
// CHECK: linalg.elemwise_binary {__intermediate_producer__
// CHECK-NOT: hfusion.store {__intermediate_producer__}
func.func @model_0(%arg0: tensor<24x128x256x192xbf16>, %arg1: tensor<24x128x256x192xf32>, %arg2: tensor<24x128xf32>) -> tensor<24x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
  %collapsed = tensor.collapse_shape %arg2 [[0, 1]] : tensor<24x128xf32> into tensor<3072xf32>
  %collapsed_0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<24x128x256x192xbf16> into tensor<3072x49152xbf16>
  %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1], [2, 3]] : tensor<24x128x256x192xf32> into tensor<3072x49152xf32>
  %0 = tensor.empty() : tensor<3072x49152xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed_0 : tensor<3072x49152xbf16>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %collapsed_1 : tensor<3072x49152xf32>, tensor<3072x49152xf32>) outs(%0 : tensor<3072x49152xf32>) -> tensor<3072x49152xf32>
  %reduced = linalg.reduce ins(%2 : tensor<3072x49152xf32>) outs(%collapsed : tensor<3072xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %3 = arith.addf %in, %init : f32
      linalg.yield %3 : f32
    }
  %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [24, 128] : tensor<3072xf32> into tensor<24x128xf32>
  return %expanded : tensor<24x128xf32>
}

// -----

// CHECK-DEBUG-LABEL: @test_slice_from_producer
// CHECK-DEBUG-SAME: %[[arg0:.*]]: tensor<32x16x8xf32>
// CHECK-DEBUG: tensor.extract_slice %[[arg0]]{{.*}}{{\[}}1, 1, 1] : tensor<32x16x8xf32> to tensor<32x16x4xf32>
// CHECK-DEBUG: %[[brc:.*]] = linalg.broadcast ins({{.*}} : tensor<32x16xf32>) outs({{.*}} : tensor<32x16x4xf32>)
// CHECK-DEBUG: linalg.elemwise_binary {{.*}} ins(%[[brc]], {{.*}}
func.func @test_slice_from_producer(%arg0: tensor<32x16x8xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<32x16x4xf32>) -> (tensor<32x16x4xf32>, tensor<32x16x4xf32>)
 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {

  %extracted_slice = tensor.extract_slice %arg0[0, 0, 4] [32, 16, 4] [1, 1, 1] : tensor<32x16x8xf32> to tensor<32x16x4xf32>
  %0 = tensor.empty() : tensor<32x16x4xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %arg2 : tensor<32x16x4xf32>, tensor<32x16x4xf32>) 
                                                             outs(%0 : tensor<32x16x4xf32>) -> tensor<32x16x4xf32>
  %2 = tensor.empty() : tensor<32x16x8xf32>
  %broadcasted = linalg.broadcast ins(%arg1 : tensor<32x16xf32>) outs(%2 : tensor<32x16x8xf32>) dimensions = [2]
  %extracted_slice_1 = tensor.extract_slice %broadcasted[0, 0, 4] [32, 16, 4] [1, 1, 1] : tensor<32x16x8xf32> to tensor<32x16x4xf32>
  %3 = tensor.empty() : tensor<32x16x4xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice_1, %arg2 : tensor<32x16x4xf32>, tensor<32x16x4xf32>) 
                                                             outs(%3 : tensor<32x16x4xf32>) -> tensor<32x16x4xf32>
  return %1, %4 : tensor<32x16x4xf32>, tensor<32x16x4xf32>
}

// -----

// CHECK-DEBUG-LABEL: @test_pad_as_producer
// CHECK-DEBUG: tensor.pad
// CHECK-DEBUG: {__intermediate_producer__} : tensor<16xf32> to tensor<32xf32>
func.func @test_pad_as_producer(%arg0: tensor<16xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %arg0 low[0] high[16] {
  ^bb0(%arg2: index):
    tensor.yield %cst : f32
  } : tensor<16xf32> to tensor<32xf32>
  %0 = tensor.empty() : tensor<32xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%padded, %arg1 : tensor<32xf32>, tensor<32xf32>) 
                                                             outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @test_fuse_concat_with_extract_slice_input_0
// CHECK: scf.for
// CHECK: tensor.concat
func.func @test_fuse_concat_with_extract_slice_input_0(%arg0: tensor<4096x1x16x64xbf16>, %arg1: tensor<4096x1x16x192xbf16>, %arg2: tensor<4096x1x1x64xbf16>, %arg3: tensor<4096x1x1x64xbf16>) -> tensor<4096x1x16x64xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %cst = arith.constant 1.000000e+00 : bf16
  %extracted_slice = tensor.extract_slice %arg1[0, 0, 0, 128] [4096, 1, 16, 64] [1, 1, 1, 1] : tensor<4096x1x16x192xbf16> to tensor<4096x1x16x64xbf16>
  %0 = tensor.empty() : tensor<4096x1x16x64xbf16>
  %collapsed = tensor.collapse_shape %arg2 [[0], [1, 2], [3]] : tensor<4096x1x1x64xbf16> into tensor<4096x1x64xbf16>
  %broadcasted = linalg.broadcast ins(%collapsed : tensor<4096x1x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) dimensions = [2]
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %broadcasted : tensor<4096x1x16x64xbf16>, tensor<4096x1x16x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0, 32] [4096, 1, 16, 32] [1, 1, 1, 1] : tensor<4096x1x16x64xbf16> to tensor<4096x1x16x32xbf16>
  %extracted_slice_1 = tensor.extract_slice %1[0, 0, 0, 0] [4096, 1, 16, 32] [1, 1, 1, 1] : tensor<4096x1x16x64xbf16> to tensor<4096x1x16x32xbf16>
  %2 = tensor.empty() : tensor<4096x1x16x32xbf16>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%extracted_slice_1 : tensor<4096x1x16x32xbf16>) outs(%2 : tensor<4096x1x16x32xbf16>) -> tensor<4096x1x16x32xbf16>
  %concat = tensor.concat dim(3) %extracted_slice_0, %3 : (tensor<4096x1x16x32xbf16>, tensor<4096x1x16x32xbf16>) -> tensor<4096x1x16x64xbf16>
  %collapsed_2 = tensor.collapse_shape %arg3 [[0], [1, 2], [3]] : tensor<4096x1x1x64xbf16> into tensor<4096x1x64xbf16>
  %broadcasted_3 = linalg.broadcast ins(%collapsed_2 : tensor<4096x1x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) dimensions = [2]
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %broadcasted_3 : tensor<4096x1x16x64xbf16>, tensor<4096x1x16x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  %5 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<4096x1x16x64xbf16>, tensor<4096x1x16x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%concat, %6 : tensor<4096x1x16x64xbf16>, tensor<4096x1x16x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%7, %5 : tensor<4096x1x16x64xbf16>, tensor<4096x1x16x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %8 : tensor<4096x1x16x64xbf16>, tensor<4096x1x16x64xbf16>) outs(%0 : tensor<4096x1x16x64xbf16>) -> tensor<4096x1x16x64xbf16>
  return %9 : tensor<4096x1x16x64xbf16>
}

// -----

// CHECK-LABEL: @test_fuse_concat_with_extract_slice_input_1
// CHECK: scf.for
// CHECK: tensor.concat
func.func @test_fuse_concat_with_extract_slice_input_1(%arg0: tensor<4096x1x1x64xbf16>, %arg1: tensor<4096x1x1x64xbf16>, %arg2: tensor<4096x1x1x64xbf16>, %arg3: tensor<4096x1x1x64xbf16>) -> tensor<4096x1x1x64xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %cst = arith.constant 1.000000e+00 : bf16
  %0 = tensor.empty() : tensor<4096x1x1x64xbf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<4096x1x1x64xbf16>, tensor<4096x1x1x64xbf16>) outs(%0 : tensor<4096x1x1x64xbf16>) -> tensor<4096x1x1x64xbf16>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0, 32] [4096, 1, 1, 32] [1, 1, 1, 1] : tensor<4096x1x1x64xbf16> to tensor<4096x1x1x32xbf16>
  %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0, 0] [4096, 1, 1, 32] [1, 1, 1, 1] : tensor<4096x1x1x64xbf16> to tensor<4096x1x1x32xbf16>
  %2 = tensor.empty() : tensor<4096x1x1x32xbf16>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%extracted_slice_0 : tensor<4096x1x1x32xbf16>) outs(%2 : tensor<4096x1x1x32xbf16>) -> tensor<4096x1x1x32xbf16>
  %concat = tensor.concat dim(3) %extracted_slice, %3 : (tensor<4096x1x1x32xbf16>, tensor<4096x1x1x32xbf16>) -> tensor<4096x1x1x64xbf16>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%concat, %arg0 : tensor<4096x1x1x64xbf16>, tensor<4096x1x1x64xbf16>) outs(%0 : tensor<4096x1x1x64xbf16>) -> tensor<4096x1x1x64xbf16>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %arg1 : tensor<4096x1x1x64xbf16>, tensor<4096x1x1x64xbf16>) outs(%0 : tensor<4096x1x1x64xbf16>) -> tensor<4096x1x1x64xbf16>
  return %5 : tensor<4096x1x1x64xbf16>
}

// -----

// CHECK-LABEL: @test_fuse_scalar_arith_op_0
// CHECK: scf.for
// CHECK: arith.divf
// CHECK: arith.divf
func.func @test_fuse_scalar_arith_op_0(%arg0: tensor<1152x4x2x2xf32>, %arg1: tensor<1152x4x2x2xf32>, %arg2: tensor<1152x4x2x2xf32>, %arg3: tensor<1152x4x2x2xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<1xf32>, %arg7: tensor<1xf32>) -> (tensor<1152x4x2x2xf32>, tensor<1152x4x2x2xf32>, tensor<1152x4x2x2xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %cst = arith.constant 1.000000e-15 : f32
  %cst_0 = arith.constant 9.99987125E-4 : f32
  %cst_1 = arith.constant 9.990000e-01 : f32
  %cst_2 = arith.constant 0.100000024 : f32
  %cst_3 = arith.constant 0.899999976 : f32
  %cst_4 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<18432xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<1152x4x2x2xf32> into tensor<18432xf32>
  %collapsed_5 = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<1152x4x2x2xf32> into tensor<18432xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_1, %collapsed_5 : f32, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %collapsed_6 = tensor.collapse_shape %arg2 [[0, 1, 2, 3]] : tensor<1152x4x2x2xf32> into tensor<18432xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_3, %collapsed_6 : f32, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %collapsed_7 = tensor.collapse_shape %arg3 [[0, 1, 2, 3]] : tensor<1152x4x2x2xf32> into tensor<18432xf32>
  %extracted = tensor.extract %arg4[] : tensor<f32>
  %3 = arith.divf %cst_4, %extracted : f32
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %3 : tensor<18432xf32>, f32) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %extracted_8 = tensor.extract %arg5[] : tensor<f32>
  %5 = arith.divf %cst_4, %extracted_8 : f32
  %collapsed_9 = tensor.collapse_shape %arg6 [] : tensor<1xf32> into tensor<f32>
  %extracted_10 = tensor.extract %collapsed_9[] : tensor<f32>
  %collapsed_11 = tensor.collapse_shape %arg7 [] : tensor<1xf32> into tensor<f32>
  %extracted_12 = tensor.extract %collapsed_11[] : tensor<f32>
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_2, %4 : f32, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %4 : tensor<18432xf32>, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %6 : tensor<18432xf32>, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cst_0, %7 : f32, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %expanded = tensor.expand_shape %8 [[0, 1, 2, 3]] output_shape [1152, 4, 2, 2] : tensor<18432xf32> into tensor<1152x4x2x2xf32>
  %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<18432xf32>, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %5 : tensor<18432xf32>, f32) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %expanded_13 = tensor.expand_shape %10 [[0, 1, 2, 3]] output_shape [1152, 4, 2, 2] : tensor<18432xf32> into tensor<1152x4x2x2xf32>
  %12 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%11 : tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %12 : f32, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_10, %13 : f32, tensor<18432xf32>) outs(%0 : tensor<18432xf32>) -> tensor<18432xf32>
  %expanded_14 = tensor.expand_shape %14 [[0, 1, 2, 3]] output_shape [1152, 4, 2, 2] : tensor<18432xf32> into tensor<1152x4x2x2xf32>
  return %expanded, %expanded_13, %expanded_14 : tensor<1152x4x2x2xf32>, tensor<1152x4x2x2xf32>, tensor<1152x4x2x2xf32>
}

// -----

// CHECK-LABEL: @test_only_mark_arith_op_with_producer_operand(
// CHECK-NOT: arith{{.*}}__intermediate_producer__
module {
  func.func @test_only_mark_arith_op_with_producer_operand(%arg0: tensor<?x2560xbf16>, %arg1: tensor<?x2560xbf16>, %arg2: i64) -> tensor<?x2560xbf16> 
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x2560xbf16>
    %0 = arith.cmpi slt, %dim, %c0 : index
    %1 = arith.select %0, %dim, %c0 : index
    %2 = arith.subi %1, %c1 : index
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [%2, 2560] [1, 1] : tensor<?x2560xbf16> to tensor<?x2560xbf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [%2, 2560] [1, 1] : tensor<?x2560xbf16> to tensor<?x2560xbf16>
    %3 = tensor.empty(%2) : tensor<?x2560xbf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %extracted_slice_0 : tensor<?x2560xbf16>, tensor<?x2560xbf16>) outs(%3 : tensor<?x2560xbf16>) -> tensor<?x2560xbf16>
    return %4 : tensor<?x2560xbf16>
  }
}