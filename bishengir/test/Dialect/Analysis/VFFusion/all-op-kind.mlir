// RUN: bishengir-opt --vf-fusion="fusion-mode=all-op" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @simple_kernel_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: hfusion.bitcast
// CHECK: tensor.collapse
// CHECK-LABEL: func.func @simple_kernel(
// CHECK: tensor.empty
// CHECK: call @simple_kernel_fused_0
module {
  func.func @simple_kernel(%arg0: tensor<3x2xf16>, %arg1: tensor<3x2xf16>, %arg2: f16, %arg3: tensor<3x2xf32>) -> tensor<6xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<3x2xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg2 : tensor<3x2xf16>, f16) outs(%arg1 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<3x2xf16>) outs(%arg3 : tensor<3x2xf32>) -> tensor<3x2xf32>
    %3 = hfusion.bitcast ins(%2 : tensor<3x2xf32>) outs(%0 : tensor<3x2xi32>) -> tensor<3x2xi32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<3x2xi32> into tensor<6xi32>
    return %collapsed : tensor<6xi32>
  }
}

// -----

// CHECK-LABEL: func.func private @no_reshape_in_the_middle_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: hfusion.bitcast
// CHECK: tensor.collapse
// CHECK-LABEL: func.func @no_reshape_in_the_middle(
// CHECK: tensor.empty
// CHECK: call @no_reshape_in_the_middle_fused_0
// CHECK: linalg.elemwise_binary
module {
  func.func @no_reshape_in_the_middle(%arg0: tensor<3x2xf16>, %arg1: tensor<3x2xf16>, %arg2: f16, %arg3: tensor<3x2xf32>) -> tensor<6xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<3x2xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg2 : tensor<3x2xf16>, f16) outs(%arg1 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<3x2xf16>) outs(%arg3 : tensor<3x2xf32>) -> tensor<3x2xf32>
    %3 = hfusion.bitcast ins(%2 : tensor<3x2xf32>) outs(%0 : tensor<3x2xi32>) -> tensor<3x2xi32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<3x2xi32> into tensor<6xi32>
    %4 = tensor.empty() : tensor<6xi32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %collapsed : tensor<6xi32>, tensor<6xi32>) outs(%4 : tensor<6xi32>) -> tensor<6xi32>
    return %5 : tensor<6xi32>
  }
}

// -----

// CHECK-LABEL: func.func private @no_reshape_in_the_middle_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: tensor.collapse
// CHECK: tensor.collapse
// CHECK-LABEL: func.func @no_reshape_in_the_middle(
// CHECK: tensor.empty
// CHECK: call @no_reshape_in_the_middle_fused_0
// CHECK: linalg.elemwise_binary
module {
  func.func @no_reshape_in_the_middle(%arg0: tensor<1x3x2xf16>, %arg1: tensor<1x3x2xf16>, %arg2: f16, %arg3: tensor<3x2xf16>) -> tensor<6xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<1x3x2xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg2 : tensor<1x3x2xf16>, f16) outs(%arg1 : tensor<1x3x2xf16>) -> tensor<1x3x2xf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<1x3x2xf16> into tensor<1x6xf16>
    %collapsed_1 = tensor.collapse_shape %collapsed [[0, 1]] : tensor<1x6xf16> into tensor<6xf16>
    %4 = tensor.empty() : tensor<6xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed_1, %collapsed_1 : tensor<6xf16>, tensor<6xf16>) outs(%4 : tensor<6xf16>) -> tensor<6xf16>
    return %5 : tensor<6xf16>
  }
}

// -----

// CHECK-LABEL: func.func private @no_reshape_in_the_middle_fused_0(
// CHECK: tensor.collapse
// CHECK: tensor.collapse
// CHECK: tensor.collapse
// CHECK-LABEL: func.func @no_reshape_in_the_middle(
// CHECK: tensor.empty
// CHECK: call @no_reshape_in_the_middle_fused_0
// CHECK: linalg.elemwise_binary
module {
  func.func @no_reshape_in_the_middle(%arg0: tensor<1x1x3x2xf16>, %arg1: tensor<1x1x3x2xf16>, %arg2: f16, %arg3: tensor<3x2xf16>) -> tensor<6xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %1 = tensor.empty() : tensor<1x1x3x2xf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<1x1x3x2xf16> into tensor<1x1x6xf16>
    %collapsed_1 = tensor.collapse_shape %collapsed [[0], [1, 2]] : tensor<1x1x6xf16> into tensor<1x6xf16>
    %collapsed_2 = tensor.collapse_shape %collapsed_1 [[0, 1]] : tensor<1x6xf16> into tensor<6xf16>
    %4 = tensor.empty() : tensor<6xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed_2, %collapsed_2 : tensor<6xf16>, tensor<6xf16>) outs(%4 : tensor<6xf16>) -> tensor<6xf16>
    return %5 : tensor<6xf16>
  }
}

// -----

// CHECK-LABEL: func.func private @reshape_diamond_shape_test_fused_0(
// CHECK: tensor.collapse
// CHECK: tensor.collapse
// CHECK: tensor.collapse
// CHECK-LABEL: func.func @reshape_diamond_shape_test(
// CHECK: tensor.empty
// CHECK: call @reshape_diamond_shape_test_fused_0
// CHECK: linalg.elemwise_binary
module {
  func.func @reshape_diamond_shape_test(%arg0: tensor<1x1x3x2xf16>, %arg1: tensor<1x1x3x2xf16>, %arg2: f16, %arg3: tensor<3x2xf16>) -> tensor<6xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %1 = tensor.empty() : tensor<1x1x3x2xf16>
    %collapsed = tensor.collapse_shape %1 [[0], [1], [2, 3]] : tensor<1x1x3x2xf16> into tensor<1x1x6xf16>
    %collapsed_1 = tensor.collapse_shape %collapsed [[0], [1, 2]] : tensor<1x1x6xf16> into tensor<1x6xf16>
    %collapsed_2 = tensor.collapse_shape %collapsed_1 [[0, 1]] : tensor<1x6xf16> into tensor<6xf16>
    %4 = tensor.empty() : tensor<6xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed_2, %collapsed_2 : tensor<6xf16>, tensor<6xf16>) outs(%4 : tensor<6xf16>) -> tensor<6xf16>
    return %5 : tensor<6xf16>
  }
}

// -----

// CHECK-LABEL: func.func private @reshape_diamond_shape_test_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: tensor.collapse
// CHECK: linalg.elemwise_binary
// CHECK-LABEL: func.func @reshape_diamond_shape_test(
// CHECK: call @reshape_diamond_shape_test_fused_0
// CHECK: tensor.empty
// CHECK: linalg.elemwise_binary
module {
  func.func @reshape_diamond_shape_test(%arg0: tensor<1xf16>, %arg1: tensor<1xf16>, %arg2: f16, %arg3: tensor<3x2xf16>) -> tensor<1xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<1xf16>, tensor<1xf16>) outs(%0 : tensor<1xf16>) -> tensor<1xf16>
    %2 = tensor.empty() : tensor<1xf16>
    %collapsed = tensor.collapse_shape %1 [] : tensor<1xf16> into tensor<f16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %1 : tensor<1xf16>, tensor<1xf16>) outs(%2 : tensor<1xf16>) -> tensor<1xf16>
    %4 = tensor.empty() : tensor<1xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %collapsed : tensor<1xf16>, tensor<f16>) outs(%4 : tensor<1xf16>) -> tensor<1xf16>
    return %5 : tensor<1xf16>
  }
}

