// RUN: bishengir-opt %s -hfusion-auto-schedule="enable-manage-host-resources=true" -split-input-file | FileCheck %s

module {
  func.func @foo(%arg0: tensor<10240xf32>, %arg1: tensor<10240xf32>, %arg2: tensor<10240xf32>, %arg3: tensor<10240xf32>) -> tensor<10240xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<10240xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<10240xf32>, tensor<10240xf32>) outs(%0 : tensor<10240xf32>) -> tensor<10240xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<10240xf32>, tensor<10240xf32>) outs(%0 : tensor<10240xf32>) -> tensor<10240xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%1, %2 : tensor<10240xf32>, tensor<10240xf32>) outs(%arg3 : tensor<10240xf32>) -> tensor<10240xf32>
    return %3 : tensor<10240xf32>
  }
  // CHECK: @main
  // CHECK-SAME: %[[ARG0:.*]]: tensor<10240xf32>, %[[ARG1:.*]]: tensor<10240xf32>, %[[ARG2:.*]]: tensor<10240xf32>, %[[ARG3:.*]]: tensor<10240xf32>
  func.func @main(%arg0: tensor<10240xf32>, %arg1: tensor<10240xf32>, %arg2: tensor<10240xf32>, %arg3: tensor<10240xf32>) -> tensor<10240xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK: %[[TILING:.*]]:5 = call @foo_tiling_function(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : ({{.*}}) -> (i64, i64, i64, i64, i64)
    // CHECK: call @foo_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[TILING]]#0, %[[TILING]]#1, %[[TILING]]#2, %[[TILING]]#3, %[[TILING]]#4)
    %0 = call @foo(%arg0, %arg1, %arg2, %arg3) : (tensor<10240xf32>, tensor<10240xf32>, tensor<10240xf32>, tensor<10240xf32>) -> tensor<10240xf32>
    return %0 : tensor<10240xf32>
  }
}

// -----

module {
  func.func @foo(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [1]
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf16> into tensor<?xf16>
    %broadcasted_2 = linalg.broadcast ins(%collapsed_1 : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
  // CHECK: @main
  // CHECK-SAME: %[[ARG0:.*]]: tensor<?x1xf16>, %[[ARG1:.*]]: tensor<1x?xf16>, %[[ARG2:.*]]: tensor<?x?xf16>
  func.func @main(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK: %[[TILING0:.*]]:6 = call @foo_tiling_function(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : ({{.*}}) -> (i64, i64, i64, i64, i64, i64)
    // CHECK: call @foo_1(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[TILING0]]#0, %[[TILING0]]#1, %[[TILING0]]#2, %[[TILING0]]#3, %[[TILING0]]#4, %[[TILING0]]#5)
    // CHECK: call @foo_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[TILING0]]#0, %[[TILING0]]#1, %[[TILING0]]#2, %[[TILING0]]#3, %[[TILING0]]#4, %[[TILING0]]#5)
    %0 = call @foo(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    // CHECK: %[[TILING1:.*]]:6 = call @foo_tiling_function(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : ({{.*}}) -> (i64, i64, i64, i64, i64, i64)
    // CHECK: call @foo_1(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[TILING1]]#0, %[[TILING1]]#1, %[[TILING1]]#2, %[[TILING1]]#3, %[[TILING1]]#4, %[[TILING1]]#5)
    // CHECK: call @foo_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[TILING1]]#0, %[[TILING1]]#1, %[[TILING1]]#2, %[[TILING1]]#3, %[[TILING1]]#4, %[[TILING1]]#5)
    %1 = call @foo(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    return %0, %1 : tensor<?x?xf16>, tensor<?x?xf16>
  }
}

// -----

module {
  func.func @foo(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf16>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [1]
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x?xf16> into tensor<?xf16>
    %broadcasted_2 = linalg.broadcast ins(%collapsed_1 : tensor<?xf16>) outs(%0 : tensor<?x?xf16>) dimensions = [0]
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %broadcasted_2 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %1 : tensor<?x?xf16>
  }
  // CHECK: @device_caller
  // CHECK: %[[ARG0:.*]]: tensor<?x1xf16>, %[[ARG1:.*]]: tensor<1x?xf16>, %[[ARG2:.*]]: tensor<?x?xf16>
  // CHECK: %[[ARG3:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}, %[[ARG4:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[ARG5:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[ARG6:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[ARG7:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, %[[ARG8:.*]]: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
  func.func @device_caller(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @foo_1(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]])
    // CHECK: call @foo_0(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]], %[[ARG8]])
    %0 = call @foo(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
  // CHECK: @main
  // CHECK: %[[ARG0:.*]]: tensor<?x1xf16>, %[[ARG1:.*]]: tensor<1x?xf16>, %[[ARG2:.*]]: tensor<?x?xf16>
  func.func @main(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) -> tensor<?x?xf16> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK: %[[TILING0:.*]]:6 = call @foo_tiling_function(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : ({{.*}}) -> (i64, i64, i64, i64, i64, i64)
    // CHECK: call @device_caller(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[TILING0]]#0, %[[TILING0]]#1, %[[TILING0]]#2, %[[TILING0]]#3, %[[TILING0]]#4, %[[TILING0]]#5)
    %0 = call @device_caller(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
}
