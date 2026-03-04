// RUN: bishengir-opt %s -hfusion-hoist-tensor-empty -split-input-file -verify-diagnostics | FileCheck %s

// CHECK:      func.func @test_matmul_infer_workspace_shape_function()
// CHECK-SAME:   hacc.function_kind = #hacc.function_kind<HOST>
// CHECK-SAME:   hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>
//
// CHECK:      func.func @test_matmul(%[[ARG0:.*]]: tensor<1024x1024xf32>, %[[ARG1:.*]]: tensor<1024x1024xf32>,
// CHECK:                             %[[ARG2:.*]]: memref<1048576xf32> {hacc.arg_type = #hacc.arg_type<workspace>})
// CHECK-SAME: hacc.infer_workspace_shape_function = #hacc.infer_workspace_shape_function<@test_matmul_infer_workspace_shape_function>
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      %[[SUBVIEW:.*]] = memref.subview %[[ARG2]][%[[C0]]] [1048576] [1]
// CHECK:      %[[REINTERPRETER_CAST:.*]] = memref.reinterpret_cast %[[SUBVIEW]] to offset: [%[[C0]]], sizes: [1048576], strides: [1]
// CHECK:      %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[REINTERPRETER_CAST]] restrict writable
// CHECK:      %[[EXPANDED:.*]] = tensor.expand_shape %[[TO_TENSOR]] {{\[}}[0, 1]]
// CHECK:      %[[RESULT:.*]] = hivm.hir.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%[[EXPANDED]] : tensor<1024x1024xf32>)
func.func @test_matmul(%arg0: tensor<1024x1024xf32>, %arg1 : tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>)
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
  %0 = tensor.empty() : tensor<1024x1024xf32>
  %1 = hivm.hir.matmul ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %1 : tensor<1024x1024xf32>
}

// -----

// CHECK:        func.func @test_matmul2_infer_workspace_shape_function()
// CHECK-SAME:     hacc.function_kind = #hacc.function_kind<HOST>
// CHECK-SAME:     hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>
//
// CHECK:        func.func @test_matmul2(
// CHECK-SAME:                          %[[ARG0:.*]]: tensor<1024x1024xf32>, %[[ARG1:.*]]: tensor<1024x1024xf32>, %[[ARG2:.*]]: tensor<1024x1024xf32>,
// CHECK-SAME:                          %[[ARG3:.*]]: memref<2097152xf32> {hacc.arg_type = #hacc.arg_type<workspace>}
// CHECK-SAME:     hacc.infer_workspace_shape_function = #hacc.infer_workspace_shape_function<@test_matmul2_infer_workspace_shape_function>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[C1048576:.*]] = arith.constant 1048576 : index
// CHECK:        %[[SUBVIEW_0:.*]] = memref.subview %[[ARG3]][%[[C0]]] [1048576] [1]
// CHECK:        %[[REINTERPRETER_CAST_0:.*]] = memref.reinterpret_cast %[[SUBVIEW_0]] to offset: [%[[C0]]], sizes: [1048576], strides: [1]
// CHECK:        %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[REINTERPRETER_CAST_0]] restrict writable
// CHECK:        %[[EXPANDED_0:.*]] = tensor.expand_shape %[[TO_TENSOR_0]] {{\[}}[0, 1]]
// CHECK:        %[[OFFSET:.*]] = arith.addi %[[C0]], %[[C1048576]] : index
// CHECK:        %[[MATMUL_0:.*]] = hivm.hir.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%[[EXPANDED_0]] : tensor<1024x1024xf32>)
// CHECK:        %[[SUBVIEW_1:.*]] = memref.subview %[[ARG3]][%[[OFFSET]]] [1048576] [1]
// CHECK:        %[[REINTERPRETER_CAST_1:.*]] = memref.reinterpret_cast %[[SUBVIEW_1]] to offset: [%[[OFFSET]]], sizes: [1048576], strides: [1]
// CHECK:        %[[TO_TENSOR_1:.*]] = bufferization.to_tensor %[[REINTERPRETER_CAST_1]] restrict writable
// CHECK:        %[[EXPANDED_1:.*]] = tensor.expand_shape %[[TO_TENSOR_1]] {{\[}}[0, 1]]
// CHECK:        %[[MATMUL_1:.*]] = hivm.hir.matmul ins(%[[MATMUL_0]], %[[ARG2]] : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%[[EXPANDED_1]] : tensor<1024x1024xf32>)
func.func @test_matmul2(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>} {
  %0 = tensor.empty() : tensor<1024x1024xf32>
  %1 = hivm.hir.matmul ins(%arg0, %arg1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %2 = tensor.empty() : tensor<1024x1024xf32>
  %3 = hivm.hir.matmul ins(%1, %arg2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %3 : tensor<1024x1024xf32>
}
