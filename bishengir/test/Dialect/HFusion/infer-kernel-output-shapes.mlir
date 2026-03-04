// RUN: bishengir-opt --hfusion-infer-out-shapes %s --split-input-file | FileCheck %s

func.func @test_elemwise_binary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @test_elemwise_binary
// CHECK-SAME: hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@test_elemwise_binary_infer_output_shape_function>
//
// CHECK: func.func @test_elemwise_binary_infer_output_shape_function
// CHECK: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_output_shape_function>}
// CHECK: %c1 = arith.constant 1
// CHECK: %c0 = arith.constant 0
// CHECK: %dim = tensor.dim %arg0, %c0
// CHECK: %dim_0 = tensor.dim %arg0, %c1
// CHECK: %from_elements = tensor.from_elements %dim, %dim_0
// CHECK: return %from_elements

// -----

func.func @test_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: tensor<?x?xf32>, %arg6: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.matmul_transpose_b ins(%arg3, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = linalg.matmul ins(%1, %2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: @test_matmul_infer_output_shape_function
// CHECK: %dim = tensor.dim %arg2, %c0
// CHECK: %dim_0 = tensor.dim %arg5, %c1
// CHECK: tensor.from_elements %dim, %dim_0

// -----

func.func @test_dot_add(%arg0: tensor<?x512xf32>, %arg1: tensor<512x4096xf32>, %arg2: tensor<?x4096xf32>) -> tensor<?x4096xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x512xf32>
  %0 = tensor.empty(%dim) : tensor<?x4096xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x512xf32>, tensor<512x4096xf32>) outs(%0 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  %3 = shape.shape_of %arg2 : tensor<?x4096xf32> -> tensor<2xindex>
  %c0_0 = arith.constant 0 : index
  %extracted = tensor.extract %3[%c0_0] : tensor<2xindex>
  %4 = tensor.empty(%extracted) : tensor<?x4096xf32>
  %mapped = linalg.elemwise_binary { add, fun = #linalg.binary_fn<add> } ins(%arg2, %2 : tensor<?x4096xf32>, tensor<?x4096xf32>) outs(%4: tensor<?x4096xf32>) -> tensor<?x4096xf32>
  return %mapped : tensor<?x4096xf32>
}
// CHECK-LABEL: @test_dot_add_infer_output_shape_function
// CHECK: %dim = tensor.dim %arg2, %c0
// CHECK: tensor.from_elements %dim, %c4096

// -----

func.func @test_multi_results(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: tensor<?x?xf32>, %arg6: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = linalg.matmul_transpose_a {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = linalg.matmul_transpose_b {fun = #linalg.binary_fn<add>} ins(%arg3, %arg4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg5 : tensor<?x?xf32>) -> tensor<?x?xf32>

    %3 = linalg.matmul {fun = #linalg.binary_fn<add>} ins(%1, %2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>

    return %2, %3 : tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: @test_multi_results_infer_output_shape_function
// CHECK: %dim = tensor.dim %arg3, %c0
// CHECK: %dim_0 = tensor.dim %arg4, %c0
// CHECK: %from_elements = tensor.from_elements %dim, %dim_0
// CHECK: %dim_1 = tensor.dim %arg2, %c0
// CHECK: %dim_2 = tensor.dim %arg5, %c1
// CHECK: %from_elements_3 = tensor.from_elements %dim_1, %dim_2
// CHECK: return %from_elements, %from_elements_3

// -----

mesh.mesh @mesh0(shape = 4)

func.func @test_mesh_op(%arg0: tensor<?x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<?x4096xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0: index
    %dim0 = tensor.dim %arg0, %c0: tensor<?x4096xf32>
    %0 = tensor.empty(%dim0) : tensor<?x4096xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x4096xf32>, tensor<4096x4096xf32>) outs(%0 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
    %all_reduce = mesh.all_reduce %1 on @mesh0 mesh_axes = [0] : tensor<?x4096xf32> -> tensor<?x4096xf32>
    return %all_reduce : tensor<?x4096xf32>
}
// CHECK-LABEL: @test_mesh_op_infer_output_shape_function
// CHECK: %dim = tensor.dim %arg0, %c0
// CHECK: %from_elements = tensor.from_elements %dim, %c4096

// -----

func.func @test_out_param(%arg0: tensor<?x4096xf16>, %arg1: tensor<6144x4096xf16>, %arg2: tensor<?x6144xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> tensor<?x6144xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = linalg.matmul_transpose_b ins(%arg0, %arg1: tensor<?x4096xf16>, tensor<6144x4096xf16>) outs(%arg2: tensor<?x6144xf16>) -> tensor<?x6144xf16>
    return %0: tensor<?x6144xf16>
}
// CHECK-LABEL: @test_out_param_infer_output_shape_function
// CHECK: (%arg0: tensor<?x4096xf16>, %arg1: tensor<6144x4096xf16>)
// CHECK-NOT: hacc.entry
// CHECK: %dim = tensor.dim %arg0, %c0
// CHECK: %from_elements = tensor.from_elements %dim, %c6144
