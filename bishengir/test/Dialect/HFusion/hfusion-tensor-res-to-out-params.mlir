// RUN: bishengir-opt -canonicalize-ext --hfusion-tensor-results-to-out-params -split-input-file %s | FileCheck %s
// RUN: bishengir-opt -canonicalize-ext --hfusion-tensor-results-to-out-params="include-symbols=target" -split-input-file %s | FileCheck %s -check-prefix=CHECK-INCLUDE-SYM
// RUN: bishengir-opt -canonicalize-ext --hfusion-tensor-results-to-out-params="enable-manage-host-resources=true" -split-input-file %s | FileCheck %s -check-prefix=CHECK-HOST-RESOURCE

module {
  // CHECK-LABEL: @test
  // CHECK: {{.*}}: tensor<8xf32>, {{.*}}: tensor<8xf32>, {{.*}}: tensor<8xf32>,
  // CHECK: %[[OUT:.*]]: tensor<8xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>})
  func.func @test(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    // CHECK: mul
    // CHECK: add
    // CHECK: outs(%[[OUT]] : tensor<8xf32>)
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = tensor.empty() : tensor<8xf32>
    %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
    return %3 : tensor<8xf32>
  }
  // CHECK-LABEL: @call_test
  // CHECK: {{.*}}: tensor<8xf32>, {{.*}}: tensor<8xf32>, {{.*}}: tensor<8xf32>,
  // CHECK: %[[OUT1:.*]]: tensor<8xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  // CHECK: %[[OUT2:.*]]: tensor<8xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}
  // CHECK: %[[OUT3:.*]]: tensor<8xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}
  func.func @call_test(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %0 = tensor.empty() : tensor<8xf32>
    // CHECK: mul
    // CHECK: add
    // CHECK: outs(%[[OUT1]] : tensor<8xf32>
    // CHECK: call @test(%arg0, %arg1, %arg2, %[[OUT2]])
    // CHECK: call @test(%arg0, %arg1, %arg2, %[[OUT3]])
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = tensor.empty() : tensor<8xf32>
    %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
    %4 = call @test(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %5 = call @test(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %3, %4, %5 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
  }
}

// -----


module {
  // CHECK-LABEL: @test_multi_outs
  // CHECK: {{.*}}: tensor<8xf32>, {{.*}}: tensor<8xf32>, {{.*}}: tensor<8xf32>,
  // CHECK: %[[OUT1:.*]]: tensor<8xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  // CHECK: %[[OUT2:.*]]: tensor<8xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}
  func.func @test_multi_outs(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %0 = tensor.empty() : tensor<8xf32>
    // CHECK: mul
    // CHECK-DAG: add
    // CHECK-DAG: outs(%[[OUT1]] : tensor<8xf32>)
    // CHECK-DAG: sub
    // CHECK-DAG: outs(%[[OUT2]] : tensor<8xf32>)
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = tensor.empty() : tensor<8xf32>
    %3 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
    %4 = tensor.empty() : tensor<8xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>, sub} ins(%1, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%4 : tensor<8xf32>) -> tensor<8xf32>
    return %3, %5 : tensor<8xf32>, tensor<8xf32>
  }
}

// -----

module {
  // CHECK-LABEL: @test_return_expanded_shape
  // CHECK: %[[ARG0:.*]]: tensor<24x192x192xf32>
  // CHECK: %[[ARG1:.*]]: tensor<24x192x1xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  func.func @test_return_expanded_shape(%arg0: tensor<24x192x192xf32>) -> tensor<24x192x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x192xf32>
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2]] : tensor<24x192x1xf32> into tensor<24x192xf32>
  // CHECK: linalg.reduce {{.*}} outs(%[[COLLAPSED:.*]] : tensor<24x192xf32>)
    %reduced = linalg.reduce ins(%arg0 : tensor<24x192x192xf32>) outs(%0 : tensor<24x192xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %1 = arith.addf %in, %init : f32
        linalg.yield %1 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [24, 192, 1] : tensor<24x192xf32> into tensor<24x192x1xf32>
    return %expanded : tensor<24x192x1xf32>
  }
}

// -----

module {
  // CHECK-LABEL: @test_return_collapsed_shape
  // CHECK: %[[ARG0:.*]]: tensor<24x192x192x1xf32>
  // CHECK: %[[ARG1:.*]]: tensor<24x192xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  func.func @test_return_collapsed_shape(%arg0: tensor<24x192x192x1xf32>) -> tensor<24x192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<24x192x1xf32>
    // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2]] output_shape {{\[}}24, 192, 1] : tensor<24x192xf32> into tensor<24x192x1xf32>
    // CHECK: linalg.reduce {{.*}} outs(%[[EXPANDED:.*]] : tensor<24x192x1xf32>)
      %reduced = linalg.reduce ins(%arg0 : tensor<24x192x192x1xf32>) outs(%0 : tensor<24x192x1xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %1 = arith.addf %in, %init : f32
        linalg.yield %1 : f32
      }
    %collapsed = tensor.collapse_shape %reduced [[0], [1, 2]] : tensor<24x192x1xf32> into tensor<24x192xf32>
    return %collapsed : tensor<24x192xf32>
  }
}


// -----

module {
  // CHECK-LABEL: @test_return_collapsed_shape_from_arg
  // CHECK: %[[ARG0:.*]]: tensor<24x192xf32>) -> tensor<24x192x1xf32>
    func.func @test_return_collapsed_shape_from_arg(%arg0: tensor<24x192xf32>) -> tensor<24x192x1xf32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [24, 192, 1] : tensor<24x192xf32> into tensor<24x192x1xf32>
    return %expanded : tensor<24x192x1xf32>
  }
}

// -----

module {
  // CHECK-LABEL: @test_return_collapsed_shape_from_reshape
  // CHECK: %[[ARG0:.*]]: tensor<24xf32>
  // CHECK: %[[ARG1:.*]]: tensor<24x1x1xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  func.func @test_return_collapsed_shape_from_reshape(%arg0: tensor<24xf32>) -> tensor<24x1x1xf32> {
    %0 = tensor.empty() : tensor<24xf32>
    // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1, 2]] : tensor<24x1x1xf32> into tensor<24xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<24xf32>, tensor<24xf32>) outs(%0 : tensor<24xf32>) -> tensor<24xf32>
    %expanded = tensor.expand_shape %1 [[0, 1]] output_shape [24, 1] : tensor<24xf32> into tensor<24x1xf32>
    %expanded_0 = tensor.expand_shape %expanded [[0], [1, 2]] output_shape [24, 1, 1] : tensor<24x1xf32> into tensor<24x1x1xf32>
    return %expanded_0 : tensor<24x1x1xf32>
  }
}

// -----

module {
  // CHECK-LABEL: @test_return_collapsed_shape_from_reshape_nop
  func.func @test_return_collapsed_shape_from_reshape_nop(%arg0: tensor<24xf32>, %arg1: tensor<24x1x1xf32>) -> tensor<24x1x1xf32> {
    %0 = tensor.empty() : tensor<24xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2]] : tensor<24x1x1xf32> into tensor<24xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg0 : tensor<24xf32>, tensor<24xf32>) outs(%collapsed : tensor<24xf32>) -> tensor<24xf32>
    %expanded = tensor.expand_shape %1 [[0, 1, 2]] output_shape [24, 1, 1] : tensor<24xf32> into tensor<24x1x1xf32>
    return %expanded : tensor<24x1x1xf32>
  }
}

// -----

module {
  func.func @caller(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = call @nested_caller(%arg1, %arg0, %arg2) : (tensor<?xf32>, index, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func @nested_caller(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = call @foo(%arg1, %arg0, %arg2) : (index, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @foo
  // CHECK: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>,
  // CHECK: %[[ARG2:.*]]: tensor<?xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  func.func @foo(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = tensor.empty(%arg0) : tensor<?xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

// -----

module {
  // CHECK-INCLUDE-SYM: func.func @target
  // CHECK-INCLUDE-SYM: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  func.func @target(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
  // CHECK-INCLUDE-SYM: func.func @not_target
  // CHECK-INCLUDE-SYM: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32>
  func.func @not_target(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

// -----

module {
  // CHECK-LABEL: @test_many_reshape_before_return
  // CHECK: %[[ARG0:.*]]: tensor<768x6144xf32>, %[[ARG1:.*]]: tensor<24x256x768xbf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>})
   func.func @test_many_reshape_before_return(%arg0: tensor<768x6144xf32>) -> tensor<24x256x768xbf16> {
    %0 = tensor.empty() : tensor<768x6144xbf16>
    // CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2, 3, 4], [5, 6]] output_shape {{\[}}24, 32, 1, 1, 8, 32, 24] : tensor<24x256x768xbf16> into tensor<24x32x1x1x8x32x24xbf16>
    // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[EXPANDED]] {{\[}}[0, 1, 2, 3], [4, 5, 6]] : tensor<24x32x1x1x8x32x24xbf16> into tensor<768x6144xbf16>
    // CHECK: hfusion.cast ins(%[[ARG0]] : tensor<768x6144xf32>) outs(%[[COLLAPSED]] : tensor<768x6144xbf16>)
    %1 = hfusion.cast ins(%arg0 : tensor<768x6144xf32>) outs(%0 : tensor<768x6144xbf16>) -> tensor<768x6144xbf16>
    %expanded = tensor.expand_shape %1 [[0, 1, 2, 3], [4, 5, 6]] output_shape [24, 32, 1, 1, 8, 32, 24] : tensor<768x6144xbf16> into tensor<24x32x1x1x8x32x24xbf16>
    %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3, 4], [5, 6]] : tensor<24x32x1x1x8x32x24xbf16> into tensor<24x256x768xbf16>
    return %collapsed : tensor<24x256x768xbf16>
  }
}

// -----

module {
  // CHECK-LABEL: @test_many_reshape_from_many_return
    func.func @test_many_reshape_from_many_return(%arg0: tensor<768x4x3072xf32>, %arg1: tensor<768x4x3072xf32>, %arg2: tensor<768x4xf32>) -> (tensor<24x128xf32>, tensor<24x128x1x1xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<768x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<768x4xf32>) -> tensor<768x4xf32>
    // CHECK-DAG: %[[EXPANDED:.*]] = tensor.expand_shape {{.*}} {{\[}}[0], [1, 2, 3, 4]] output_shape {{\[}}24, 32, 4, 1, 1]
    // CHECK-DAG: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[EXPANDED]] {{\[}}[0, 1], [2, 3, 4]]
    // CHECK-DAG: %[[EXPANDED_0:.*]] = tensor.expand_shape {{.*}} {{\[}}[0], [1, 2], [3], [4]] output_shape {{\[}}24, 32, 4, 1, 1]
    // CHECK-DAG: %[[COLLAPSED_1:.*]] = tensor.collapse_shape %[[EXPANDED_0]] {{\[}}[0, 1], [2, 3, 4]]
    // CHECK: linalg.reduce ins({{.*}}) outs(%[[COLLAPSED]], %[[COLLAPSED_1]] : tensor<768x4xf32>, tensor<768x4xf32>)
    %reduced:2 = linalg.reduce ins(%arg0, %arg1 : tensor<768x4x3072xf32>, tensor<768x4x3072xf32>) outs(%1, %1 : tensor<768x4xf32>, tensor<768x4xf32>) dimensions = [2]  {hfusion.reduce_composed = ""}
      (%in: f32, %in_2: f32, %init: f32, %init_3: f32) {
        %3 = arith.addf %in, %init : f32
        %4 = arith.addf %in_2, %init_3 : f32
        linalg.yield %3, %4 : f32, f32
      }
    %expanded = tensor.expand_shape %reduced#0 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed = tensor.collapse_shape %expanded [[0], [1, 2, 3, 4]] : tensor<24x32x4x1x1xf32> into tensor<24x128xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced#1, %arg2 : tensor<768x4xf32>, tensor<768x4xf32>) outs(%0 : tensor<768x4xf32>) -> tensor<768x4xf32>
    %expanded_0 = tensor.expand_shape %reduced#1 [[0, 1], [2, 3, 4]] output_shape [24, 32, 4, 1, 1] : tensor<768x4xf32> into tensor<24x32x4x1x1xf32>
    %collapsed_1 = tensor.collapse_shape %expanded_0 [[0], [1, 2], [3], [4]] : tensor<24x32x4x1x1xf32> into tensor<24x128x1x1xf32>
    return %collapsed, %collapsed_1 : tensor<24x128xf32>, tensor<24x128x1x1xf32>
  }
}

// -----

module {
  // CHECK: %[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>, %[[ARG3:.*]]: tensor<?xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  func.func @caller(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = call @nested_caller(%arg1, %arg0, %arg2) : (tensor<?xf32>, index, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func @nested_caller(%arg0: tensor<?xf32>, %arg1: index, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = call @foo(%arg1, %arg0, %arg2) : (index, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func @foo(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = tensor.empty(%arg0) : tensor<?xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %arg2 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

// -----

module {
  // CHECK-HOST-RESOURCE: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16xf32>
  func.func @test_manage_host_resources(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK-HOST-RESOURCE: %[[TENSOR0:.*]] = tensor.empty()
    // CHECK-HOST-RESOURCE: %[[TENSOR1:.*]] = tensor.empty()
    // CHECK-HOST-RESOURCE: call @callee_test_manage_host_resources(%[[ARG0]], %[[ARG1]], %[[TENSOR0]], %[[TENSOR1]])
    %0:2 = call @callee_test_manage_host_resources(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>)
    return %0#0, %0#1 : tensor<16xf32>, tensor<16xf32>
  }
  func.func @callee_test_manage_host_resources(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    return %1, %2 : tensor<16xf32>, tensor<16xf32>
  }
}

// -----

module {  
  // CHECK-LABEL: @test_output_num_from_call_result
  // CHECK: %[[ARG0:.*]]: tensor<128x256xf32>, %[[ARG1:.*]]: tensor<256xf32>, %[[ARG2:.*]]: tensor<768x256xf32>, 
  // CHECK: %[[ARG3:.*]]: tensor<128xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
  // CHECK: %[[ARG4:.*]]: tensor<128x1xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>},
  // CHECK: %[[ARG5:.*]]: tensor<128x768xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}
  func.func @test_output_num_from_call_result(%arg0: tensor<128x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<768x256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %0 = tensor.empty() : tensor<128x768xf32>
    // CHECK: %[[TENSOR:.*]] = tensor.empty() : tensor<128x256xf32>
    // CHECK: call @callee_test_output_num_from_call_result(%[[ARG0]], %[[ARG1]], %[[ARG3]], %[[ARG4]], %[[TENSOR]])
    %1:3 = call @callee_test_output_num_from_call_result(%arg0, %arg1) : (tensor<128x256xf32>, tensor<256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>)
    // CHECK: linalg.matmul_transpose_b ins({{.*}}, %[[ARG2]] : tensor<128x256xf32>, tensor<768x256xf32>) outs(%[[ARG5]] : tensor<128x768xf32>) -> tensor<128x768xf32>
    %2 = linalg.matmul_transpose_b ins(%1#2, %arg2 : tensor<128x256xf32>, tensor<768x256xf32>) outs(%0 : tensor<128x768xf32>) -> tensor<128x768xf32>
    return %1#0, %1#1, %2 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>
  }
  // CHECK: @callee_test_output_num_from_call_result
  // CHECK: %[[ARG0_0:.*]]: tensor<128x256xf32>, %[[ARG1_0:.*]]: tensor<256xf32>
  // CHECK: %[[ARG2_0:.*]]: tensor<128xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
  // CHECK: %[[ARG3_0:.*]]: tensor<128x1xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>},
  // CHECK: %[[ARG4_0:.*]]: tensor<128x256xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<2>}
  func.func @callee_test_output_num_from_call_result(%arg0: tensor<128x256xf32>, %arg1: tensor<256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 2.560000e+02 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = tensor.empty() : tensor<128x1xf32>
    %2 = tensor.empty() : tensor<128x256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%1 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<256xf32>) outs(%2 : tensor<128x256xf32>) dimensions = [0]
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcasted : tensor<128x256xf32>, tensor<128x256xf32>) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    return %3, %4, %5 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>
  }
  // CHECK-HOST-RESOURCE: main
  func.func @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<768x256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK-HOST-RESOURCE: %[[TENSOR0:.*]] = tensor.empty() : tensor<128x768xf32>
    // CHECK-HOST-RESOURCE: %[[TENSOR1:.*]] = tensor.empty() : tensor<128xf32>
    // CHECK-HOST-RESOURCE: %[[TENSOR2:.*]] = tensor.empty() : tensor<128x1xf32>
    // CHECK-HOST-RESOURCE: call @test_output_num_from_call_result({{.*}}, {{.*}}, {{.*}}, %[[TENSOR1]], %[[TENSOR2]], %[[TENSOR0]])
    %0:3 = call @test_output_num_from_call_result(%arg0, %arg1, %arg2) : (tensor<128x256xf32>, tensor<256xf32>, tensor<768x256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>)
    return %0#0, %0#1, %0#2 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>
  }
}
