// RUN: bishengir-opt %s -hfusion-auto-schedule | FileCheck %s

// CHECK: %[[RET:.*]] = scf.index_switch
// CHECK: case 1
// CHECK:   func.call @test_dynamic_shape_single_output_1
// CHECK: case 0
// CHECK:   func.call @test_dynamic_shape_single_output_0
// CHECK: default
// CHECK:   scf.yield
// CHECK: return %[[RET]]
module {
  func.func @test_dynamic_shape_single_output(%arg0: tensor<?x1xf16>,
                                              %arg1: tensor<1x?xf16>,
                                              %arg2: tensor<?x?xf16>) -> tensor<?x?xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
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

  func.func @main(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>, %arg2: tensor<?x?xf16>) -> (tensor<?x?xf16>) 
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %ret = func.call @test_dynamic_shape_single_output(%arg0, %arg1, %arg2) : (tensor<?x1xf16>, tensor<1x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>)
    return %ret : tensor<?x?xf16>
  }
}