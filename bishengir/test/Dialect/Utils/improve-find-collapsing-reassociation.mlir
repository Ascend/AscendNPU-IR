// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s

// CHECK: LLVMDialectModule
module {
  func.func @test_improve_find_collapsing_reassociation_in_reshape_ops_utils(%arg0: tensor<1x2047x2048xf32>) -> (tensor<1x2047x1xf32>, tensor<1x2047x2048xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 1.000000e-10 : f64
    %cst_2 = arith.constant 2.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x2047x2048xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1x2047x2048xf32>) -> tensor<1x2047x2048xf32>
    %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %1 : tensor<1x2047x2048xf32>, tensor<1x2047x2048xf32>) outs(%0 : tensor<1x2047x2048xf32>) -> tensor<1x2047x2048xf32>
    %3 = tensor.empty() : tensor<1x2047xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x2047xf32>) -> tensor<1x2047xf32>
    %reduced = linalg.reduce ins(%2 : tensor<1x2047x2048xf32>) outs(%4 : tensor<1x2047xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %18 = arith.addf %in, %init : f32
        linalg.yield %18 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [1, 2047, 1] : tensor<1x2047xf32> into tensor<1x2047x1xf32>
    %5 = tensor.empty() : tensor<1x2047x1xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%expanded, %6 : tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %8 = arith.truncf %cst_1 : f64 to f32
    %9 = linalg.fill ins(%8 : f32) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %10 = linalg.fill ins(%cst_3 : f32) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%9, %10 : tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%7, %11 : tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %13 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%12 : tensor<1x2047x1xf32>) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %14 = arith.truncf %cst_0 : f64 to f32
    %15 = linalg.fill ins(%14 : f32) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%13, %15 : tensor<1x2047x1xf32>, tensor<1x2047x1xf32>) outs(%5 : tensor<1x2047x1xf32>) -> tensor<1x2047x1xf32>
    %collapsed = tensor.collapse_shape %16 [[0], [1, 2]] : tensor<1x2047x1xf32> into tensor<1x2047xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<1x2047xf32>) outs(%0 : tensor<1x2047x2048xf32>) dimensions = [2] 
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<1x2047x2048xf32>, tensor<1x2047x2048xf32>) outs(%0 : tensor<1x2047x2048xf32>) -> tensor<1x2047x2048xf32>
    return %expanded, %17 : tensor<1x2047x1xf32>, tensor<1x2047x2048xf32>
  }
}