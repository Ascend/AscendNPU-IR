// RUN: bishengir-opt %s --bubble-pad-up --cse --canonicalize-ext --split-input-file | FileCheck %s

// CHECK-LABEL: test_pad
// CHECK: pad
// CHECK: pad
// CHECK: elemwise_unary
// CHECK: return
module {
  func.func @test_pad(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %empty1 = tensor.empty() : tensor<2x3xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<2x3xf32>) outs(%empty1 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %pad   = tensor.pad %unary low[0, 1] high[0, 0] { ^bb0(%arg9: index, %arg10: index):
      tensor.yield %cst : f32
    } : tensor<2x3xf32> to tensor<2x4xf32>
    return %pad : tensor<2x4xf32>
  }
}

// -----

// CHECK-LABEL: test_pad_unaligned
// CHECK: elemwise_unary
// CHECK: pad
// CHECK: return
module {
  func.func @test_pad_unaligned(%arg0: tensor<2x3xf32>) -> tensor<2x35xf32> {
    %empty1 = tensor.empty() : tensor<2x3xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<2x3xf32>) outs(%empty1 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %pad   = tensor.pad %unary low[0, 32] high[0, 0] { ^bb0(%arg9: index, %arg10: index):
      tensor.yield %cst : f32
    } : tensor<2x3xf32> to tensor<2x35xf32>
    return %pad : tensor<2x35xf32>
  }
}


// -----

// CHECK-LABEL: test_pad_unaligned_non_back
// CHECK: elemwise_unary
// CHECK: pad
// CHECK: return
module {
  func.func @test_pad_unaligned_non_back(%arg0: tensor<2x3xf32>) -> tensor<7x35xf32> {
    %empty1 = tensor.empty() : tensor<2x3xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %unary = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<2x3xf32>) outs(%empty1 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %pad   = tensor.pad %unary low[5, 32] high[0, 0] { ^bb0(%arg9: index, %arg10: index):
      tensor.yield %cst : f32
    } : tensor<2x3xf32> to tensor<7x35xf32>
    return %pad : tensor<7x35xf32>
  }
}