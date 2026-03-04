// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_concat
func.func @test_concat() -> tensor<136x4096xf32> {
  //CHECK-DAG: %[[empty0:.*]] = tensor.empty() : tensor<136x2048xf32>
  //CHECK-DAG: %[[empty1:.*]] = tensor.empty() : tensor<136x2048xf32>
  //CHECK-DAG: %[[empty2:.*]] = tensor.empty() : tensor<136x4096xf32>
  %0 = tensor.empty() : tensor<136x2048xf32>
  %1 = tensor.empty() : tensor<136x2048xf32>
  %init = tensor.empty() : tensor<136x4096xf32>
  //CHECK: %[[res:.*]] = tensor.concat dim(1) %0, %1 : (tensor<136x2048xf32>, tensor<136x2048xf32>) -> tensor<136x4096xf32>
  //CHECK: return %[[res]] : tensor<136x4096xf32>
  hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>) outs(%init : tensor<136x4096xf32>) -> tensor<136x4096xf32>
  return %init : tensor<136x4096xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_dyn
func.func @test_concat_dyn(%arg0: tensor<?x2048xf32>, %arg1: tensor<?x2048xf32>, %out: tensor<?x4096xf32>) -> tensor<?x4096xf32> {
  //CHECK: %[[res:.*]] = tensor.concat dim(1) %arg0, %arg1 : (tensor<?x2048xf32>, tensor<?x2048xf32>) -> tensor<?x4096xf32>
  //CHECK: return %[[res]] : tensor<?x4096xf32>
  hivm.hir.vconcat dim(1) ins(%arg0, %arg1 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%out : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  return %out : tensor<?x4096xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_on_dynamic_dim
func.func @test_concat_on_dynamic_dim(%arg0: tensor<?x2048xf32>, %arg1: tensor<?x2048xf32>, %out: tensor<?x2048xf32>) -> tensor<?x2048xf32> {
  //CHECK: %[[res:.*]] = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x2048xf32>, tensor<?x2048xf32>) -> tensor<?x2048xf32>
  //CHECK: return %[[res]] : tensor<?x2048xf32>
  hivm.hir.vconcat dim(0) ins(%arg0, %arg1 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%out : tensor<?x2048xf32>) -> tensor<?x2048xf32>
  return %out : tensor<?x2048xf32>
}

// -----
// CHECK-LABEL: func.func @test_pad_static
func.func @test_pad_static(%arg0 : tensor<2047xf32>) -> tensor<4093xf32> {
    //CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
    //CHECK-DAG: %[[padded:.*]] = tensor.pad %[[arg0:.*]] low[2046] high[0] {
    //CHECK-DAG:   ^bb0(%[[idx:.*]]: index):
    //CHECK-DAG:     tensor.yield %[[cst]] : f32
    //CHECK-DAG:   } : tensor<2047xf32> to tensor<4093xf32>
    //CHECK: return %[[padded]] : tensor<4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4093xf32>
    %padded = hivm.hir.vpad ins(%arg0 : tensor<2047xf32>) outs(%init : tensor<4093xf32>) low[2046] high[0] pad_value %cst : f32 -> tensor<4093xf32>
    return %init : tensor<4093xf32>
}

// -----
// CHECK-LABEL: func.func @test_hivm_pad_to_tensor_dynamic
func.func @test_hivm_pad_to_tensor_dynamic(%arg0: tensor<2x?x4x5xf32>, %out: tensor<5x?x13x26xf32>) -> tensor<5x?x13x26xf32> {
    //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[cst_1:.*]] = arith.constant 1 : index
    //CHECK-DAG: %[[cst_2:.*]] = arith.constant 2 : index
    //CHECK-DAG: %[[padded:.*]] = tensor.pad %arg0 low[%[[cst_1]], %[[cst_1]], 7, %[[cst_1]]] high[%[[cst_2]], %[[cst_2]], %[[cst_2]], 20] {
    //CHECK-DAG:  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
    //CHECK-DAG:  tensor.yield %cst : f32
    //CHECK-DAG: } : tensor<2x?x4x5xf32> to tensor<5x?x13x26xf32>
    //CHECK: return %[[padded:.*]] : tensor<5x?x13x26xf32> 
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = hivm.hir.vpad ins(%arg0 : tensor<2x?x4x5xf32>) outs(%out : tensor<5x?x13x26xf32>) low[%c1, %c1, 7, %c1] high[%c2, %c2, %c2, 20] pad_value %cst : f32 -> tensor<5x?x13x26xf32>
    return %0 : tensor<5x?x13x26xf32>
}


