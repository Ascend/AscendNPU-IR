// RUN: bishengir-opt -convert-tensor-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_concat
func.func @test_concat() -> tensor<136x4096xf32> {
  //CHECK-DAG: %[[empty0:.*]] = tensor.empty() : tensor<136x2048xf32>
  //CHECK-DAG: %[[empty1:.*]] = tensor.empty() : tensor<136x2048xf32>
  //CHECK-DAG: %[[empty2:.*]] = tensor.empty() : tensor<136x4096xf32>
  %0 = tensor.empty() : tensor<136x2048xf32>
  %1 = tensor.empty() : tensor<136x2048xf32>
  //CHECK: %[[res:.*]] = hivm.hir.vconcat dim(1) ins(%[[empty0]], %[[empty1]] : tensor<136x2048xf32>, tensor<136x2048xf32>) outs(%[[empty2]] : tensor<136x4096xf32>) -> tensor<136x4096xf32>
  //CHECK: return %[[res]] : tensor<136x4096xf32>
  %2 = tensor.concat dim(1) %0, %1 : (tensor<136x2048xf32>, tensor<136x2048xf32>) -> tensor<136x4096xf32>
  return %2 : tensor<136x4096xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_dyn
func.func @test_concat_dyn(%arg0: tensor<?x2048xf32>, %arg1: tensor<?x2048xf32>) -> tensor<?x4096xf32> {
  //CHECK-DAG: %[[dim:.*]] = tensor.dim {{.*}}
  //CHECK-DAG: %[[empty:.*]] = tensor.empty(%[[dim]]) : tensor<?x4096xf32>
  //CHECK: %[[res:.*]] = hivm.hir.vconcat dim(1) ins(%arg0, %arg1 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%[[empty]] : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  //CHECK: return %[[res]] : tensor<?x4096xf32>
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<?x2048xf32>, tensor<?x2048xf32>) -> tensor<?x4096xf32>
  return %0 : tensor<?x4096xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_on_dynamic_dim
func.func @test_concat_on_dynamic_dim(%arg0: tensor<?x2048xf32>, %arg1: tensor<?x2048xf32>) -> tensor<?x2048xf32> {
  //CHECK-DAG: %[[dim0:.*]] = tensor.dim %arg0, {{.*}}
  //CHECK-DAG: %[[dim1:.*]] = tensor.dim %arg1, {{.*}}
  //CHECK: %[[dimsum:.*]] = affine.apply #map()[%[[dim0]], %[[dim1]]]
  //CHECK: %[[empty:.*]] = tensor.empty(%[[dimsum]]) : tensor<?x2048xf32>
  //CHECK: %[[res:.*]] = hivm.hir.vconcat dim(0) ins(%arg0, %arg1 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%[[empty]] : tensor<?x2048xf32>) -> tensor<?x2048xf32>
  //CHECK: return %[[res]] : tensor<?x2048xf32>
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x2048xf32>, tensor<?x2048xf32>) -> tensor<?x2048xf32>
  return %0 : tensor<?x2048xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_on_dynamic_dim_with_static_output_0
func.func @test_concat_on_dynamic_dim_with_static_output_0(%arg0: tensor<?x2048xf32>, %arg1: tensor<?x2048xf32>) -> tensor<1x2048xf32> {
  // CHECK: %[[empty:.*]] = tensor.empty() : tensor<1x2048xf32>
  // CHECK: %[[res:.*]] = hivm.hir.vconcat dim(0) ins(%arg0, %arg1 : tensor<?x2048xf32>, tensor<?x2048xf32>) outs(%[[empty]] : tensor<1x2048xf32>) -> tensor<1x2048xf32>
  // CHECK: return %[[res]] : tensor<1x2048xf32>
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x2048xf32>, tensor<?x2048xf32>) -> tensor<1x2048xf32>
  return %0 : tensor<1x2048xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_on_dynamic_dim_with_static_output_1
func.func @test_concat_on_dynamic_dim_with_static_output_1(%arg0: tensor<?x2048x?xf32>, %arg1: tensor<?x2048x?xf32>) -> tensor<1x2048x?xf32> {
  // CHECK: %[[c2:.*]] = arith.constant 2 : index
  // CHECK: %[[dim:.*]] = tensor.dim %arg0, %[[c2]] : tensor<?x2048x?xf32>
  // CHECK: %[[empty:.*]] = tensor.empty(%[[dim]]) : tensor<1x2048x?xf32>
  // CHECK: %[[res:.*]] = hivm.hir.vconcat dim(0) ins(%arg0, %arg1 : tensor<?x2048x?xf32>, tensor<?x2048x?xf32>) outs(%[[empty]] : tensor<1x2048x?xf32>) -> tensor<1x2048x?xf32>
  // CHECK: return %[[res]] : tensor<1x2048x?xf32>
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x2048x?xf32>, tensor<?x2048x?xf32>) -> tensor<1x2048x?xf32>
  return %0 : tensor<1x2048x?xf32>
}

// -----
// CHECK-LABEL: func.func @test_concat_on_dynamic_dim_with_static_output_2
func.func @test_concat_on_dynamic_dim_with_static_output_2(%arg0: tensor<?x2048x?xf32>, %arg1: tensor<?x2048x?xf32>) -> tensor<1x2048x1xf32> {
  // CHECK: %[[empty:.*]] = tensor.empty() : tensor<1x2048x1xf32>
  // CHECK: %[[res:.*]] = hivm.hir.vconcat dim(0) ins(%arg0, %arg1 : tensor<?x2048x?xf32>, tensor<?x2048x?xf32>) outs(%[[empty]] : tensor<1x2048x1xf32>) -> tensor<1x2048x1xf32>
  // CHECK: return %[[res]] : tensor<1x2048x1xf32>
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x2048x?xf32>, tensor<?x2048x?xf32>) -> tensor<1x2048x1xf32>
  return %0 : tensor<1x2048x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_pad_to_hivm_static
func.func @test_tensor_pad_to_hivm_static(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
    //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
    //CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
    //CHECK-DAG: %[[pad:.*]] = hivm.hir.vpad ins(%[[source]] : tensor<2047xf32>) outs(%[[empty]] : tensor<4093xf32>) low[2046] high[0] pad_value %[[cst_0]] : f32 -> tensor<4093xf32>
    //CHECK: return %[[pad]] : tensor<4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %padded = tensor.pad %collapsed low[2046] high[0] {
    ^bb0(%arg1: index):
      tensor.yield %cst : f32
    } : tensor<2047xf32> to tensor<4093xf32>
    return %padded : tensor<4093xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_pad_to_hivm_dynamic
func.func @test_tensor_pad_to_hivm_dynamic(%arg0 : tensor<2x?x4x5xf32>) -> tensor<5x?x13x26xf32> {
    //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[cst_1:.*]] = arith.constant 1 : index
    //CHECK-DAG: %[[cst_2:.*]] = arith.constant 2 : index
    //CHECK-DAG: %[[empty:.*]] = tensor.empty(%{{.*}}) : tensor<5x?x13x26xf32>
    //CHECK-DAG: %[[pad:.*]] = hivm.hir.vpad ins(%arg0 : tensor<2x?x4x5xf32>) outs(%[[empty]] : tensor<5x?x13x26xf32>) low[%[[cst_1]], %[[cst_1]], 7, %[[cst_1]]] high[%[[cst_2]], %[[cst_2]], %[[cst_2]], 20] pad_value %[[cst_0]] : f32 -> tensor<5x?x13x26xf32>
    //CHECK: return %[[pad]] : tensor<5x?x13x26xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %left = arith.constant 1 : index
    %right = arith.constant 2 : index
    %padded = tensor.pad %arg0 low[%left, %left, 7, %left] high[%right, %right, %right, 20] {
    ^bb0(%arg1: index,%arg2: index,%arg3: index,%arg4: index):
      tensor.yield %cst : f32
    } : tensor<2x?x4x5xf32> to tensor<5x?x13x26xf32>
    return %padded : tensor<5x?x13x26xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_pad_to_hivm_with_cast
func.func @test_tensor_pad_to_hivm_with_cast(%arg0 : tensor<?x?x512xf32>) -> tensor<1x?x512xf32> {
    //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[cst_2:.*]] = arith.constant 2 : index
    //CHECK-DAG: %[[empty:.*]] = tensor.empty(%{{.*}}) : tensor<?x?x512xf32>
    //CHECK-DAG: %[[pad:.*]] = hivm.hir.vpad ins(%arg0 : tensor<?x?x512xf32>) outs(%[[empty]] : tensor<?x?x512xf32>) low[0, 0, 0] high[0, %[[cst_2]], 0] pad_value %[[cst_0]] : f32 -> tensor<?x?x512xf32>
    //CHECK: %[[cast:.*]] = tensor.cast %[[pad]] : tensor<?x?x512xf32> to tensor<1x?x512xf32>
    //CHECK: return %[[cast]] : tensor<1x?x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %right = arith.constant 2 : index
    %padded = tensor.pad %arg0 low[0, 0, 0] high[0, %right, 0] {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):
      tensor.yield %cst : f32
    } {__intermediate_producer__} : tensor<?x?x512xf32> to tensor<1x?x512xf32>
    return %padded : tensor<1x?x512xf32>
}

