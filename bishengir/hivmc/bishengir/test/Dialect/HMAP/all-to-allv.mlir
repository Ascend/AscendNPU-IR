// RUN: bishengir-opt %s -split-input-file | FileCheck %s

// -----

module {
  mesh.mesh @device_mesh(shape = 2x2)
  //CHECK-LABEL: @distributed_matmul_with_all_to_all
  func.func @distributed_matmul_with_all_to_all(
    %arg0: tensor<128x64xf32>,
    %arg1: tensor<64x128xf32>) -> tensor<128x128xf32> {

    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x128xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

    %matmul = linalg.matmul
      ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>

    %c4 = arith.constant 4 : i32
    %c32 = arith.constant 32 : i32
    %reshape_shape = tensor.from_elements %c4, %c32, %c32, %c4 : tensor<4xi32>

    %reshape = tensor.reshape %matmul(%reshape_shape)
              : (tensor<128x128xf32>, tensor<4xi32>) -> tensor<4x32x32x4xf32>
    %input_splits = tensor.empty() : tensor<1x1xi64>
    %output_splits = tensor.empty() : tensor<1x1xi64>
    %all_to_all = hmap.all_to_all_v %reshape on @device_mesh
      input_splits = %input_splits : tensor<1x1xi64>
      output_splits = %output_splits : tensor<1x1xi64> : tensor<4x32x32x4xf32> -> tensor<4x32x32x4xf32>


    %c128 = arith.constant 128 : i32
    %final_shape = tensor.from_elements %c128, %c128 : tensor<2xi32>

    %result = tensor.reshape %all_to_all(%final_shape)
              : (tensor<4x32x32x4xf32>, tensor<2xi32>) -> tensor<128x128xf32>

    return %result : tensor<128x128xf32>
  }
}
