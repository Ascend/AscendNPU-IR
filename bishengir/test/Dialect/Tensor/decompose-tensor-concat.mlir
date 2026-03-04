// RUN: bishengir-opt --decompose-tensor-concat %s | FileCheck %s

// CHECK-LABEL: @func
// CHECK: tensor.empty
// CHECK-2: tensor.insert_slice

func.func @func(%arg0: tensor<2560xf32>, %arg1: tensor<2560xf32>) -> tensor<5120xf32> {
    %1 = tensor.concat dim(0) %arg0, %arg1: (tensor<2560xf32>, tensor<2560xf32>) -> tensor<5120xf32>
    func.return %1: tensor<5120xf32>
}
