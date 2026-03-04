// RUN: bishengir-opt --test-assign-fusion-kind --fusion-kind="PURE_ELEMWISE" -hfusion-fuse-ops -split-input-file %s | FileCheck %s

// CHECK-NOT: PURE_ELEMWISE
// CHECK-LABEL: func.func @dynamic_shape(
func.func @dynamic_shape(%arg0: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %emp = tensor.empty(%dim) : tensor<?xf32>
  %processed_1 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
                   ins(%arg0 : tensor<?xf32>) outs(%emp : tensor<?xf32>) -> tensor<?xf32>
  
  %dim_1 = tensor.dim %processed_1, %c0 : tensor<?xf32>
  %0 = tensor.empty(%dim_1) : tensor<?xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
         ins(%processed_1 : tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  return %1, %processed_1 : tensor<?xf32>, tensor<?xf32>
}
