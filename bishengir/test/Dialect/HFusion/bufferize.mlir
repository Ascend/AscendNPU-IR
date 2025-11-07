// RUN: bishengir-opt --one-shot-bufferize="dialect-filter=hfusion,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -canonicalize-ext -cse -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --one-shot-bufferize="bufferize-function-boundaries" -canonicalize-ext -cse -split-input-file %s | FileCheck %s --check-prefix=ONE-SHOT

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[TENSOR:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG:       %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<4xf32>
// CHECK-DAG:       %[[RESULT_MEMREF:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK:           hfusion.elemwise_unary
// CHECK-SAME:      ins(%[[MEMREF]] : memref<4xf32>)
// CHECK-SAME:      outs(%[[RESULT_MEMREF]] : memref<4xf32>)
// CHECK:           %[[RESULT:.*]] = bufferization.to_tensor %[[RESULT_MEMREF]] : memref<4xf32>
// CHECK:           return %[[RESULT]] : tensor<4xf32>
func.func @basic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = hfusion.elemwise_unary ins(%arg0 : tensor<4xf32>) outs(%arg0 : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

#map0 = affine_map<(d0) -> (d0)>

// Same as above but with tensor.empty op.

// CHECK-LABEL: func @empty_tensor(
// CHECK-SAME:      %[[IN:.*]]: tensor<?xf32>, %[[SIZE:.*]]: index)
// CHECK-DAG:     %[[MEMREF:.*]] = bufferization.to_memref %[[IN]] : memref<?xf32>
// CHECK-DAG:     %[[OUT_BUF:.*]] = memref.alloc(%[[SIZE]]) {{.*}} : memref<?xf32>
// CHECK:         hfusion.elemwise_unary
// CHECK-SAME:    ins(%[[MEMREF]] : memref<?xf32>)
// CHECK-SAME:    outs(%[[OUT_BUF]] : memref<?xf32>)
func.func @empty_tensor(%in : tensor<?xf32>, %size: index) -> tensor<?xf32> {
  %init = tensor.empty(%size) : tensor<?xf32>
  %0 = hfusion.elemwise_unary ins(%in : tensor<?xf32>) outs(%init : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// ONE-SHOT-LABEL:   func @elemwise_inplace(
// ONE-SHOT-SAME:                %[[SIZE:.*]]: index)
// ONE-SHOT:           %[[MEMREF:.*]] = memref.alloc(%[[SIZE]]) {{.*}} : memref<?xf32>
// ONE-SHOT:           hfusion.elemwise_unary
// ONE-SHOT-SAME:      ins(%[[MEMREF]] : memref<?xf32>)
// ONE-SHOT-SAME:      outs(%[[MEMREF]] : memref<?xf32>)
func.func @elemwise_inplace(%size: index) -> tensor<?xf32> {
  %init = tensor.empty(%size) : tensor<?xf32>
  %0 = hfusion.elemwise_unary ins(%init : tensor<?xf32>) outs(%init : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
