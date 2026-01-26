// RUN: bishengir-opt --one-shot-bufferize="dialect-filter=annotation,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -canonicalize-ext -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @mark
// CHECK-SAME:     %[[t1:.*]]: tensor<?xf32>, %[[t2:.*]]: tensor<?xf32>
func.func @mark(
    %t1 : tensor<?xf32>,
    %t2 : tensor<?xf32>,
    %c : i1)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index
  %w = tensor.insert %cst into %t1[%idx] : tensor<?xf32>
  // CHECK: %[[select:.*]] = arith.select %{{.*}}, %[[t1]], %[[t2]]
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>
  // CHECK: %[[select:.*]] = bufferization.to_memref {{.*}} : memref<?xf32>
  // CHECK: annotation.mark %[[select]] {attr = 2 : i32}
  annotation.mark %s {attr = 2 : i32} : tensor<?xf32>
  return %s, %w : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dyn_attr
// CHECK-SAME:     %[[t1:.*]]: tensor<?xf32>, %[[t2:.*]]: tensor<?xf32>
func.func @dyn_attr(
    %t1 : tensor<?xf32>,
    %t2 : tensor<?xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: bufferization.to_memref
  // CHECK: bufferization.to_memref
  // CHECK: annotation.mark
  annotation.mark %t1 keys = ["key"] values = [%t2: tensor<?xf32>] : tensor<?xf32>
  return %t1, %t2 : tensor<?xf32>, tensor<?xf32>
}