// RUN: bishengir-opt %s -canonicalize-ext -cse -split-input-file | FileCheck %s

// -----
module {
// CHECK-LABEL: func @expand_shape_constant_fold_arg
// CHECK: %[[cast_from_src:.*]] = tensor.cast %[[arg:.*]] : tensor<?xf32> to tensor<1xf32>
// CHECK-NEXT: %[[expanded:.*]] = tensor.expand_shape %[[cast_from_src]] {{\[\[}}0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
// CHECK-NEXT: %[[cast_to_dest:.*]] = tensor.cast %[[expanded]] : tensor<1x1xf32> to tensor<?x1xf32>
// CHECK-NEXT: return %[[cast_to_dest]]
func.func @expand_shape_constant_fold_arg(%src : tensor<?xf32>) -> tensor<?x1xf32> {
    %c1 = arith.constant 1 : index
    %expanded = tensor.expand_shape %src [[0, 1]] output_shape [%c1, 1] : tensor<?xf32> into tensor<?x1xf32>
    return %expanded : tensor<?x1xf32>
}
}

// -----
module {
// CHECK-LABEL: func @expand_shape_constant_fold_with_extract_slice
// CHECK: %[[tensor:.*]] = tensor.empty() : tensor<400xf32>
// CHECK-NOT: arith.constant
// CHECK-NEXT: %[[slice:.*]] = tensor.extract_slice %[[tensor]][{{.*}}] [1] [1] : tensor<400xf32> to tensor<1xf32>
// CHECK-NEXT: %[[expanded:.*]] = tensor.expand_shape %[[slice]] {{\[\[}}0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
// CHECK-NEXT: %[[cast_to_dest:.*]] = tensor.cast %[[expanded]] : tensor<1x1xf32> to tensor<?x1xf32>
// CHECK-NEXT: return %[[cast_to_dest]]
func.func @expand_shape_constant_fold_with_extract_slice(%arg0 : index) -> tensor<?x1xf32> {
    %0 = tensor.empty() : tensor<400xf32>
    %c1 = arith.constant 1 : index
    %slice = tensor.extract_slice %0[%arg0] [%c1] [1] : tensor<400xf32> to tensor<?xf32>
    %expanded = tensor.expand_shape %slice [[0, 1]] output_shape [%c1, 1] : tensor<?xf32> into tensor<?x1xf32>
    return %expanded : tensor<?x1xf32>
}
}

// -----

// CHECK-LABEL: func.func @test_expand_dyn_within_single_reassociation_group
// CHECK-NOT: memref.expand_shape
// CHECK-NOT: memref.collapse_shape
func.func @test_expand_dyn_within_single_reassociation_group(%arg0: memref<32x4xf32>, %arg1: index, %arg2: index) -> memref<?x4xf32, strided<[4, 1], offset: ?>>{
  %subview = memref.subview %arg0[%arg1, 0] [%arg2, 4] [1, 1] : memref<32x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
  %expand_shape_1 = memref.expand_shape %subview [[0, 1], [2]] 
                    output_shape [1, %arg2, 4] : memref<?x4xf32, strided<[4, 1], offset: ?>> 
                    into memref<1x?x4xf32, strided<[?, 4, 1], offset: ?>>
  %collapse_shape = memref.collapse_shape %expand_shape_1 [[0, 1], [2]] 
                    : memref<1x?x4xf32, strided<[?, 4, 1], offset: ?>> 
                    into memref<?x4xf32, strided<[4, 1], offset: ?>>
  return %collapse_shape: memref<?x4xf32, strided<[4, 1], offset: ?>>
}
