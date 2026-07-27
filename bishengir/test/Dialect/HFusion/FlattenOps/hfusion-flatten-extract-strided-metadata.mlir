// RUN: bishengir-opt %s -hfusion-flatten-ops="flatten-mode=tidy register-based=true" -split-input-file | FileCheck %s

func.func @flatten_extract_strided_metadata(%arg0: index) {
  %alloc = memref.alloc() : memref<64xf32>
  %subview = memref.subview %alloc[%arg0] [16] [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: ?>>
  // CHECK: %[[EXPAND:.*]] = memref.expand_shape %{{.*}} {{\[\[}}0, 1, 2, 3]] output_shape [1, 1, 1, 16]
  %expand_shape = memref.expand_shape %subview [[0, 1, 2, 3]] output_shape [1, 1, 1, 16] : memref<16xf32, strided<[1], offset: ?>> into memref<1x1x1x16xf32, strided<[16, 16, 16, 1], offset: ?>>
  // CHECK: %{{.*}}, %{{.*}}, %{{.*}}:4, %{{.*}}:4 = memref.extract_strided_metadata %[[EXPAND]] 
  %base_buffer, %offset, %size:4, %strides:4 = memref.extract_strided_metadata %expand_shape : memref<1x1x1x16xf32, strided<[16, 16, 16, 1], offset: ?>> -> memref<f32>, index, index, index, index, index, index, index, index, index
  return
}