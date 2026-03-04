// RUN: bishengir-opt %s  -split-input-file | FileCheck %s

// CHECK-LABEL: bitcast_tensor
// CHECK: hivm.hir.bitcast
func.func @bitcast_tensor(%arg0 : tensor<2x3xf32>) -> tensor<2x3xi32> {
    %res = hivm.hir.bitcast %arg0 : tensor<2x3xf32> -> tensor<2x3xi32>
    return %res : tensor<2x3xi32> 
}

// -----

// CHECK-LABEL: bitcast_memref
// CHECK: hivm.hir.bitcast
func.func @bitcast_memref(%arg0 : memref<2x3xf32>) -> memref<2x3xi32> {
    %res = hivm.hir.bitcast %arg0 : memref<2x3xf32> -> memref<2x3xi32>
    return %res : memref<2x3xi32>
}