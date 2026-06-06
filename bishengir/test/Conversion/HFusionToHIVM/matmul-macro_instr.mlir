// RUN: bishengir-opt -convert-hfusion-to-hivm="mm-map-mode=macro_instr" -canonicalize %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_matmul_with_scope_matmul_limited_in_cube
// CHECK-SAME:    (%arg0: memref<16x16xf16>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scope.scope : () -> ()
// CHECK: %[[OUT:.+]] = bufferization.to_tensor %arg0 restrict writable : memref<16x16xf16>
// CHECK: %[[ALLOC_A:.+]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[A:.+]] = bufferization.to_tensor %[[ALLOC_A]] restrict writable : memref<16x16xf16>
// CHECK: %[[ALLOC_B:.+]] = memref.alloc() : memref<16x16xf16>
// CHECK: %[[B:.+]] = bufferization.to_tensor %[[ALLOC_B]] restrict writable : memref<16x16xf16>
// CHECK: %[[RET:.+]] = hivm.hir.mmadL1 ins(%[[A]], %[[B]], %[[FALSE]], %[[C0]], %[[C0]], %[[C0]] : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%[[OUT]] : tensor<16x16xf16>) -> tensor<16x16xf16>
// CHECK: %[[CAST:.+]] = memref.cast %arg0 : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
// CHECK: bufferization.materialize_in_destination %[[RET]] in writable %[[CAST]] : (tensor<16x16xf16>, memref<16x16xf16, strided<[16, 1]>>) -> ()
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
// CHECK: return
func.func @test_matmul_with_scope_matmul_limited_in_cube(%arg1: memref<16x16xf16>){
  scope.scope : () -> () {
    %cst = arith.constant 0.000000e+00 : f32
    %mc = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf16>
    %ma = memref.alloc() : memref<16x16xf16>
    %ma_tensor = bufferization.to_tensor %ma restrict writable : memref<16x16xf16>
    %mb = memref.alloc() : memref<16x16xf16>
    %mb_tensor = bufferization.to_tensor %mb restrict writable : memref<16x16xf16>
    %ret = linalg.matmul ins(%ma_tensor, %mb_tensor : tensor<16x16xf16>, tensor<16x16xf16>) outs(%mc : tensor<16x16xf16>) -> tensor<16x16xf16>
    %subview_104 = memref.subview %arg1[0, 0][16, 16][1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1], offset: 0>>
    bufferization.materialize_in_destination %ret in writable %subview_104 : (tensor<16x16xf16>, memref<16x16xf16, strided<[16, 1], offset: 0>>) -> ()
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>, hivm.matmul_limited_in_cube}
  return
}