// RUN: bishengir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

module {
  // CHECK-LABEL func.func @print_test
  // CHECK: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: tensor<32xf32>, %[[ARG2:.*]]: tensor<32xi8>)
  // CHECK: hfusion.print "PID: " {hex = true} %[[ARG0]] : i32
  // CHECK: hfusion.print "Val: " {hex = false} %[[ARG1]] : tensor<32xf32>
  // CHECK: hfusion.print "" {hex = false} %[[ARG2]] : tensor<32xi8>
  func.func @print_test(%arg0 : i32, %arg1 : tensor<32xf32>, %arg2 : tensor<32xi8>) {
    hfusion.print "PID: " {hex = true} %arg0 : i32
    hfusion.print "Val: " {hex = false} %arg1 : tensor<32xf32>
    hfusion.print "" {hex = false} %arg2 : tensor<32xi8>
    func.return
  }
}

// -----
// CHECK-LABEL: @test_group_matmul
func.func @test_group_matmul(%w1 : tensor<2x?x?xf32>, %tokens : tensor<?x?xf32>, %tpe : tensor<2xi64>, %out : tensor<?x?xf32>) {
  %res = hfusion.group_matmul
    ins(%w1, %tokens, %tpe : tensor<2x?x?xf32>, tensor<?x?xf32>, tensor<2xi64>)
    outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    return
}

// -----
func.func @histogram_nomask(%arg0: tensor<8xi32>) -> tensor<4xi32> {
  // CHECK-LABEL: func.func @histogram_nomask
  // CHECK: hfusion.histogram
  // CHECK: return
  %res = hfusion.histogram %arg0, 4 : tensor<8xi32> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----
func.func @histogram_mask(%arg0: tensor<8xi32>, %mask: tensor<8xi1>)
    -> tensor<4xi32> {
  // CHECK-LABEL: func.func @histogram_mask
  // CHECK: hfusion.histogram
  // CHECK: return
  %res = hfusion.histogram %arg0, 4, %mask
         : tensor<8xi32>, tensor<8xi1> -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// -----
module {
  // CHECK-LABEL func.func @test_matmulscale_acc
  // CHECK: hfusion.matmul_mx
  func.func @test_matmulscale_acc(%arg0: memref<4x8xf8E4M3FN>, %arg1: memref<8x16xf8E4M3FN>, %arg2: memref<1xui8>, %arg3: memref<1xui8>, %arg4: memref<4x16xf8E4M3FN>, %arg5: memref<4x16xf8E5M2>) {
    %0 = bufferization.to_tensor %arg0 : memref<4x8xf8E4M3FN>
    %1 = bufferization.to_tensor %arg1 : memref<8x16xf8E4M3FN>
    %2 = bufferization.to_tensor %arg2 : memref<1xui8>
    %3 = bufferization.to_tensor %arg3 : memref<1xui8>
    %4 = bufferization.to_tensor %arg4 : memref<4x16xf8E4M3FN>
    %5 = hfusion.matmul_mx ins(%0, %1, %2, %3 : tensor<4x8xf8E4M3FN>, tensor<8x16xf8E4M3FN>, tensor<1xui8>, tensor<1xui8>) outs(%4 : tensor<4x16xf8E4M3FN>) -> tensor<4x16xf8E4M3FN>
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @test_matmulscale_noacc
  // CHECK: hfusion.matmul_mx
  func.func @test_matmulscale_noacc(%arg0: memref<4x8xf8E4M3FN>, %arg1: memref<8x16xf8E4M3FN>, %arg2: memref<1xui8>, %arg3: memref<1xui8>, %arg4: memref<4x16xf8E4M3FN>, %arg5: memref<4x16xf8E5M2>) {
    %0 = bufferization.to_tensor %arg0 : memref<4x8xf8E4M3FN>
    %1 = bufferization.to_tensor %arg1 : memref<8x16xf8E4M3FN>
    %2 = bufferization.to_tensor %arg2 : memref<1xui8>
    %3 = bufferization.to_tensor %arg3 : memref<1xui8>
    %4 = bufferization.to_tensor %arg4 : memref<4x16xf8E4M3FN>
    %empty = tensor.empty() : tensor<4x16xf8E4M3FN>
    %5 = hfusion.matmul_mx ins(%0, %1, %2, %3 : tensor<4x8xf8E4M3FN>, tensor<8x16xf8E4M3FN>, tensor<1xui8>, tensor<1xui8>) outs(%empty : tensor<4x16xf8E4M3FN>) -> tensor<4x16xf8E4M3FN>
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @test_matmulscale_e5m2
  // CHECK: hfusion.matmul_mx
  func.func @test_matmulscale_e5m2(%arg0: memref<4x8xf8E5M2>, %arg1: memref<8x16xf8E5M2>, %arg2: memref<1xui8>, %arg3: memref<1xui8>, %arg4: memref<4x16xf8E5M2>, %arg5: memref<4x16xf8E5M2>) {
    %0 = bufferization.to_tensor %arg0 : memref<4x8xf8E5M2>
    %1 = bufferization.to_tensor %arg1 : memref<8x16xf8E5M2>
    %2 = bufferization.to_tensor %arg2 : memref<1xui8>
    %3 = bufferization.to_tensor %arg3 : memref<1xui8>
    %4 = bufferization.to_tensor %arg4 : memref<4x16xf8E5M2>
    %empty = tensor.empty() : tensor<4x16xf8E5M2>
    %5 = hfusion.matmul_mx ins(%0, %1, %2, %3 : tensor<4x8xf8E5M2>, tensor<8x16xf8E5M2>, tensor<1xui8>, tensor<1xui8>) outs(%empty : tensor<4x16xf8E5M2>) -> tensor<4x16xf8E5M2>
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @test_matmulscale_e5m2_memref
  // CHECK: hfusion.matmul_mx
  func.func @test_matmulscale_e5m2_memref(%arg0: memref<4x8xf8E5M2>, %arg1: memref<8x16xf8E5M2>, %arg2: memref<1xui8>, %arg3: memref<1xui8>, %arg4: memref<4x16xf8E5M2>, %arg5: memref<4x16xf8E5M2>, %arg6: memref<4x16xf8E5M2>) {
    %5 = hfusion.matmul_mx ins(%arg0,  %arg1, %arg2, %arg3 : memref<4x8xf8E5M2>, memref<8x16xf8E5M2>,  memref<1xui8>, memref<1xui8>) outs(%arg6: memref<4x16xf8E5M2>) -> memref<4x16xf8E5M2>
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @test_matmulscale_error1
  func.func @test_matmulscale_error1(%arg0: memref<4x8xf8E5M2>, %arg1: memref<7x16xf8E5M2>, %arg2: memref<1xui8>, %arg3: memref<1xui8>, %arg4: memref<4x16xf8E5M2>, %arg5: memref<4x16xf8E5M2>, %arg6: memref<4x16xf8E5M2>) {
    // expected-error@+1 {{'hfusion.matmul_mx' op requires inner dimension of matmul matrix to match}}
    %5 = hfusion.matmul_mx ins(%arg0,  %arg1, %arg2, %arg3 : memref<4x8xf8E5M2>, memref<7x16xf8E5M2>,  memref<1xui8>, memref<1xui8>) outs(%arg6: memref<4x16xf8E5M2>) -> memref<4x16xf8E5M2>
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @test_matmulscale_error2
  func.func @test_matmulscale_error2(%arg0: memref<4x8xf8E5M2>, %arg1: memref<8x16xf8E5M2>, %arg2: memref<1xui8>, %arg3: memref<1xui8>, %arg4: memref<4x16xf8E5M2>, %arg5: memref<4x16xf8E5M2>, %arg6: memref<4x17xf8E5M2>) {
    // expected-error@+1 {{'hfusion.matmul_mx' op requires output shape to match with input shapes}}
    %5 = hfusion.matmul_mx ins(%arg0,  %arg1, %arg2, %arg3 : memref<4x8xf8E5M2>, memref<8x16xf8E5M2>,  memref<1xui8>, memref<1xui8>) outs(%arg6: memref<4x17xf8E5M2>) -> memref<4x17xf8E5M2>
    return
  }
}

