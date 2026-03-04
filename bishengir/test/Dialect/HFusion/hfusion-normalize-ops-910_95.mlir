// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_linalg_floor_to_hfusion_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf16>)
// CHECK: %[[DST0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[RES1:.*]] = hfusion.cast {{.*}} ins(%[[arg0]] : tensor<1024xf16>) outs(%[[DST0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[RES1]]
func.func @test_linalg_floor_to_hfusion_cast(%src: tensor<1024xf16>) -> tensor<1024xf16> {
    %dst = tensor.empty() : tensor<1024xf16>
    %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<1024xf16>) outs(%dst : tensor<1024xf16>) -> tensor<1024xf16>
   return %res : tensor<1024xf16>
}

// -----
// CHECK-LABEL: func.func @linalg_reduce_with_index
// CHECK-NOT: i16
func.func @linalg_reduce_with_index(%0: tensor<3x1x1x2x600xi8>, %1: tensor<3x1x1x2x600xi32>) -> tensor<3x1x1x2xi32> {
  %2 = tensor.empty() : tensor<3x1x1x2xi8>
  %3 = tensor.empty() : tensor<3x1x1x2xi32>
  %reduced:2 = linalg.reduce ins(%0, %1 : tensor<3x1x1x2x600xi8>, tensor<3x1x1x2x600xi32>) outs(%2, %3 : tensor<3x1x1x2xi8>, tensor<3x1x1x2xi32>) dimensions = [4]  {reduce_mode = "max_with_index", tie_break_left = "true"}
      (%in: i8, %in_6: i32, %init: i8, %init_7: i32) {
        %8 = arith.cmpi sgt, %in, %init : i8
        %9 = arith.cmpi eq, %in, %init : i8
        %10 = arith.cmpi slt, %in_6, %init_7 : i32
        %11 = arith.andi %9, %10 : i1
        %12 = arith.ori %8, %11 : i1
        %13 = arith.select %12, %in, %init : i8
        %14 = arith.select %12, %in_6, %init_7 : i32
        linalg.yield %13, %14 : i8, i32
      }
  return %reduced#1 : tensor<3x1x1x2xi32>
}

// -----
// CHECK-LABEL: func.func @triton_sum_dim0_dim2
// CHECK:   %reduced = linalg.reduce ins({{.*}} : tensor<5x8x7xi16>) outs({{.*}} : tensor<8xi16>) dimensions = [0, 2] 
func.func @triton_sum_dim0_dim2(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>) {
  %c0_i8 = arith.constant 0 : i8
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [5, 8, 7], strides: [56, 7, 1] : memref<?xi8> to memref<5x8x7xi8, strided<[56, 7, 1]>>
  %alloc = memref.alloc() : memref<5x8x7xi8>
  memref.copy %reinterpret_cast, %alloc : memref<5x8x7xi8, strided<[56, 7, 1]>> to memref<5x8x7xi8>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<5x8x7xi8>
  %1 = tensor.empty() : tensor<8xi8>
  %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<8xi8>) -> tensor<8xi8>
  %reduced = linalg.reduce ins(%0 : tensor<5x8x7xi8>) outs(%2 : tensor<8xi8>) dimensions = [0, 2] 
    (%in: i8, %init: i8) {
      %3 = arith.addi %in, %init : i8
      linalg.yield %3 : i8
    }
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1]>>
  bufferization.materialize_in_destination %reduced in writable %reinterpret_cast_0 : (tensor<8xi8>, memref<8xi8, strided<[1]>>) -> ()
  return
}