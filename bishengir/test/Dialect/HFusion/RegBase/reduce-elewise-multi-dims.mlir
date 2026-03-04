// RUN: bishengir-compile -target=Ascend310B4 --enable-hfusion-compile=true  --enable-triton-kernel-compile=true -enable-lir-compile=false  %s | FileCheck %s

module {
// CHECK-LABEL: triton_mul_reduce_4d_5d
// CHECK: call <64 x float> @llvm.hivm.vadd.v64f32.x
  func.func @triton_mul_reduce_4d_5d_65(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %0 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [2, 3, 4, 5, 65], strides: [3900, 1300, 325, 65, 1] : memref<?xf32> to memref<2x3x4x5x65xf32, strided<[3900, 1300, 325, 65, 1], offset: ?>>
    %alloc = memref.alloc() : memref<2x3x4x5x65xf32>
    memref.copy %reinterpret_cast, %alloc : memref<2x3x4x5x65xf32, strided<[3900, 1300, 325, 65, 1], offset: ?>> to memref<2x3x4x5x65xf32>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<2x3x4x5x65xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [2, 3, 4, 5, 65], strides: [3900, 1300, 325, 65, 1] : memref<?xf32> to memref<2x3x4x5x65xf32, strided<[3900, 1300, 325, 65, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<2x3x4x5x65xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<2x3x4x5x65xf32, strided<[3900, 1300, 325, 65, 1], offset: ?>> to memref<2x3x4x5x65xf32>
    %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x3x4x5x65xf32>

    %filled = tensor.empty() : tensor<2x4x5x65xf32>

    %xx = tensor.empty() : tensor<2x3x4x5x65xf32>
    %3 = linalg.mul ins(%1, %2 : tensor<2x3x4x5x65xf32>, tensor<2x3x4x5x65xf32>) outs(%xx: tensor<2x3x4x5x65xf32>) -> tensor<2x3x4x5x65xf32>
    %sum0 = linalg.reduce {arith.addf} ins(%3 : tensor<2x3x4x5x65xf32>) outs(%filled : tensor<2x4x5x65xf32>) dimensions = [1]

    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%0], sizes: [2, 4, 5, 65], strides: [1300, 325, 65, 1] : memref<?xf32> to memref<2x4x5x65xf32, strided<[1300, 325, 65, 1], offset: ?>>
    bufferization.materialize_in_destination %sum0 in writable %reinterpret_cast_2 : (tensor<2x4x5x65xf32>, memref<2x4x5x65xf32, strided<[1300, 325, 65, 1], offset: ?>>) -> ()
    return
  }
// -----



// CHECK-LABEL: define dso_local void @triton_mul_4d_5d
  func.func @triton_mul_4d_5d(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %0 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [2, 3, 4, 5, 6], strides: [360, 120, 30, 6, 1] : memref<?xf32> to memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>>
    %alloc = memref.alloc() : memref<2x3x4x5x6xf32>
    memref.copy %reinterpret_cast, %alloc : memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>> to memref<2x3x4x5x6xf32>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<2x3x4x5x6xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [2, 3, 4, 5, 6], strides: [360, 120, 30, 6, 1] : memref<?xf32> to memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<2x3x4x5x6xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>> to memref<2x3x4x5x6xf32>
    %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x3x4x5x6xf32>
    // CHECK: call <64 x float> @llvm.hivm.vmul.v64f32.x
    %3 = arith.mulf %1, %2 : tensor<2x3x4x5x6xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%0], sizes: [2, 3, 4, 5, 6], strides: [360, 120, 30, 6, 1] : memref<?xf32> to memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>>
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<2x3x4x5x6xf32>, memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>>) -> ()
    return
  }
// -----


// CHECK: llvm.hivm.vadd.v64f32.x
// CHECK-LABEL: define dso_local void @triton_mul_reduce_4d_5d
  func.func @triton_mul_reduce_4d_5d(%arg0: memref<?xi8>, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %0 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [2, 3, 4, 5, 6], strides: [360, 120, 30, 6, 1] : memref<?xf32> to memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>>
    %alloc = memref.alloc() : memref<2x3x4x5x6xf32>
    memref.copy %reinterpret_cast, %alloc : memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>> to memref<2x3x4x5x6xf32>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<2x3x4x5x6xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [2, 3, 4, 5, 6], strides: [360, 120, 30, 6, 1] : memref<?xf32> to memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<2x3x4x5x6xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<2x3x4x5x6xf32, strided<[360, 120, 30, 6, 1], offset: ?>> to memref<2x3x4x5x6xf32>
    %2 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x3x4x5x6xf32>

    %filled = tensor.empty() : tensor<2x4x5x6xf32>

    %xx = tensor.empty() : tensor<2x3x4x5x6xf32>
    %3 = linalg.mul ins(%1, %2 : tensor<2x3x4x5x6xf32>, tensor<2x3x4x5x6xf32>) outs(%xx: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>

    %sum0 = linalg.reduce {arith.addf} ins(%3 : tensor<2x3x4x5x6xf32>) outs(%filled : tensor<2x4x5x6xf32>) dimensions = [1]

    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%0], sizes: [2, 4, 5, 6], strides: [120, 30, 6, 1] : memref<?xf32> to memref<2x4x5x6xf32, strided<[120, 30, 6, 1], offset: ?>>
    bufferization.materialize_in_destination %sum0 in writable %reinterpret_cast_2 : (tensor<2x4x5x6xf32>, memref<2x4x5x6xf32, strided<[120, 30, 6, 1], offset: ?>>) -> ()
    return
  }
// -----

// CHECK-LABEL: define dso_local void @triton_abs
  func.func @triton_abs(%arg0: memref<?xi8>, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 1, 23], strides: [23, 23, 1] : memref<?xi8> to memref<1x1x23xi8, strided<[23, 23, 1]>>
    %alloc = memref.alloc() : memref<1x1x23xi8>
    memref.copy %reinterpret_cast, %alloc : memref<1x1x23xi8, strided<[23, 23, 1]>> to memref<1x1x23xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1x1x23xi8>
    // CHECk: call <256 x i8> @llvm.hivm.vabs.v256s8.x
    %1 = math.absi %0 : tensor<1x1x23xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1, 23], strides: [23, 23, 1] : memref<?xi8> to memref<1x1x23xi8, strided<[23, 23, 1]>>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast_0 : (tensor<1x1x23xi8>, memref<1x1x23xi8, strided<[23, 23, 1]>>) -> ()
    return
  }
// -----

// CHECK-LABEL: define dso_local void @triton_add
  func.func @triton_add(%arg0: memref<?xi8>, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [13, 5, 31], strides: [155, 31, 1] : memref<?xi8> to memref<13x5x31xi8, strided<[155, 31, 1]>>
    %alloc = memref.alloc() : memref<13x5x31xi8>
    memref.copy %reinterpret_cast, %alloc : memref<13x5x31xi8, strided<[155, 31, 1]>> to memref<13x5x31xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<13x5x31xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [13, 5, 31], strides: [155, 31, 1] : memref<?xi8> to memref<13x5x31xi8, strided<[155, 31, 1]>>
    %alloc_1 = memref.alloc() : memref<13x5x31xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<13x5x31xi8, strided<[155, 31, 1]>> to memref<13x5x31xi8>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<13x5x31xi8>
    // CHECK: llvm.hivm.vdups.v256s8.z
    %2 = arith.addi %0, %1 : tensor<13x5x31xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13, 5, 31], strides: [155, 31, 1] : memref<?xi8> to memref<13x5x31xi8, strided<[155, 31, 1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_2 : (tensor<13x5x31xi8>, memref<13x5x31xi8, strided<[155, 31, 1]>>) -> ()
    return
  }
// -----

// CHECK-LABEL: define dso_local void @triton_and
  func.func @triton_and(%arg0: memref<?xi8>, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c0_i8 = arith.constant 0 : i8
    %0 = tensor.empty() : tensor<1x1x23xi8>
    %1 = linalg.fill ins(%c0_i8 : i8) outs(%0 : tensor<1x1x23xi8>) -> tensor<1x1x23xi8>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 1, 23], strides: [23, 23, 1] : memref<?xi8> to memref<1x1x23xi8, strided<[23, 23, 1]>>
    %alloc = memref.alloc() : memref<1x1x23xi8>
    memref.copy %reinterpret_cast, %alloc : memref<1x1x23xi8, strided<[23, 23, 1]>> to memref<1x1x23xi8>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<1x1x23xi8>
    %3 = arith.cmpi ne, %2, %1 : tensor<1x1x23xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1, 23], strides: [23, 23, 1] : memref<?xi8> to memref<1x1x23xi8, strided<[23, 23, 1]>>
    %alloc_1 = memref.alloc() : memref<1x1x23xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1x1x23xi8, strided<[23, 23, 1]>> to memref<1x1x23xi8>
    %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x1x23xi8>
    %5 = arith.cmpi ne, %4, %1 : tensor<1x1x23xi8>
    // CHECK: llvm.hivm.pand.z
    %6 = arith.andi %3, %5 : tensor<1x1x23xi1>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1, 23], strides: [23, 23, 1] : memref<?xi8> to memref<1x1x23xi8, strided<[23, 23, 1]>>
    %7 = arith.extui %6 : tensor<1x1x23xi1> to tensor<1x1x23xi8>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_2 : (tensor<1x1x23xi8>, memref<1x1x23xi8, strided<[23, 23, 1]>>) -> ()
    return
  }
}
