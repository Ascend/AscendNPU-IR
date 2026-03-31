// RUN: bishengir-compile -target=Ascend310B4 -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=1 -enable-vf-fusion=true -enable-triton-kernel-compile=true %s | FileCheck %s

module {
  // CHECK-LABEL: @triton_kernel_i8_fused_0_outlined_vf_0
  // CHECK: @llvm.hivm.pge.b8
  // CHECK: @llvm.hivm.vldsx1.v256s8
  // CHECK: @llvm.hivm.vabs.x
  // CHECK: @llvm.hivm.vstsx1.v256s8
  func.func @triton_kernel_i8(%arg0: memref<?xi8>, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi slt, %arg6, %c1_i32 : i32
    %1 = arith.index_cast %arg6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1xi8>
    %2 = arith.index_castui %0 : i1 to index
    %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%2] [1] : memref<1xi8> to memref<?xi8, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1]>>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
    %4 = math.absi %3 : tensor<1xi8>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %4[0] [%2] [1] : tensor<1xi8> to tensor<?xi8>
    %subview_2 = memref.subview %reinterpret_cast_1[0] [%2] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xi8>, memref<?xi8, strided<[1], offset: ?>>) -> ()
    return
  }

  // CHECK-LABEL: triton_kernel_i8_fused_0_outlined_vf_0
  // CHECK: @llvm.hivm.pge.b16
  // CHECK: @llvm.hivm.vldsx1.v128s16
  // CHECK: @llvm.hivm.vabs.x
  // CHECK: @llvm.hivm.vstsx1.v128s16
  func.func @triton_kernel_i16(%arg0: memref<?xi8>, %arg1: memref<?xi16> {tt.divisibility = 16 : i32}, %arg2: memref<?xi16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.cmpi slt, %arg6, %c1_i32 : i32
  %1 = arith.index_cast %arg6 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1], strides: [1] : memref<?xi16> to memref<1xi16, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<1xi16>
  %2 = arith.index_castui %0 : i1 to index
  %subview = memref.subview %reinterpret_cast[0] [%2] [1] : memref<1xi16, strided<[1], offset: ?>> to memref<?xi16, strided<[1], offset: ?>>
  %subview_0 = memref.subview %alloc[0] [%2] [1] : memref<1xi16> to memref<?xi16, strided<[1]>>
  memref.copy %subview, %subview_0 : memref<?xi16, strided<[1], offset: ?>> to memref<?xi16, strided<[1]>>
  %3 = bufferization.to_tensor %alloc restrict writable : memref<1xi16>
  %4 = math.absi %3 : tensor<1xi16>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1], strides: [1] : memref<?xi16> to memref<1xi16, strided<[1], offset: ?>>
  %extracted_slice = tensor.extract_slice %4[0] [%2] [1] : tensor<1xi16> to tensor<?xi16>
  %subview_2 = memref.subview %reinterpret_cast_1[0] [%2] [1] : memref<1xi16, strided<[1], offset: ?>> to memref<?xi16, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xi16>, memref<?xi16, strided<[1], offset: ?>>) -> ()
  return
  }
}