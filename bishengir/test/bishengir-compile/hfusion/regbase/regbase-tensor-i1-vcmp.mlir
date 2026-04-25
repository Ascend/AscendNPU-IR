// RUN: bishengir-compile %s --enable-auto-multi-buffer=True --enable-hfusion-compile=true \
// RUN: -enable-lir-compile=false -target=Ascend310B4 --enable-triton-kernel-compile=true

module {
  func.func @triton_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %0 = tensor.empty() : tensor<1xi8>
    %1 = linalg.fill ins(%c0_i8 : i8) outs(%0 : tensor<1xi8>) -> tensor<1xi8>
    %2 = arith.cmpi slt, %arg7, %c1_i32 : i32
    %3 = arith.index_cast %arg7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1xi8>
    %4 = arith.index_castui %2 : i1 to index
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%4] [1] : memref<1xi8> to memref<?xi8, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1]>>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
    %6 = arith.cmpi ne, %5, %1 : tensor<1xi8>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1xi8>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%4] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%4] [1] : memref<1xi8> to memref<?xi8, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1]>>
    %7 = bufferization.to_tensor %alloc_2 restrict writable : memref<1xi8>
    %8 = arith.cmpi ne, %7, %1 : tensor<1xi8>
    %9 = arith.andi %6, %8 : tensor<1xi1>
    %10 = arith.extui %9 : tensor<1xi1> to tensor<1xi8>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %10[0] [%4] [1] : tensor<1xi8> to tensor<?xi8>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%4] [1] : memref<1xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xi8>, memref<?xi8, strided<[1], offset: ?>>) -> ()
    return
  }
}
