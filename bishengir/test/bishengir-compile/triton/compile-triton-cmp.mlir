// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -enable-triton-kernel-compile %s | FileCheck %s

// CHECK: LLVMDialectModule
#map = affine_map<(d0) -> (d0)>
module {
  func.func @triton_elementwise_binary(%arg0: memref<?xi8>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi8>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32], strides: [1] : memref<?xi16> to memref<32xi16, strided<[1]>>
    %alloc = memref.alloc() : memref<32xi16>
    %subview = memref.subview %reinterpret_cast[0] [3] [1] : memref<32xi16, strided<[1]>> to memref<3xi16, strided<[1]>>
    %subview_0 = memref.subview %alloc[0] [3] [1] : memref<32xi16> to memref<3xi16, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<3xi16, strided<[1]>> to memref<3xi16, strided<[1]>>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<32xi16>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32], strides: [1] : memref<?xi16> to memref<32xi16, strided<[1]>>
    %alloc_2 = memref.alloc() : memref<32xi16>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [3] [1] : memref<32xi16, strided<[1]>> to memref<3xi16, strided<[1]>>
    %subview_4 = memref.subview %alloc_2[0] [3] [1] : memref<32xi16> to memref<3xi16, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<3xi16, strided<[1]>> to memref<3xi16, strided<[1]>>
    %1 = bufferization.to_tensor %alloc_2 restrict writable : memref<32xi16>
    %2 = arith.cmpi sge, %0, %1 : tensor<32xi16>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32], strides: [1] : memref<?xi8> to memref<32xi8, strided<[1]>>
    %3 = arith.extui %2 : tensor<32xi1> to tensor<32xi8>
    %extracted_slice = tensor.extract_slice %3[0] [3] [1] : tensor<32xi8> to tensor<3xi8>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [3] [1] : memref<32xi8, strided<[1]>> to memref<3xi8, strided<[1]>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<3xi8>, memref<3xi8, strided<[1]>>) -> ()
    return
  }
}
