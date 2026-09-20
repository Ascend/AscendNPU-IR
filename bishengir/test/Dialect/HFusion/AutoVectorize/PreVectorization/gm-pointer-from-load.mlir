// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion | FileCheck %s

// CHECK: annotation.mark %alloc keys = ["pad_const"] values = [%c256_i64 : i64] : memref<8xi64>

module {
  func.func @_triton_logits_processor_kernel(%arg0: memref<?xi64>, // array of pointers to GM
                                             %arg1: i32)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c8 = arith.constant 8 : index
    %c256_i64 = arith.constant 256 : i64
    %c0 = arith.constant 0 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    %8 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>>
    %10 = hivm.hir.pointer_cast(%8) [%c8] : memref<?xi64>
    annotation.mark %10 {address_space = #hivm.address_space<gm>} : memref<?xi64>
    %reinterpret_cast_4 = memref.reinterpret_cast %10 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64>
    %alloc = memref.alloc() : memref<8xi64>
    %11 = arith.index_cast %arg1 : i32 to index
    %12 = arith.maxsi %11, %c0 : index
    %13 = arith.minsi %12, %c8 : index
    %14 = arith.cmpi slt, %13, %c8 : index
    scf.if %14 {
      linalg.fill ins(%c256_i64 : i64) outs(%alloc : memref<8xi64>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast_4[0] [%13] [1] : memref<8xi64> to memref<?xi64, strided<[1]>>
    %subview_5 = memref.subview %alloc[0] [%13] [1] : memref<8xi64> to memref<?xi64, strided<[1]>>
    memref.copy %subview, %subview_5 : memref<?xi64, strided<[1]>> to memref<?xi64, strided<[1]>>
    %15 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    hfusion.print " Y: " {hex = false} %15 : tensor<8xi64>
    return
  }
}
