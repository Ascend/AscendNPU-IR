// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-hfusion-compile=true --enable-triton-kernel-compile -o %t.o

// Reduced from PD-1848: masked scalar f8 bias load with a dynamic offset,
// followed by f32 addition and a broadcast store.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @masked_scalar_f8_bias_add(%bias: memref<?xf8E4M3FN>, %dst: memref<?xf8E4M3FN>,
                                    %rows: i32, %block: i32, %sum: i32,
                                    %row_stride: i32, %col_stride: i32, %total: i32, %pid: i32)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %zero = arith.constant 0.0 : f8E4M3FN
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : i32
    %bias_index = arith.remsi %rows, %c128 : i32
    %mask = arith.cmpi slt, %bias_index, %c128 : i32
    %bias_offset = arith.index_cast %bias_index : i32 to index
    %bias_view = memref.reinterpret_cast %bias to offset: [%bias_offset], sizes: [1], strides: [1] : memref<?xf8E4M3FN> to memref<1xf8E4M3FN, strided<[1], offset: ?>>
    %value = scf.if %mask -> (f8E4M3FN) {
      %loaded = memref.load %bias_view[%c0] : memref<1xf8E4M3FN, strided<[1], offset: ?>>
      scf.yield %loaded : f8E4M3FN
    } else {
      scf.yield %zero : f8E4M3FN
    }
    %extended = arith.extf %value {round_mode = #hfusion.round_mode<rint>} : f8E4M3FN to f32
    %sum_f32 = arith.sitofp %sum : i32 to f32
    %scalar = arith.addf %sum_f32, %extended : f32
    %truncated = arith.truncf %scalar {round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
    %empty = tensor.empty() : tensor<128xf8E4M3FN>
    %filled = linalg.fill ins(%truncated : f8E4M3FN) outs(%empty : tensor<128xf8E4M3FN>) -> tensor<128xf8E4M3FN>
    bufferization.materialize_in_destination %filled in writable %dst : (tensor<128xf8E4M3FN>, memref<?xf8E4M3FN>) -> ()
    return
  }
}
