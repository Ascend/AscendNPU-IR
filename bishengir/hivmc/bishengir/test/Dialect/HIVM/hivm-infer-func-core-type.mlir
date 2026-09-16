// RUN: bishengir-opt -hivm-infer-func-core-type %s | FileCheck %s

// CHECK: hivm.module_core_type = #hivm.module_core_type<MIX>
module {
  // CHECK: @F1{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @F1() {
    func.call @F2() : ()->()
    func.call @CUBE() : ()->()
    return
  }

  // CHECK: @F2{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @F2() {
    func.call @CUBE() : ()->()
    return
  }

  // CHECK: @F3{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @F3() {
    func.call @F4() : () -> ()
    return
  }

  // CHECK: @F4{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @F4() {
    func.call @VEC() : () -> ()
    return
  }

  // CHECK: @F5{{.*}}hivm.func_core_type = #hivm.func_core_type<MIX>
  func.func @F5() {
    func.call @CUBE() : () -> ()
    func.call @VEC() : () -> ()
    return
  }

  // CHECK: @CUBE{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @CUBE() {
    %mc = memref.alloc() : memref<256x256xf32>
    %start = arith.constant 0 : index
    %end = arith.constant 1024 : index
    %step = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    scf.for %arg0 = %start to %end step %step {
      %ma = memref.alloc() : memref<256x128xf16>
      %mb = memref.alloc() : memref<128x256xf16>
      %init_condition = arith.cmpi eq, %arg0, %start : index
      hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                            memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                      outs(%mc : memref<256x256xf32>)
    }

    return
  }

  // CHECK: @VEC{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @VEC() {
    %src1 = memref.alloc() : memref<1x10xi32>
    %dst1 = memref.alloc() : memref<3x10xi32>
    %tmp_3 = memref.alloc() : memref<16xi32>
    hivm.hir.vbrc ins(%src1 : memref<1x10xi32>)
                  outs(%dst1 : memref<3x10xi32>)
                  temp_buffer(%tmp_3 : memref<16xi32>)
                  broadcast_dims = [0]
    return
  }

  // CHECK: @CUBEVEC{{.*}}hivm.func_core_type = #hivm.func_core_type<MIX>
  func.func @CUBEVEC() {
    %src1 = memref.alloc() : memref<1x10xi32>
    %dst1 = memref.alloc() : memref<3x10xi32>
    %tmp_3 = memref.alloc() : memref<16xi32>
    hivm.hir.vbrc ins(%src1 : memref<1x10xi32>)
                  outs(%dst1 : memref<3x10xi32>)
                  temp_buffer(%tmp_3 : memref<16xi32>)
                  broadcast_dims = [0]

    %mc = memref.alloc() : memref<256x256xf32>
    %start = arith.constant 0 : index
    %end = arith.constant 1024 : index
    %step = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    scf.for %arg0 = %start to %end step %step {
      // Data loaded to L1
      %ma = memref.alloc() : memref<256x128xf16>
      %mb = memref.alloc() : memref<128x256xf16>
      // L0C data is cleared for the first iteration
      %init_condition = arith.cmpi eq, %arg0, %start : index
      hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                            memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                      outs(%mc : memref<256x256xf32>)
    }

    return
  }

  // CHECK: @fn_rope1{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @fn_rope1(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {func_dyn_memref_args = dense<[true, true, false]> : vector<3xi1>, global_kernel = "local", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.entry = ""} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256_i32 = arith.constant 256 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %1, %c256_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [16, 16], strides: [%c32, %c1] : memref<?xf32> to memref<16x16xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xf32>
    hivm.hir.copy ins(%reinterpret_cast : memref<16x16xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<16x16xf32>)
    %4 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    %5 = arith.index_cast %2 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [16, 16], strides: [%c32, %c1] : memref<?xf32> to memref<16x16xf32, strided<[?, ?], offset: ?>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_0 : (tensor<16x16xf32>, memref<16x16xf32, strided<[?, ?], offset: ?>>) -> ()
    return
  }

  // CHECK: @hivm_tensor_copy_gm_to_ub{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @hivm_tensor_copy_gm_to_ub() -> tensor<16x16xf32> {
  %src = tensor.empty() : tensor<16x16xf32>
  %dst = tensor.empty() : tensor<16x16xf32>
  %res = hivm.hir.copy ins(%src : tensor<16x16xf32>) outs(%dst : tensor<16x16xf32>)
                       -> tensor<16x16xf32>
  return %res : tensor<16x16xf32>
}

  // CHECK: @fused_dot_add1{{.*}}hivm.func_core_type = #hivm.func_core_type<MIX>
  func.func @fused_dot_add1(%arg0: i64 {hfusion.ffts_base_address}, %arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>, %arg3: tensor<256x256xf32> {hfusion.func_arg_for_v}, %arg4: tensor<256x256xf32>, %arg5: tensor<65536xf32>) -> tensor<256x256xf32> attributes {ascend_rt.lower_to_hfusion, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "fused_dot_add1_tiling_func", hacc.block_dim = 20 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>} {
  %extracted_slice = tensor.extract_slice %arg5[0] [65536] [1] : tensor<65536xf32> to tensor<65536xf32>
  %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [256, 256] : tensor<65536xf32> into tensor<256x256xf32>
  %c4_i64 = arith.constant 4 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c256_i64 = arith.constant 256 : i64
  %c128_i64 = arith.constant 128 : i64
  %0 = hivm.hir.mix_matmul {post_vector_func = "bishengir_gen_vector_epilogue_func"} ins(%arg1, %arg2 : tensor<256x256xf32>, tensor<256x256xf32>) post_vector_func_ins(%arg3 : tensor<256x256xf32>) outs(%expanded : tensor<256x256xf32>) block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64) process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64) swizzle_offset = %c1_i64 : i64 swizzle_direction = %c0_i64 : i64 epilogue_p_tiles = %c4_i64 : i64 -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
  }
}

// -----

// CHECK: @mix_matmul_workspace_no_post_vec_func{{.*}}hivm.func_core_type = #hivm.func_core_type<MIX>
func.func @mix_matmul_workspace_no_post_vec_func(%arg0: i64 {hfusion.ffts_base_address},
                %arg1: tensor<14336x4096xf16>,
                %arg2: tensor<128x4096xf16>,
                %arg3: tensor<128x14336xf16> {hacc.arg_type = #hacc.arg_type<output>},
                %arg4: tensor<1835008xf16>) -> tensor<128x14336xf16>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
            hacc.tiling_func = "main_tiling_func",
            hacc.block_dim = 20 : i64,
            hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>} {
  %extracted_slice = tensor.extract_slice %arg4[0] [1835008] [1] : tensor<1835008xf16> to tensor<1835008xf16>
  %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [128, 14336] : tensor<1835008xf16> into tensor<128x14336xf16>
  %c128_i64 = arith.constant 128 : i64
  %c256_i64 = arith.constant 256 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %0 = hivm.hir.mix_matmul
         ins(%arg2, %arg1 : tensor<128x4096xf16>, tensor<14336x4096xf16>)
         workspace_ins(%expanded : tensor<128x14336xf16>)
         outs(%arg3 : tensor<128x14336xf16>)
         b_transpose block_sizes(%c256_i64, %c256_i64, %c128_i64 : i64, i64, i64)
         process_sizes(%c64_i64, %c256_i64, %c128_i64 : i64, i64, i64)
         swizzle_offset = %c1_i64 : i64
         swizzle_direction = %c1_i64 : i64
         epilogue_p_tiles = %c4_i64 : i64 -> tensor<128x14336xf16>
  return%0 : tensor<128x14336xf16>
}

// -----

// CHECK: hivm.module_core_type = #hivm.module_core_type<AIC>
module {
  // CHECK: @CUBE{{.*}}hivm.func_core_type = #hivm.func_core_type<AIC>
  func.func @CUBE() {
    %mc = memref.alloc() : memref<256x256xf32>
    %start = arith.constant 0 : index
    %end = arith.constant 1024 : index
    %step = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    scf.for %arg0 = %start to %end step %step {
      %ma = memref.alloc() : memref<256x128xf16>
      %mb = memref.alloc() : memref<128x256xf16>
      %init_condition = arith.cmpi eq, %arg0, %start : index
      hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                            memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                      outs(%mc : memref<256x256xf32>)
    }

    return
  }
}

// -----

// CHECK: hivm.module_core_type = #hivm.module_core_type<AIV>
module {
  // CHECK: @VEC{{.*}}hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @VEC() {
    %src1 = memref.alloc() : memref<1x10xi32>
    %dst1 = memref.alloc() : memref<3x10xi32>
    %tmp_3 = memref.alloc() : memref<16xi32>
    hivm.hir.vbrc ins(%src1 : memref<1x10xi32>)
                  outs(%dst1 : memref<3x10xi32>)
                  temp_buffer(%tmp_3 : memref<16xi32>)
                  broadcast_dims = [0]
    return
  }
}
