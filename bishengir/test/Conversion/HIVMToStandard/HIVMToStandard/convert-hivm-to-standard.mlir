// RUN: bishengir-opt %s -convert-hivm-to-std  -split-input-file| FileCheck %s

// -----
module {
// CHECK-LABEL: test_mmadL1_with_k_init
// CHECK-NOT: hivm
func.func @test_mmadL1_with_k_init() {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i8
  // CHECK: %[[C_1:.*]] = arith.constant -1 : i64
  // CHECK: %[[C:.*]] = memref.alloc() : memref<256x256xf32>
  %mc = memref.alloc() : memref<256x256xf32>
  %start = arith.constant 0 : index
  %end = arith.constant 1024 : index
  %step = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  scf.for %arg0 = %start to %end step %step {
    // Data loaded to L1
    // CHECK: %[[A:.*]] = memref.alloc() : memref<256x128xf16>
    %ma = memref.alloc() : memref<256x128xf16>
    // CHECK: %[[B:.*]] = memref.alloc() : memref<128x256xf16>
    %mb = memref.alloc() : memref<128x256xf16>

    // L0C data is cleared for the first iteration
    // CHECK: %[[INIT:.*]] = arith.cmpi
    %init_condition = arith.cmpi eq, %arg0, %start : index

    // Create dynamic layout memerf
    // CHECK: %[[CASTA:.*]] = memref.cast %[[A]] : memref<256x128xf16> to memref<?x?xf16, strided<[?, ?], offset: ?>>
    // CHECK: %[[CASTB:.*]] = memref.cast %[[B]] : memref<128x256xf16> to memref<?x?xf16, strided<[?, ?], offset: ?>>
    // CHECK: %[[CASTC:.*]] = memref.cast %[[C]] : memref<256x256xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>

    // CHECK: func.call @mma_tile_half_to_float(%[[CASTA]], %[[CASTB]], %[[INIT]], {{.*}}, {{.*}}, {{.*}}, %[[CASTC]], %[[C_1]], %[[C_1]], %[[C_1]], %[[C_1]], %[[C_1]], %[[C_1]], %[[C_1]], %[[C0]]
    // CHECK-SAME:               (memref<?x?xf16, strided<[?, ?], offset: ?>>,
    // CHECK-SAME:                memref<?x?xf16, strided<[?, ?], offset: ?>>,
    // CHECK-SAME:                i1,
    // CHECK-SAME:                index,
    // CHECK-SAME:                index,
    // CHECK-SAME:                index,
    // CHECK-SAME:                memref<?x?xf32, strided<[?, ?], offset: ?>>,
    // CHECK-SAME:                i64, i64, i64, i64, i64, i64, i64,
    // CHECK-SAME:                i8) -> ()
    hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                          memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                    outs(%mc : memref<256x256xf32>)
  }

  return
}
}

// -----
module {
  // CHECK-LABEL: test_mmadL1
  // CHECK-NOT: hivm
  func.func @test_mmadL1() {
    %ma = memref.alloc() : memref<256x128xf16>
    %mb = memref.alloc() : memref<128x256xf16>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    %ma_t = memref.alloc() : memref<128x256xf16>
    // CHECK: call @mma_tile_half_to_float_ta
    hivm.hir.mmadL1 {a_transpose}
                   ins(%ma_t, %mb, %init, %c256, %c128, %c256 :
                         memref<128x256xf16>, memref<128x256xf16>, i1, index, index, index)
                   outs(%mc : memref<256x256xf32>)

    %mb_t = memref.alloc() : memref<256x128xf16>
    // CHECK: call @mma_tile_half_to_float_tb
    hivm.hir.mmadL1 {b_transpose}
                   ins(%ma, %mb_t, %init, %c256, %c128, %c256 :
                         memref<256x128xf16>, memref<256x128xf16>, i1, index, index, index)
                   outs(%mc : memref<256x256xf32>)

    // CHECK: call @mma_tile_half_to_float_ta_tb
    hivm.hir.mmadL1 {a_transpose, b_transpose}
                   ins(%ma_t, %mb_t, %init, %c256, %c128, %c256 :
                         memref<128x256xf16>, memref<256x128xf16>, i1, index, index, index)
                   outs(%mc : memref<256x256xf32>)
    return
 }
}

// -----
module {
  // CHECK-LABEL: test_mmadmxL1_transpose
  func.func @test_mmadmxL1_transpose() {
    %ma = memref.alloc() : memref<256x128xf8E5M2>
    %mb = memref.alloc() : memref<128x256xf8E5M2>
    %ma_t = memref.alloc() : memref<128x256xf8E5M2>
    %mb_t = memref.alloc() : memref<256x128xf8E5M2>
    %scaleA = memref.alloc() : memref<256x4xui8>
    %scaleB = memref.alloc() : memref<256x4xui8>
    %mc = memref.alloc() : memref<256x256xf32>
    %init = arith.constant 1 : i1
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index

    // CHECK: call @mmadmxL1_float8_e5m2_t_to_float_ta
    hivm.hir.mmadmxL1 {a_transpose}
      ins(%ma_t, %mb, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<128x256xf8E5M2>, memref<128x256xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_float8_e5m2_t_to_float_tb
    hivm.hir.mmadmxL1 {b_transpose}
      ins(%ma, %mb_t, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<256x128xf8E5M2>, memref<256x128xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)

    // CHECK: call @mmadmxL1_float8_e5m2_t_to_float_ta_tb
    hivm.hir.mmadmxL1 {a_transpose, b_transpose}
      ins(%ma_t, %mb_t, %scaleA, %scaleB, %init, %c256, %c128, %c256 :
          memref<128x256xf8E5M2>, memref<256x128xf8E5M2>,
          memref<256x4xui8>, memref<256x4xui8>, i1, index, index, index)
      outs(%mc : memref<256x256xf32>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_nd2nz
  // CHECK-NOT: hivm
  func.func @test_nd2nz() {
    %gmA = memref.alloc() : memref<1024x2048xf16>
    %gmASubview = memref.subview %gmA[0, 0][256, 128][1, 1]
                         : memref<1024x2048xf16> to
                           memref<256x128xf16, strided<[2048, 1], offset: 0>>
    %l1A = memref.alloc() : memref<256x128xf16>
    // CHECK: call @nd2nz
    hivm.hir.nd2nz {dst_continuous}
      ins(%gmASubview : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
      outs(%l1A : memref<256x128xf16>)
    return
 }
}

// -----
module {
  func.func @test_fixpipe_nz2nd_no_dual() {
    %gmC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<gm>>
    %l0c = memref.alloc() : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_nz2nd_float_to_half_4d_to_2d_gm
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%l0c : memref<2x2x16x16xf32, #hivm.address_space<cc>>)
      outs(%gmC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<gm>>)
    return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_fixpipe_nz2nd_row_split() {
    %ubC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_nz2nd_dual_float_to_half_4d_to_2d_ub
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>}
      ins(%l0c : memref<2x2x16x16xf32, #hivm.address_space<cc>>)
      outs(%ubC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_fixpipe_nz2nd_column_split() {
    %ubC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_nz2nd_dual_float_to_half_4d_to_2d_ub
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, dual_dst_mode = #hivm.fixpipe_dual_dst_mode<COLUMN_SPLIT>}
      ins(%l0c : memref<2x2x16x16xf32, #hivm.address_space<cc>>)
      outs(%ubC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    return
 }
}

// -----
module {
  func.func @test_fixpipe_normal_nodual() {
    %gmC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<gm>>
    %l0c = memref.alloc() : memref<32x32xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_normal_float_to_half_2d_to_2d_gm
    hivm.hir.fixpipe {}
      ins(%l0c : memref<32x32xf32, #hivm.address_space<cc>>)
      outs(%gmC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<gm>>)
    return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_fixpipe_normal_ROW_SPLIT() {
    %ubC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<32x32xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_normal_dual_float_to_half_2d_to_2d_ub
    hivm.hir.fixpipe {dual_dst_mode = #hivm.fixpipe_dual_dst_mode<ROW_SPLIT>}
      ins(%l0c : memref<32x32xf32, #hivm.address_space<cc>>)
      outs(%ubC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_fixpipe_normal_COLUMN_SPLIT() {
    %ubC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<32x32xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_normal_dual_float_to_half_2d_to_2d_ub
    hivm.hir.fixpipe {dual_dst_mode = #hivm.fixpipe_dual_dst_mode<COLUMN_SPLIT>}
      ins(%l0c : memref<32x32xf32, #hivm.address_space<cc>>)
      outs(%ubC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_fixpipe_nz2dn() {
    %gmC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<gm>>
    %l0c = memref.alloc() : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    // CHECK: call @fixpipe_nz2dn_float_to_half_4d_to_2d_gm
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>}
      ins(%l0c : memref<2x2x16x16xf32, #hivm.address_space<cc>>)
      outs(%gmC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<gm>>)
    return
 }
}

// -----
func.func @hivm_memref_copy_gm_to_ub_bf16(%src : memref<2048xbf16, #hivm.address_space<gm>>,
    %dst : memref<2048xbf16, #hivm.address_space<ub>>) {
  // CHECK: call @load_gm_to_ubuf_1d_bfloat16_t
  hivm.hir.load
    ins(%src : memref<2048xbf16, #hivm.address_space<gm>>)
    outs(%dst : memref<2048xbf16, #hivm.address_space<ub>>)

  return
}

// -----
func.func @hivm_memref_store_ub_to_gm_bf16(%src : memref<2048xbf16, #hivm.address_space<ub>>,
    %dst : memref<2048xbf16, #hivm.address_space<gm>>) {
  // CHECK: call @store_ubuf_to_gm_1d_bfloat16_t
  hivm.hir.store
    ins(%src : memref<2048xbf16, #hivm.address_space<ub>>)
    outs(%dst : memref<2048xbf16, #hivm.address_space<gm>>)

  return
}

// -----
func.func @hivm_memref_load_gm_to_ub_bf16(%src : memref<2048xbf16, #hivm.address_space<gm>>,
    %dst : memref<2048xbf16, #hivm.address_space<ub>>) {
  // CHECK: call @load_gm_to_ubuf_1d_bfloat16_t
  hivm.hir.load
    ins(%src : memref<2048xbf16, #hivm.address_space<gm>>)
    outs(%dst : memref<2048xbf16, #hivm.address_space<ub>>)

  return
}

// -----
module {
  func.func @copyop1d() {
    // CHECK: func.func private @load_gm_to_ubuf_1d_int8_t(memref<{{.*}}, #hivm.address_space<gm>>,
    // CHECK: memref<{{.*}}, #hivm.address_space<ub>>, i32, i8, index, i32)
    %src = memref.alloc() : memref<16xi8, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<16xi8, #hivm.address_space<ub>>
    // CHECK: call @load_gm_to_ubuf_1d_int8_t(%{{.*}}, %{{.*}}, %c0_i32, %c0_i8, %{{.*}}, %c0_i32)
    hivm.hir.load ins(%src : memref<16xi8, #hivm.address_space<gm>>)
                  outs(%dst : memref<16xi8, #hivm.address_space<ub>>)
    return
  }
}


// -----
module {
  // CHECK-LABEL: @copyop1d_with_eviction_policy
  func.func @copyop1d_with_eviction_policy() {
    %src = memref.alloc() : memref<16xi8, #hivm.address_space<gm>>
    %dst_first = memref.alloc() : memref<16xi8, #hivm.address_space<ub>>
    %dst_last = memref.alloc() : memref<16xi8, #hivm.address_space<ub>>

    // CHECK: call @load_gm_to_ubuf_1d_int8_t(%{{.*}}, %{{.*}}, %c0_i32, %c0_i8, %{{.*}}, %c0_i32)
    hivm.hir.load ins(%src : memref<16xi8, #hivm.address_space<gm>>)
                  outs(%dst_first : memref<16xi8, #hivm.address_space<ub>>)
                  eviction_policy = <EvictFirst>

    // CHECK: call @load_gm_to_ubuf_1d_int8_t(%{{.*}}, %{{.*}}, %c0_i32, %c0_i8, %{{.*}}, %c1_i32)
    hivm.hir.load ins(%src : memref<16xi8, #hivm.address_space<gm>>)
                  outs(%dst_last : memref<16xi8, #hivm.address_space<ub>>)
                  eviction_policy = <EvictLast>

    return
  }
}

// -----
module {
  func.func @copyop2d() {
    // CHECK: func.func private @store_ubuf_to_gm_2d_float(memref<{{.*}}, #hivm.address_space<ub>>,
    // CHECK: memref<{{.*}}, #hivm.address_space<gm>>, i32)
    %src = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<16x16xf32, #hivm.address_space<gm>>
    // CHECK: call @store_ubuf_to_gm_2d_float
    hivm.hir.store ins(%src : memref<16x16xf32, #hivm.address_space<ub>>)
                   outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
    return
 }
}

// -----
module {
  func.func @copyop3d() {
    // CHECK: func.func private @load_gm_to_ubuf_3d_half(memref<{{.*}}>, memref<{{.*}}>, i32, f16, index, i32)
    // CHECK: attributes{{.*}}llvm.emit_c_interface
    %val = arith.constant 10.0 : f16
    %src = memref.alloc() : memref<2x1024x2048xf16, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<2x1024x2048xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<2x1024x2048xf16, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x1024x2048xf16, #hivm.address_space<ub>>)
                  pad_mode = #hivm.padmode<PadValue>
                  pad_value = %val : f16
    // CHECK: call @load_gm_to_ubuf_3d_half(%[[src:.*]], %[[dst:.*]], %[[padMode:.*]], %[[padValue:.*]], %[[leftPad:.*]], %[[evict:.*]])
    return
 }
}

// -----
module {
  func.func @copyop4d() {
    // CHECK: func.func private @load_gm_to_ubuf_3d_int16_t(memref<{{.*}}>, memref<{{.*}}>, i32, i16, index, i32)
    // CHECK: attributes{{.*}}llvm.emit_c_interface
    %val = arith.constant 10.0 : f16
    %src = memref.alloc() : memref<2x4x8x32xi16, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<2x4x8x32xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<2x4x8x32xi16, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x4x8x32xi16, #hivm.address_space<ub>>)
                  pad_mode = #hivm.padmode<PadFirstElem>
    // CHECK: call @load_gm_to_ubuf_3d_int16_t(%[[src:.*]], %[[dst:.*]], %[[padMode:.*]], %[[padValue:.*]], %[[leftPad:.*]], %[[evict:.*]])
    return
 }
}

// -----
module {
  func.func @copyop5d() {
    %src = memref.alloc() : memref<16x8x16x32x16xui64, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<16x8x16x32x16xui64, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16x8x16x32x16xui64, #hivm.address_space<gm>>)
                  outs(%dst : memref<16x8x16x32x16xui64, #hivm.address_space<ub>>)
    return
 }
}
// CHECK: %[[zero:.*]] = arith.constant 0 : index
// CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
// CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
// CHECK: %[[srcSlice:.*]] = memref.subview {{.*}}[%[[arg0]], %[[arg1]], 0, 0, 0
// CHECK: %[[dstSlice:.*]] = memref.subview {{.*}}[%[[arg0]], %[[arg1]], 0, 0, 0
// CHECK: func.call @load_gm_to_ubuf_3d_uint64_t

// -----



// -----
module {
  // CHECK: func.func private @vadd_1d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}})
  // CHECK: func.func private @vmul_1d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}})
  // CHECK: func.func private @vsub_1d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}})
  // CHECK: func.func private @vdiv_1d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}})
  // CHECK: func.func private @vmax_1d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}})
  // CHECK: func.func private @vmin_1d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}})
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vv_1d() {
    %ubA = memref.alloc() : memref<16xf16>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xf16> to memref<?xf16>
    %ubB = memref.alloc() : memref<16xf16>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xf16> to memref<?xf16>
    %ubC = memref.alloc() : memref<16xf16>
    %ubC_reinterpret_cast = memref.reinterpret_cast %ubC to offset: [0], sizes: [16], strides: [1] : memref<16xf16> to memref<?xf16>
    hivm.hir.vadd ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xf16>, memref<?xf16>) outs(%ubC_reinterpret_cast : memref<?xf16>)
    hivm.hir.vmul ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xf16>, memref<?xf16>) outs(%ubC_reinterpret_cast : memref<?xf16>)
    hivm.hir.vsub ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xf16>, memref<?xf16>) outs(%ubC_reinterpret_cast : memref<?xf16>)
    hivm.hir.vdiv ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xf16>, memref<?xf16>) outs(%ubC_reinterpret_cast : memref<?xf16>)
    hivm.hir.vmax ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xf16>, memref<?xf16>) outs(%ubC_reinterpret_cast : memref<?xf16>)
    hivm.hir.vmin ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xf16>, memref<?xf16>) outs(%ubC_reinterpret_cast : memref<?xf16>)
    // CHECK: call @vadd_1d_half
    // CHECK: call @vmul_1d_half
    // CHECK: call @vsub_1d_half
    // CHECK: call @vdiv_1d_half
    // CHECK: call @vmax_1d_half
    // CHECK: call @vmin_1d_half
    return
  }
}

// -----
module
{
  func.func @triton_cdiv_1D() {
    %ubA = memref.alloc() : memref<16xi16>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    %ubB = memref.alloc() : memref<16xi16>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    %ubC = memref.alloc() : memref<16xi16>
    %ubC_reinterpret_cast = memref.reinterpret_cast %ubC to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    hivm.hir.vdiv ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xi16>, memref<?xi16>) outs(%ubC_reinterpret_cast : memref<?xi16>) isSigned = false
    // CHECK: call @vdiv_1d_uint16_t
    return
  }
}

// -----
module {
  // CHECK: func.func private @vor_1d_int16_t(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vand_1d_int16_t(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vv_1d_int16() {
    %ubA = memref.alloc() : memref<16xi16>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    %ubB = memref.alloc() : memref<16xi16>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    %ubC = memref.alloc() : memref<16xi16>
    %ubC_reinterpret_cast = memref.reinterpret_cast %ubC to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    hivm.hir.vor ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xi16>, memref<?xi16>) outs(%ubC_reinterpret_cast : memref<?xi16>)
    hivm.hir.vand ins(%ubA_reinterpret_cast, %ubB_reinterpret_cast : memref<?xi16>, memref<?xi16>) outs(%ubC_reinterpret_cast : memref<?xi16>)
    // CHECK: call @vor_1d_int16_t
    // CHECK: call @vand_1d_int16_t
    return
  }
}

// -----
module {
  // CHECK: func.func private @vshls_vs_1d_int16_t(memref<{{.*}}>, i16, memref<{{.*}}>)
  // CHECK: func.func private @vshrs_vs_1d_int16_t(memref<{{.*}}>, i16, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vs_1d_int16(%cst : i16) {
    %ubA = memref.alloc() : memref<16xi16>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    %ubB = memref.alloc() : memref<16xi16>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    hivm.hir.vshl ins(%ubA_reinterpret_cast, %cst : memref<?xi16>, i16) outs(%ubB_reinterpret_cast : memref<?xi16>)
    hivm.hir.vshr ins(%ubA_reinterpret_cast, %cst : memref<?xi16>, i16) outs(%ubB_reinterpret_cast : memref<?xi16>)
    hivm.hir.vshr ins(%ubA_reinterpret_cast, %cst : memref<?xi16>, i16) outs(%ubB_reinterpret_cast : memref<?xi16>)
    round: false
    // CHECK: call @vshls_vs_1d_int16_t
    // CHECK: call @vshrs_vs_1d_int16_t
    // CHECK: call @vshrs_vs_1d_int16_t
    return
  }
}
// -----
module {
  // CHECK: func.func private @vsub_vs_1d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_vs_1d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_elmwise_vector_vs_1d_float32(%cst : f32) {
    %ubA = memref.alloc() : memref<16xf32>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xf32> to memref<?xf32>
    %ubB = memref.alloc() : memref<16xf32>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xf32> to memref<?xf32>
    hivm.hir.vsub ins(%ubA_reinterpret_cast, %cst : memref<?xf32>, f32) outs(%ubB_reinterpret_cast : memref<?xf32>)
    hivm.hir.vdiv ins(%ubA_reinterpret_cast, %cst : memref<?xf32>, f32) outs(%ubB_reinterpret_cast : memref<?xf32>)
    // CHECK: call @vsub_vs_1d_float
    // CHECK: call @vdiv_vs_1d_float
    return
  }
}
// -----
module {
  // CHECK: func.func private @vsub_sv_1d_float(f32, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_sv_1d_float(f32, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_elmwise_vector_sv_1d_float32(%cst : f32) {
    %ubA = memref.alloc() : memref<16xf32>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xf32> to memref<?xf32>
    %ubB = memref.alloc() : memref<16xf32>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xf32> to memref<?xf32>
    hivm.hir.vsub ins(%cst, %ubA_reinterpret_cast : f32, memref<?xf32>) outs(%ubB_reinterpret_cast : memref<?xf32>)
    hivm.hir.vdiv ins(%cst, %ubA_reinterpret_cast : f32, memref<?xf32>) outs(%ubB_reinterpret_cast : memref<?xf32>)
    // CHECK: call @vsub_sv_1d_float
    // CHECK: call @vdiv_sv_1d_float
    return
  }
}
// -----
module {
  // CHECK: func.func private @vexp_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vabs_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vln_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vrelu_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vrsqrt_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vsqrt_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vrec_1d_half(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_unary_vector_1d() {
    %ubA = memref.alloc() : memref<16xf16>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xf16> to memref<?xf16>
    %ubB = memref.alloc() : memref<16xf16>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xf16> to memref<?xf16>
    hivm.hir.vexp ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    hivm.hir.vabs ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    hivm.hir.vln ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    hivm.hir.vrelu ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    hivm.hir.vrsqrt ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    hivm.hir.vsqrt ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    hivm.hir.vrec ins(%ubA_reinterpret_cast : memref<?xf16>) outs(%ubB_reinterpret_cast : memref<?xf16>)
    // CHECK: call @vexp_1d_half
    // CHECK: call @vabs_1d_half
    // CHECK: call @vln_1d_half
    // CHECK: call @vrelu_1d_half
    // CHECK: call @vrsqrt_1d_half
    // CHECK: call @vsqrt_1d_half
    // CHECK: call @vrec_1d_half
    return
  }
}

// -----
module {
  // CHECK: func.func private @vnot_1d_int16_t(memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_unary_vector_1d_int16_t() {
    %ubA = memref.alloc() : memref<16xi16>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    %ubB = memref.alloc() : memref<16xi16>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [16], strides: [1] : memref<16xi16> to memref<?xi16>
    hivm.hir.vnot ins(%ubA_reinterpret_cast : memref<?xi16>) outs(%ubB_reinterpret_cast : memref<?xi16>)
    // CHECK: call @vnot_1d_int16_t
    return
  }
}

// -----
module {
  // CHECK: func.func private @vadds_vs_3d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: func.func private @vmuls_vs_3d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: func.func private @vmaxs_vs_3d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: func.func private @vmins_vs_3d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vs_4d(%cst : f32) {
    %ubA = memref.alloc() : memref<2x2x2x2xf32>
    %ubB = memref.alloc() : memref<2x2x2x2xf32>
    hivm.hir.vadd ins(%ubA, %cst : memref<2x2x2x2xf32>, f32) outs(%ubB : memref<2x2x2x2xf32>)
    hivm.hir.vmul ins(%ubA, %cst : memref<2x2x2x2xf32>, f32) outs(%ubB : memref<2x2x2x2xf32>)
    hivm.hir.vmax ins(%ubA, %cst : memref<2x2x2x2xf32>, f32) outs(%ubB : memref<2x2x2x2xf32>)
    hivm.hir.vmin ins(%ubA, %cst : memref<2x2x2x2xf32>, f32) outs(%ubB : memref<2x2x2x2xf32>)
    // CHECK: call @vadds_vs_3d_float
    // CHECK: call @vmuls_vs_3d_float
    // CHECK: call @vmaxs_vs_3d_float
    // CHECK: call @vmins_vs_3d_float
    return
  }
}

// -----
module {
  // CHECK: func.func private @vadd_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmul_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vsub_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmax_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmin_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vv_2d() {
    %ubA = memref.alloc() : memref<3x16xf16>
    %ubB = memref.alloc() : memref<3x16xf16>
    %ubC = memref.alloc() : memref<3x16xf16>
    hivm.hir.vadd ins(%ubA, %ubB : memref<3x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>)
    hivm.hir.vmul ins(%ubA, %ubB : memref<3x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>)
    hivm.hir.vsub ins(%ubA, %ubB : memref<3x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>)
    hivm.hir.vdiv ins(%ubA, %ubB : memref<3x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>)
    hivm.hir.vmax ins(%ubA, %ubB : memref<3x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>)
    hivm.hir.vmin ins(%ubA, %ubB : memref<3x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>)
    // CHECK: call @vadd_2d_half
    // CHECK: call @vmul_2d_half
    // CHECK: call @vsub_2d_half
    // CHECK: call @vdiv_2d_half
    // CHECK: call @vmax_2d_half
    // CHECK: call @vmin_2d_half
    return
  }
}

// -----
module {
  // CHECK: func.func private @vadd_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmul_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vsub_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmax_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmin_2d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vv_2d_opt() {
    %ubA = memref.alloc() : memref<1x16xf16>
    %ubB = memref.alloc() : memref<3x16xf16>
    %ubC = memref.alloc() : memref<3x16xf16>
    hivm.hir.vadd ins(%ubA, %ubB : memref<1x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>) broadcast = [0]
    hivm.hir.vmul ins(%ubA, %ubB : memref<1x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>) broadcast = [0]
    hivm.hir.vsub ins(%ubA, %ubB : memref<1x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>) broadcast = [0]
    hivm.hir.vdiv ins(%ubA, %ubB : memref<1x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>) broadcast = [0]
    hivm.hir.vmax ins(%ubA, %ubB : memref<1x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>) broadcast = [0]
    hivm.hir.vmin ins(%ubA, %ubB : memref<1x16xf16>, memref<3x16xf16>) outs(%ubC : memref<3x16xf16>) broadcast = [0]
    // CHECK: call @vadd_2d_half
    // CHECK: call @vmul_2d_half
    // CHECK: call @vsub_2d_half
    // CHECK: call @vdiv_2d_half
    // CHECK: call @vmax_2d_half
    // CHECK: call @vmin_2d_half
    return
  }
}
// -----
module {
  // CHECK: func.func private @vshls_vs_2d_int16_t(memref<{{.*}}>, i16, memref<{{.*}}>)
  // CHECK: func.func private @vshrs_vs_2d_int16_t(memref<{{.*}}>, i16, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vs_2d_int16(%cst : i16) {
    %ubA = memref.alloc() : memref<1x16xi16>
    %ubB = memref.alloc() : memref<1x16xi16>
    hivm.hir.vshl ins(%ubA, %cst : memref<1x16xi16>, i16) outs(%ubB : memref<1x16xi16>)
    hivm.hir.vshr ins(%ubA, %cst : memref<1x16xi16>, i16) outs(%ubB : memref<1x16xi16>)
    hivm.hir.vshr ins(%ubA, %cst : memref<1x16xi16>, i16) outs(%ubB : memref<1x16xi16>)
    round: false
    // CHECK: call @vshls_vs_2d_int16_t
    // CHECK: call @vshrs_vs_2d_int16_t
    // CHECK: call @vshrs_vs_2d_int16_t
    return
  }
}
// -----
module {
  // CHECK: func.func private @vsub_vs_2d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_vs_2d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_elmwise_vector_vs_2d_float32(%cst : f32) {
    %ubA = memref.alloc() : memref<2x16xf32>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<2x16xf32> to memref<?x16xf32>
    %ubB = memref.alloc() : memref<2x16xf32>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<2x16xf32> to memref<?x16xf32>
    hivm.hir.vsub ins(%ubA_reinterpret_cast, %cst : memref<?x16xf32>, f32) outs(%ubB_reinterpret_cast : memref<?x16xf32>)
    hivm.hir.vdiv ins(%ubA_reinterpret_cast, %cst : memref<?x16xf32>, f32) outs(%ubB_reinterpret_cast : memref<?x16xf32>)
    // CHECK: call @vsub_vs_2d_float
    // CHECK: call @vdiv_vs_2d_float
    return
  }
}
// -----
module {
  // CHECK: func.func private @vsub_sv_2d_float(f32, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_sv_2d_float(f32, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_elmwise_vector_sv_2d_float32(%cst : f32) {
    %ubA = memref.alloc() : memref<2x16xf32>
    %ubA_reinterpret_cast = memref.reinterpret_cast %ubA to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<2x16xf32> to memref<?x16xf32>
    %ubB = memref.alloc() : memref<2x16xf32>
    %ubB_reinterpret_cast = memref.reinterpret_cast %ubB to offset: [0], sizes: [2, 16], strides: [16, 1] : memref<2x16xf32> to memref<?x16xf32>
    hivm.hir.vsub ins(%cst, %ubA_reinterpret_cast : f32, memref<?x16xf32>) outs(%ubB_reinterpret_cast : memref<?x16xf32>)
    hivm.hir.vdiv ins(%cst, %ubA_reinterpret_cast : f32, memref<?x16xf32>) outs(%ubB_reinterpret_cast : memref<?x16xf32>)
    // CHECK: call @vsub_sv_2d_float
    // CHECK: call @vdiv_sv_2d_float
    return
  }
}
// -----
module {
  // CHECK: func.func private @vadd_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmul_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vsub_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmax_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmin_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vv_3d() {
    %ubA = memref.alloc() : memref<3x8x16xf16>
    %ubB = memref.alloc() : memref<3x8x16xf16>
    %ubC = memref.alloc() : memref<3x8x16xf16>
    hivm.hir.vadd ins(%ubA, %ubB : memref<3x8x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>)
    hivm.hir.vmul ins(%ubA, %ubB : memref<3x8x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>)
    hivm.hir.vsub ins(%ubA, %ubB : memref<3x8x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>)
    hivm.hir.vdiv ins(%ubA, %ubB : memref<3x8x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>)
    hivm.hir.vmax ins(%ubA, %ubB : memref<3x8x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>)
    hivm.hir.vmin ins(%ubA, %ubB : memref<3x8x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>)
    // CHECK: call @vadd_3d_half
    // CHECK: call @vmul_3d_half
    // CHECK: call @vsub_3d_half
    // CHECK: call @vdiv_3d_half
    // CHECK: call @vmax_3d_half
    // CHECK: call @vmin_3d_half
    return
  }
}

// -----
module {
  // CHECK: func.func private @vadd_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}, memref<{{.*}}>)
  // CHECK: func.func private @vmul_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}, memref<{{.*}}>>)
  // CHECK: func.func private @vsub_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}, memref<{{.*}}>>)
  // CHECK: func.func private @vdiv_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}, memref<{{.*}}>>)
  // CHECK: func.func private @vmax_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}, memref<{{.*}}>>)
  // CHECK: func.func private @vmin_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}, memref<{{.*}}>>)
  func.func @test_binary_vector_vv_4d() {
    %ubA = memref.alloc() : memref<3x8x2x16xf16>
    %ubB = memref.alloc() : memref<3x8x2x16xf16>
    %ubC = memref.alloc() : memref<3x8x2x16xf16>
    hivm.hir.vadd ins(%ubA, %ubB : memref<3x8x2x16xf16>, memref<3x8x2x16xf16>) outs(%ubC : memref<3x8x2x16xf16>)
    hivm.hir.vmul ins(%ubA, %ubB : memref<3x8x2x16xf16>, memref<3x8x2x16xf16>) outs(%ubC : memref<3x8x2x16xf16>)
    hivm.hir.vsub ins(%ubA, %ubB : memref<3x8x2x16xf16>, memref<3x8x2x16xf16>) outs(%ubC : memref<3x8x2x16xf16>)
    hivm.hir.vdiv ins(%ubA, %ubB : memref<3x8x2x16xf16>, memref<3x8x2x16xf16>) outs(%ubC : memref<3x8x2x16xf16>)
    hivm.hir.vmax ins(%ubA, %ubB : memref<3x8x2x16xf16>, memref<3x8x2x16xf16>) outs(%ubC : memref<3x8x2x16xf16>)
    hivm.hir.vmin ins(%ubA, %ubB : memref<3x8x2x16xf16>, memref<3x8x2x16xf16>) outs(%ubC : memref<3x8x2x16xf16>)
    // CHECK: call @vadd_3d_half
    // CHECK: call @vmul_3d_half
    // CHECK: call @vsub_3d_half
    // CHECK: call @vdiv_3d_half
    // CHECK: call @vmax_3d_half
    // CHECK: call @vmin_3d_half
    return
  }
}

// -----
module {
  // CHECK: func.func private @vadd_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmul_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vsub_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmax_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vmin_3d_half(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vv_3d_opt() {
    %ubA = memref.alloc() : memref<3x1x16xf16>
    %ubB = memref.alloc() : memref<3x8x16xf16>
    %ubC = memref.alloc() : memref<3x8x16xf16>
    hivm.hir.vadd ins(%ubA, %ubB : memref<3x1x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>) broadcast = [1]
    hivm.hir.vmul ins(%ubA, %ubB : memref<3x1x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>) broadcast = [1]
    hivm.hir.vsub ins(%ubA, %ubB : memref<3x1x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>) broadcast = [1]
    hivm.hir.vdiv ins(%ubA, %ubB : memref<3x1x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>) broadcast = [1]
    hivm.hir.vmax ins(%ubA, %ubB : memref<3x1x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>) broadcast = [1]
    hivm.hir.vmin ins(%ubA, %ubB : memref<3x1x16xf16>, memref<3x8x16xf16>) outs(%ubC : memref<3x8x16xf16>) broadcast = [1]
    // CHECK: call @vadd_3d_half
    // CHECK: call @vmul_3d_half
    // CHECK: call @vsub_3d_half
    // CHECK: call @vdiv_3d_half
    // CHECK: call @vmax_3d_half
    // CHECK: call @vmin_3d_half
    return
  }
}

// -----
module {
  // CHECK: func.func private @vshls_vs_3d_int16_t(memref<{{.*}}>, i16, memref<{{.*}}>)
  // CHECK: func.func private @vshrs_vs_3d_int16_t(memref<{{.*}}>, i16, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_binary_vector_vs_3d_int16(%cst : i16) {
    %ubA = memref.alloc() : memref<3x8x16xi16>
    %ubB = memref.alloc() : memref<3x8x16xi16>
    hivm.hir.vshl ins(%ubA, %cst : memref<3x8x16xi16>, i16) outs(%ubB : memref<3x8x16xi16>)
    hivm.hir.vshr ins(%ubA, %cst : memref<3x8x16xi16>, i16) outs(%ubB : memref<3x8x16xi16>)
    hivm.hir.vshr ins(%ubA, %cst : memref<3x8x16xi16>, i16) outs(%ubB : memref<3x8x16xi16>)
    round: false
    // CHECK: call @vshls_vs_3d_int16_t
    // CHECK: call @vshrs_vs_3d_int16_t
    // CHECK: call @vshrs_vs_3d_int16_t
    return
  }
}
// -----
module {
  // CHECK: func.func private @vsub_vs_3d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_vs_3d_float(memref<{{.*}}>, f32, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_elmwise_vector_vs_3d_float32(%cst : f32) {
    %ubA = memref.alloc() : memref<3x8x16xf32>
    %ubB = memref.alloc() : memref<3x8x16xf32>
    hivm.hir.vsub ins(%ubA, %cst : memref<3x8x16xf32>, f32) outs(%ubB : memref<3x8x16xf32>)
    hivm.hir.vdiv ins(%ubA, %cst : memref<3x8x16xf32>, f32) outs(%ubB : memref<3x8x16xf32>)
    // CHECK: call @vsub_vs_3d_float
    // CHECK: call @vdiv_vs_3d_float
    return
  }
}
// -----
module {
  // CHECK: func.func private @vsub_sv_3d_float(f32, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: func.func private @vdiv_sv_3d_float(f32, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_elmwise_vector_sv_3d_float32(%cst : f32) {
    %ubA = memref.alloc() : memref<3x8x16xf32>
    %ubB = memref.alloc() : memref<3x8x16xf32>
    hivm.hir.vsub ins(%cst, %ubA : f32, memref<3x8x16xf32>) outs(%ubB : memref<3x8x16xf32>)
    hivm.hir.vdiv ins(%cst, %ubA : f32, memref<3x8x16xf32>) outs(%ubB : memref<3x8x16xf32>)
    // CHECK: call @vsub_sv_3d_float
    // CHECK: call @vdiv_sv_3d_float
    return
  }
}
// -----
module {
  // CHECK: func.func private @vadd_3d_int16_t(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_vadd_vv_5d(%src1 : memref<2x4x8x32x64xi16>, %src2 : memref<2x4x8x32x64xi16>, %dst : memref<2x4x8x32x64xi16>) {
    hivm.hir.vadd ins(%src1, %src2 : memref<2x4x8x32x64xi16>, memref<2x4x8x32x64xi16>)
                  outs(%dst : memref<2x4x8x32x64xi16>)
    // CHECK: scf.for %[[iv1:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: scf.for %[[iv2:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: %[[src1Slice:.*]] = memref.subview {{.*}}[%[[iv1]], %[[iv2]], 0, 0, 0
    // CHECK: %[[src2Slice:.*]] = memref.subview {{.*}}[%[[iv1]], %[[iv2]], 0, 0, 0
    // CHECK: %[[dstSlice:.*]] = memref.subview {{.*}}[%[[iv1]], %[[iv2]], 0, 0, 0
    // func.call @vadd_3d_int16_t
    return
  }
}

// -----
module {
  // CHECK: func.func private @vadds_vs_3d_int32_t(memref<{{.*}}>, i32, memref<{{.*}}>)
  // CHECK: attributes{{.*}}llvm.emit_c_interface
  func.func @test_vadd_vs_6d(%src1 : memref<2x4x8x32x64x128xi32>, %dst : memref<2x4x8x32x64x128xi32>, %cst : i32) {
    hivm.hir.vadd ins(%src1, %cst : memref<2x4x8x32x64x128xi32>, i32)
                  outs(%dst : memref<2x4x8x32x64x128xi32>)
    // CHECK: scf.for %[[iv1:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: scf.for %[[iv2:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: scf.for %[[iv3:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: %[[src1Slice:.*]] = memref.subview {{.*}}[%[[iv1]], %[[iv2]], %[[iv3]], 0, 0, 0
    // CHECK: %[[dstSlice:.*]] = memref.subview {{.*}}[%[[iv1]], %[[iv2]], %[[iv3]], 0, 0, 0
    // func.call @vadds_vs_3d_int32_t
    return
  }
}

// -----

// CHECK: call @load_gm_to_ubuf_1d_half
// CHECK: call @load_gm_to_ubuf_1d_half
module {
func.func @test_duplicate_op(%valueA: memref<16xf16, #hivm.address_space<gm>>,
                          %valueB : memref<16xf16, #hivm.address_space<gm>>,
                          %valueC : memref<16xf16, #hivm.address_space<gm>>)
{
  %ubA = memref.alloca() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueA : memref<16xf16, #hivm.address_space<gm>>) outs(%ubA : memref<16xf16, #hivm.address_space<ub>>)
  %ubB = memref.alloca() : memref<16xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%valueB : memref<16xf16, #hivm.address_space<gm>>) outs(%ubB : memref<16xf16, #hivm.address_space<ub>>)
  return
}
}

// -----
module {
  // CHECK: func.func private @vcast_bool_to_half_1d_with_temp(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_bool_to_float_1d_with_temp(memref<{{.*}}>, memref<{{.*}}>, i32)
  func.func @test_vcast_1d_with_temp() {
    %s1 = memref.alloc() : memref<16xi1>
    %f16 = memref.alloc() : memref<16xf16>
    %f32 = memref.alloc() : memref<16xf32>

    hivm.hir.vcast ins(%s1 : memref<16xi1>) outs(%f16 : memref<16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s1 : memref<16xi1>) outs(%f32 : memref<16xf32>)
                   round_mode = #hivm.round_mode<rint>

    // CHECK: call @vcast_bool_to_half_1d_with_temp
    // CHECK: call @vcast_bool_to_float_1d_with_temp
    return
  }
}

// -----
module {
  // CHECK: func.func private @vcast_bfloat16_t_to_float_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_bfloat16_t_to_int32_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_half_to_float_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_half_to_int16_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_half_to_int32_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_half_to_int4_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_half_to_int8_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_half_to_uint8_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_float_to_bfloat16_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_float_to_half_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_float_to_float_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_float_to_int16_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_float_to_int32_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_float_to_int64_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int16_t_to_half_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int16_t_to_float_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int32_t_to_float_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int32_t_to_int16_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int32_t_to_int64_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int4_t_to_half_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int64_t_to_float_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int64_t_to_int32_t_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_int8_t_to_half_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  // CHECK: func.func private @vcast_uint8_t_to_half_2d_with_mode(memref<{{.*}}>, memref<{{.*}}>, i32)
  func.func @test_vcast_2d() {
    %f16 = memref.alloc() : memref<2x16xf16>
    %f32 = memref.alloc() : memref<2x16xf32>
    %s4 = memref.alloc() : memref<2x16xi4>
    %s8 = memref.alloc() : memref<2x16xi8>
    %s16 = memref.alloc() : memref<2x16xi16>
    %s32 = memref.alloc() : memref<2x16xi32>
    %s64 = memref.alloc() : memref<2x16xi64>
    %bf16 = memref.alloc() : memref<2x16xbf16>
    %u8 = memref.alloc() : memref<2x16xui8>
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<odd>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%s16 : memref<2x16xi16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%s64 : memref<2x16xi64>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s4 : memref<2x16xi4>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<round>
    hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<ceil>
    hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<floor>
    hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                   round_mode = #hivm.round_mode<trunc>
    hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%s32 : memref<2x16xi32>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%s8 : memref<2x16xi8>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    hivm.hir.vcast ins(%u8 : memref<2x16xui8>) outs(%f16 : memref<2x16xf16>)
                   round_mode = #hivm.round_mode<rint>
    // CHECK: call @vcast_bfloat16_t_to_float_2d_with_mode
    // CHECK: call @vcast_bfloat16_t_to_int32_t_2d_with_mode
    // CHECK: call @vcast_bfloat16_t_to_int32_t_2d_with_mode
    // CHECK: call @vcast_bfloat16_t_to_int32_t_2d_with_mode
    // CHECK: call @vcast_bfloat16_t_to_int32_t_2d_with_mode
    // CHECK: call @vcast_bfloat16_t_to_int32_t_2d_with_mode
    // CHECK: call @vcast_half_to_float_2d_with_mode
    // CHECK: call @vcast_half_to_int16_t_2d_with_mode
    // CHECK: call @vcast_half_to_int16_t_2d_with_mode
    // CHECK: call @vcast_half_to_int16_t_2d_with_mode
    // CHECK: call @vcast_half_to_int16_t_2d_with_mode
    // CHECK: call @vcast_half_to_int16_t_2d_with_mode
    // CHECK: call @vcast_half_to_int32_t_2d_with_mode
    // CHECK: call @vcast_half_to_int32_t_2d_with_mode
    // CHECK: call @vcast_half_to_int32_t_2d_with_mode
    // CHECK: call @vcast_half_to_int32_t_2d_with_mode
    // CHECK: call @vcast_half_to_int32_t_2d_with_mode
    // CHECK: call @vcast_half_to_int4_t_2d_with_mode
    // CHECK: call @vcast_half_to_int4_t_2d_with_mode
    // CHECK: call @vcast_half_to_int4_t_2d_with_mode
    // CHECK: call @vcast_half_to_int4_t_2d_with_mode
    // CHECK: call @vcast_half_to_int4_t_2d_with_mode
    // CHECK: call @vcast_half_to_int4_t_2d_with_mode
    // CHECK: call @vcast_half_to_int8_t_2d_with_mode
    // CHECK: call @vcast_half_to_int8_t_2d_with_mode
    // CHECK: call @vcast_half_to_int8_t_2d_with_mode
    // CHECK: call @vcast_half_to_int8_t_2d_with_mode
    // CHECK: call @vcast_half_to_int8_t_2d_with_mode
    // CHECK: call @vcast_half_to_int8_t_2d_with_mode
    // CHECK: call @vcast_half_to_uint8_t_2d_with_mode
    // CHECK: call @vcast_half_to_uint8_t_2d_with_mode
    // CHECK: call @vcast_half_to_uint8_t_2d_with_mode
    // CHECK: call @vcast_half_to_uint8_t_2d_with_mode
    // CHECK: call @vcast_half_to_uint8_t_2d_with_mode
    // CHECK: call @vcast_half_to_uint8_t_2d_with_mode
    // CHECK: call @vcast_float_to_bfloat16_t_2d_with_mode
    // CHECK: call @vcast_float_to_bfloat16_t_2d_with_mode
    // CHECK: call @vcast_float_to_bfloat16_t_2d_with_mode
    // CHECK: call @vcast_float_to_bfloat16_t_2d_with_mode
    // CHECK: call @vcast_float_to_bfloat16_t_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_half_2d_with_mode
    // CHECK: call @vcast_float_to_float_2d_with_mode
    // CHECK: call @vcast_float_to_float_2d_with_mode
    // CHECK: call @vcast_float_to_float_2d_with_mode
    // CHECK: call @vcast_float_to_float_2d_with_mode
    // CHECK: call @vcast_float_to_float_2d_with_mode
    // CHECK: call @vcast_float_to_int16_t_2d_with_mode
    // CHECK: call @vcast_float_to_int16_t_2d_with_mode
    // CHECK: call @vcast_float_to_int16_t_2d_with_mode
    // CHECK: call @vcast_float_to_int16_t_2d_with_mode
    // CHECK: call @vcast_float_to_int16_t_2d_with_mode
    // CHECK: call @vcast_float_to_int32_t_2d_with_mode
    // CHECK: call @vcast_float_to_int32_t_2d_with_mode
    // CHECK: call @vcast_float_to_int32_t_2d_with_mode
    // CHECK: call @vcast_float_to_int32_t_2d_with_mode
    // CHECK: call @vcast_float_to_int32_t_2d_with_mode
    // CHECK: call @vcast_float_to_int64_t_2d_with_mode
    // CHECK: call @vcast_float_to_int64_t_2d_with_mode
    // CHECK: call @vcast_float_to_int64_t_2d_with_mode
    // CHECK: call @vcast_float_to_int64_t_2d_with_mode
    // CHECK: call @vcast_float_to_int64_t_2d_with_mode
    // CHECK: call @vcast_int16_t_to_half_2d_with_mode
    // CHECK: call @vcast_int16_t_to_half_2d_with_mode
    // CHECK: call @vcast_int16_t_to_half_2d_with_mode
    // CHECK: call @vcast_int16_t_to_half_2d_with_mode
    // CHECK: call @vcast_int16_t_to_half_2d_with_mode
    // CHECK: call @vcast_int16_t_to_half_2d_with_mode
    // CHECK: call @vcast_int16_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_float_2d_with_mode
    // CHECK: call @vcast_int32_t_to_int16_t_2d_with_mode
    // CHECK: call @vcast_int32_t_to_int64_t_2d_with_mode
    // CHECK: call @vcast_int4_t_to_half_2d_with_mode
    // CHECK: call @vcast_int64_t_to_float_2d_with_mode
    // CHECK: call @vcast_int64_t_to_float_2d_with_mode
    // CHECK: call @vcast_int64_t_to_float_2d_with_mode
    // CHECK: call @vcast_int64_t_to_float_2d_with_mode
    // CHECK: call @vcast_int64_t_to_float_2d_with_mode
    // CHECK: call @vcast_int64_t_to_int32_t_2d_with_mode
    // CHECK: call @vcast_int8_t_to_half_2d_with_mode
    // CHECK: call @vcast_uint8_t_to_half_2d_with_mode
    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_type_suffix() {
    // CHECK: call @broadcast_scalar_int16_t_to_1d
    %c_1 = arith.constant 42 : i16
    %dst_1 = memref.alloca() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_1 : i16)
                  outs(%dst_1 : memref<16xi16, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_scalar_int32_t_to_1d
    %c_3 = arith.constant 42 : i32
    %dst_3 = memref.alloca() : memref<32xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_3 : i32)
                  outs(%dst_3 : memref<32xi32, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_scalar_float_to_1d
    %c_5 = arith.constant 42.0 : f32
    %dst_5 = memref.alloca() : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_5 : f32)
                  outs(%dst_5 : memref<32xf32, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_scalar_half_to_1d
    %c_6 = arith.constant 42.0 : f16
    %dst_6 = memref.alloca() : memref<32xf16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_6 : f16)
                  outs(%dst_6 : memref<32xf16, #hivm.address_space<ub>>)

    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_type_suffix() {
    // CHECK: call @broadcast_scalar_int16_t_to_2d
    %c_1 = arith.constant 42 : i16
    %dst_1_alloca = memref.alloca() : memref<4x16xi16, #hivm.address_space<ub>>
    %dst_1 = memref.subview %dst_1_alloca[0, 0][4, 10][1, 1] :
                            memref<4x16xi16, #hivm.address_space<ub>> to
                            memref<4x10xi16, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_1 : i16)
                  outs(%dst_1 : memref<4x10xi16, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_scalar_int32_t_to_2d
    %c_2 = arith.constant 42 : i32
    %dst_2_alloca = memref.alloca() : memref<4x16xi32, #hivm.address_space<ub>>
    %dst_2 = memref.subview %dst_2_alloca[0, 0][4, 10][1, 1] :
                            memref<4x16xi32, #hivm.address_space<ub>> to
                            memref<4x10xi32, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_2 : i32)
                  outs(%dst_2 : memref<4x10xi32, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_scalar_int8_t_to_2d
    %c_3 = arith.constant 42 : i8
    %dst_3_alloca = memref.alloca() : memref<4x16xi8, #hivm.address_space<ub>>
    %dst_3 = memref.subview %dst_3_alloca[0, 0][4, 10][1, 1] :
                            memref<4x16xi8, #hivm.address_space<ub>> to
                            memref<4x10xi8, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_3 : i8)
                  outs(%dst_3 : memref<4x10xi8, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_scalar_int64_t_to_2d
    %c_4 = arith.constant 42 : i64
    %dst_4_alloca = memref.alloca() : memref<4x16xi64, #hivm.address_space<ub>>
    %dst_4 = memref.subview %dst_4_alloca[0, 0][4, 15][1, 1] :
                            memref<4x16xi64, #hivm.address_space<ub>> to
                            memref<4x15xi64, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_4 : i64)
                  outs(%dst_4 : memref<4x15xi64, strided<[16, 1], offset: 0>, #hivm.address_space<ub>>)
    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_1d() {
    // CHECK: call @broadcast_scalar_int16_t_to_1d
    %c_1 = arith.constant 42 : i16
    %dst_1 = memref.alloca() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c_1 : i16)
                  outs(%dst_1 : memref<16xi16, #hivm.address_space<ub>>)

    // CHECK: call @broadcast_1d_int16_t
    %src_2 = memref.alloca() : memref<1xi16, #hivm.address_space<ub>>
    %dst_2 = memref.alloca() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_2 : memref<1xi16, #hivm.address_space<ub>>)
                  outs(%dst_2 : memref<16xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_without_rank_reducing() {
    // CHECK: call @broadcast_first_axis_align_2d_int16_t
    // CHECK-NOT: scf.for
    %src_1 = memref.alloca() : memref<1x32xi16, #hivm.address_space<ub>>
    %dst_1 = memref.alloca() : memref<16x32xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_1 : memref<1x32xi16, #hivm.address_space<ub>>)
                  outs(%dst_1 : memref<16x32xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_first_axis_align_3d_int16_t
    // CHECK-NOT: scf.for
    %src_2 = memref.alloca() : memref<1x32x64xi16, #hivm.address_space<ub>>
    %dst_2 = memref.alloca() : memref<16x32x64xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_2 : memref<1x32x64xi16, #hivm.address_space<ub>>)
                  outs(%dst_2 : memref<16x32x64xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_last_axis_align_2d_int16_t
    // CHECK-NOT: scf.for
    %src_3 = memref.alloca() : memref<16x1xi16, #hivm.address_space<ub>>
    %dst_3 = memref.alloca() : memref<16x32xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_3 : memref<16x1xi16, #hivm.address_space<ub>>)
                  outs(%dst_3 : memref<16x32xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [1]

    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_with_rank_reducing() {
    // CHECK-COUNT-5: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @broadcast_first_axis_align_3d_int16_t
    %src_3 = memref.alloca() : memref<1x16x8x32x8x16x8x32xi16, #hivm.address_space<ub>>
    %dst_3 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_3 : memref<1x16x8x32x8x16x8x32xi16, #hivm.address_space<ub>>)
                  outs(%dst_3 : memref<16x16x8x32x8x16x8x32xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK-COUNT-6: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @broadcast_last_axis_align_2d_int16_t
    %src_4 = memref.alloca() : memref<16x16x8x32x8x16x8x1xi16, #hivm.address_space<ub>>
    %dst_4 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_4 : memref<16x16x8x32x8x16x8x1xi16, #hivm.address_space<ub>>)
                  outs(%dst_4 : memref<16x16x8x32x8x16x8x32xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [7]

    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_with_rank_reducing_dynamic_shape(
                         %src_5 : memref<1x?x?x?x8x?x8x32xi16, #hivm.address_space<ub>>,
                         %dst_5 : memref<16x?x?x?x8x?x8x32xi16, #hivm.address_space<ub>>) {
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @broadcast_first_axis_align_3d_int16_t
    hivm.hir.vbrc ins(%src_5 : memref<1x?x?x?x8x?x8x32xi16, #hivm.address_space<ub>>)
                  outs(%dst_5 : memref<16x?x?x?x8x?x8x32xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_1d_alignment(%dst_3 : memref<?xi16, #hivm.address_space<ub>>) {
    // CHECK: call @broadcast_1d_int16_t
    %src_1 = memref.alloca() : memref<1xi16, #hivm.address_space<ub>>
    %dst_1 = memref.alloca() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_1 : memref<1xi16, #hivm.address_space<ub>>)
                  outs(%dst_1 : memref<16xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_1d_int16_t
    %src_2 = memref.alloca() : memref<1xi16, #hivm.address_space<ub>>
    %dst_2 = memref.alloca() : memref<17xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_2 : memref<1xi16, #hivm.address_space<ub>>)
                  outs(%dst_2 : memref<17xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_1d_int16_t
    %src_3 = memref.alloca() : memref<1xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_3 : memref<1xi16, #hivm.address_space<ub>>)
                  outs(%dst_3 : memref<?xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_broadcast_op_check_2d_alignment(%dst_9 : memref<?x?xi16, #hivm.address_space<ub>>) {
    // CHECK: call @broadcast_first_axis_align_2d_int16_t
    %src_2 = memref.alloca() : memref<1x32xi16, strided<[32, 1]>, #hivm.address_space<ub>>
    %dst_2 = memref.alloca() : memref<16x32xi16, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_2 : memref<1x32xi16, strided<[32, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_2 : memref<16x32xi16, strided<[32, 1]>, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_first_axis_unalign_2d_int16_t
    %src_3 = memref.alloca() : memref<1x32xi16, strided<[33, 1]>, #hivm.address_space<ub>>
    %dst_3 = memref.alloca() : memref<16x32xi16, strided<[33, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_3 : memref<1x32xi16, strided<[33, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_3 : memref<16x32xi16, strided<[33, 1]>, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_last_axis_align_2d_int16_t
    %src_4 = memref.alloca() : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    %dst_4 = memref.alloca() : memref<32x16xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_4 : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_4 : memref<32x16xi16, strided<[16, 1]>, #hivm.address_space<ub>>)
                  broadcast_dims = [1]

    // CHECK: call @broadcast_last_axis_align_2d_int16_t
    %src_5 = memref.alloca() : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    %dst_5 = memref.alloca() : memref<32x30xi16, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_5 : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_5 : memref<32x30xi16, strided<[32, 1]>, #hivm.address_space<ub>>)
                  broadcast_dims = [1]

    // CHECK: call @broadcast_last_axis_unalign_2d_int16_t
    %src_6 = memref.alloca() : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    %dst_6 = memref.alloca() : memref<32x15xi16, strided<[15, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_6 : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_6 : memref<32x15xi16, strided<[15, 1]>, #hivm.address_space<ub>>)
                  broadcast_dims = [1]

    // CHECK: call @broadcast_last_axis_unalign_2d_int16_t
    %src_7 = memref.alloca() : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    %dst_7 = memref.alloca() : memref<32x15xi16, strided<[33, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_7 : memref<32x1xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_7 : memref<32x15xi16, strided<[33, 1]>, #hivm.address_space<ub>>)
                  broadcast_dims = [1]

    // CHECK: call @broadcast_first_axis_unalign_2d_int16_t
    %src_8 = memref.alloca() : memref<1x31xi16, #hivm.address_space<ub>>
    %dst_8 = memref.alloca() : memref<5x31xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_8 : memref<1x31xi16, #hivm.address_space<ub>>)
                  outs(%dst_8 : memref<5x31xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    // CHECK: call @broadcast_first_axis_unknown_align_2d_int16_t
    %src_9 = memref.alloca() : memref<1x31xi16, strided<[32, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_9 : memref<1x31xi16, strided<[32, 1]>, #hivm.address_space<ub>>)
                  outs(%dst_9 : memref<?x?xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    return
  }
}

// -----

// CHECK-LABEL: func @test_broadcast_op_scalar_3d
// CHECK: scf.for
// CHECK: broadcast_scalar_float_to_2d
module {
  func.func @test_broadcast_op_scalar_3d() {
    %cst = arith.constant 0.0 : f32
    %dst_1 = memref.alloca() : memref<16x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32)
                  outs(%dst_1 : memref<16x16x16xf32, #hivm.address_space<ub>>)

    return
  }
}

// -----

// CHECK-LABEL: func @test_set_l1_2d
// CHECK: @set_l1_2d_float8_e4m3_t
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_set_l1_2d() {
    %cst = arith.constant 0.0 : f8E4M3FN
    %dst_1 = memref.alloca() : memref<256xf8E4M3FN, #hivm.address_space<cbuf>>
    hivm.hir.vbrc ins(%cst : f8E4M3FN)
                  outs(%dst_1 : memref<256xf8E4M3FN, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

// CHECK-LABEL: func @test_broadcast_op_middle
module {
  func.func @test_broadcast_op_middle() {
    // CHECK-COUNT-1: scf.for
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK: broadcast_middle_axis_align_3d
    %src_4r2 = memref.alloca() : memref<16x16x1x16xf32, #hivm.address_space<ub>>
    %dst_4r2 = memref.alloca() : memref<16x16x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_4r2 : memref<16x16x1x16xf32, #hivm.address_space<ub>>)
                  outs(%dst_4r2 : memref<16x16x16x16xf32, #hivm.address_space<ub>>)
                  broadcast_dims = [2]

    // CHECK-COUNT-1: scf.for
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK-COUNT-1: scf.for
    // CHECK: broadcast_first_axis_align_3d_float
    %src_5r1 = memref.alloca() : memref<16x1x16x16x16xf32, #hivm.address_space<ub>>
    %dst_5r1 = memref.alloca() : memref<16x16x16x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_5r1 : memref<16x1x16x16x16xf32, #hivm.address_space<ub>>)
                  outs(%dst_5r1 : memref<16x16x16x16x16xf32, #hivm.address_space<ub>>)
                  broadcast_dims = [1]

    // CHECK-COUNT-4: scf.for
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK-COUNT-3: scf.for
    // CHECK: broadcast_first_axis_align_3d_float
    %src_10r4 = memref.alloca() : memref<16x16x16x16x1x16x16x16x16x16xf32, #hivm.address_space<ub>>
    %dst_10r4 = memref.alloca() : memref<16x16x16x16x16x16x16x16x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_10r4 : memref<16x16x16x16x1x16x16x16x16x16xf32, #hivm.address_space<ub>>)
                  outs(%dst_10r4 : memref<16x16x16x16x16x16x16x16x16x16xf32, #hivm.address_space<ub>>)
                  broadcast_dims = [4]

    // CHECK-COUNT-7: scf.for
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK: %[[subview:.*]] = memref.subview
    // CHECK: broadcast_middle_axis_align_3d
    %src_10r8 = memref.alloca() : memref<16x16x16x16x16x16x16x16x1x16xf32, #hivm.address_space<ub>>
    %dst_10r8 = memref.alloca() : memref<16x16x16x16x16x16x16x16x16x16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_10r8 : memref<16x16x16x16x16x16x16x16x1x16xf32, #hivm.address_space<ub>>)
                  outs(%dst_10r8 : memref<16x16x16x16x16x16x16x16x16x16xf32, #hivm.address_space<ub>>)
                  broadcast_dims = [8]
    return
  }
}


// -----

module {
  func.func @test_broadcast_op_check_temp_buffer() {
    // CHECK: call @broadcast_first_axis_align_2d_int16_t(%{{.+}}, %{{.+}}, %{{.+}})
    %src_2 = memref.alloca() : memref<1x32xi16, #hivm.address_space<ub>>
    %dst_2 = memref.alloca() : memref<16x32xi16, #hivm.address_space<ub>>
    %tmp_2 = memref.alloca() : memref<512xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src_2 : memref<1x32xi16, #hivm.address_space<ub>>)
                  outs(%dst_2 : memref<16x32xi16, #hivm.address_space<ub>>)
                  temp_buffer (%tmp_2 : memref<512xi16, #hivm.address_space<ub>>)
                  broadcast_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_reduce_op_check_arith_op() {
    // CHECK: %[[CST5:.*]] = arith.constant -1 : i32
    // CHECK: %[[CST4:.*]] = arith.constant 2147483647 : i32
    // CHECK: %[[CST3:.*]] = arith.constant -2147483648 : i32
    // CHECK: %[[CST2:.*]] = arith.constant 1 : i32
    // CHECK: %[[CST1:.*]] = arith.constant 0 : i32

    // CHECK: call @reduce_sum_r_int32_t({{.*}}, %[[CST1:.*]])
    %src_2 = memref.alloca() : memref<16xi32>
    %dst_2 = memref.alloca() : memref<1xi32>
    hivm.hir.vreduce <sum>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            unsigned_src = false
                            reduce_dims = [0]

    // CHECK: call @reduce_prod_r_int32_t({{.*}}, %[[CST2:.*]])
    hivm.hir.vreduce <prod>  ins(%src_2 : memref<16xi32>)
                             outs(%dst_2 : memref<1xi32>)
                             unsigned_src = false
                             reduce_dims = [0]

    // CHECK: call @reduce_max_r_int32_t({{.*}}, %[[CST3:.*]])
    hivm.hir.vreduce <max>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            unsigned_src = false
                            reduce_dims = [0]

    // CHECK: call @reduce_min_r_int32_t({{.*}}, %[[CST4:.*]])
    hivm.hir.vreduce <min>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            unsigned_src = false
                            reduce_dims = [0]

    // CHECK: call @reduce_andi_r_int32_t({{.*}}, %[[CST5:.*]])
    hivm.hir.vreduce <andi>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            unsigned_src = false
                            reduce_dims = [0]

    return
  }
}

// -----

module {
  // CHECK: func.func private @reduce_sum_ar_int32_t(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>, i32)
  func.func @test_reduce_op_4d() {
    %src_2 = memref.alloca() : memref<5x2x3x16xi32>
    %dst_2 = memref.alloca() : memref<5x2x3x1xi32>
    // CHECK: %[[CST:.*]] = arith.constant 0 : i32
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: func.call @reduce_sum_ar_int32_t({{.*}}, %[[CST:.*]])
    hivm.hir.vreduce <sum>  ins(%src_2 : memref<5x2x3x16xi32>)
                            outs(%dst_2 : memref<5x2x3x1xi32>)
                            unsigned_src = false
                            reduce_dims = [3]
    return
  }
}

// -----

module {
  func.func @test_reduce_op_check_cross() {
    // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: call @enablevc_reduce_sum_r_half({{.*}}, %[[CST:.*]])
    %src_2 = memref.alloca() : memref<16xf16>
    %dst_2 = memref.alloca() : memref<1xf16>
    hivm.hir.vreduce <sum>  ins(%src_2 : memref<16xf16>)
                            outs(%dst_2 : memref<1xf16>)
                            unsigned_src = false
                            reduce_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_reduce_op_check_with_rank_reducing() {
    // CHECK: %[[CST2:.*]] = arith.constant 2147483647 : i32
    // CHECK: %[[CST1:.*]] = arith.constant -2147483648 : i32
    
    // CHECK-COUNT-5: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @reduce_max_ra0a1_int32_t({{.*}}, %[[CST1:.*]])
    %src_3 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi32>
    %dst_3 = memref.alloca() : memref<1x16x8x32x8x16x8x32xi32>
    hivm.hir.vreduce <max> ins(%src_3 : memref<16x16x8x32x8x16x8x32xi32>)
                           outs(%dst_3 : memref<1x16x8x32x8x16x8x32xi32>)
                           unsigned_src = false
                           reduce_dims = [0]

    // CHECK-COUNT-6: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @reduce_min_ar_int32_t({{.*}}, %[[CST2:.*]])
    %src_4 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi32>
    %dst_4 = memref.alloca() : memref<16x16x8x32x8x16x8x1xi32>
    hivm.hir.vreduce <min> ins(%src_4 : memref<16x16x8x32x8x16x8x32xi32>)
                           outs(%dst_4 : memref<16x16x8x32x8x16x8x1xi32>)
                           unsigned_src = false
                           reduce_dims = [7]

    return
  }
}

// -----

module {
  func.func @test_reduce_op_check_with_rank_reducing_dynamic_shape(
                         %src_5 : memref<16x?x?x?x8x?x8x32xf16>,
                         %dst_5 : memref<1x?x?x?x8x?x8x32xf16>) {
    // CHECK锛?[[CST:.*]] = arith.constant 0xFC00 : f16
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @reduce_max_ra0a1_half({{.*}}, %[[CST:.*]])
    hivm.hir.vreduce <max> ins(%src_5 : memref<16x?x?x?x8x?x8x32xf16>)
                           outs(%dst_5 : memref<1x?x?x?x8x?x8x32xf16>)
                           unsigned_src = false
                           reduce_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_reduce_op_check_temp_buffer() {
    // CHECK: %[[CST:.*]] = arith.constant -2147483648 : i32
    // CHECK: call @reduce_max_ra0a1_int32_t(%{{.+}}, %{{.+}}, %{{.+}}, %[[CST:.*]])
    %src_3 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi32>
    %dst_3 = memref.alloca() : memref<1x16x8x32x8x16x8x32xi32>
    %tmp_3 = memref.alloca() : memref<2147483648xi32>
    hivm.hir.vreduce <max> ins(%src_3 : memref<16x16x8x32x8x16x8x32xi32>)
                           outs(%dst_3 : memref<1x16x8x32x8x16x8x32xi32>)
                           temp_buffer (%tmp_3 : memref<2147483648xi32>)
                           unsigned_src = false
                           reduce_dims = [0]

    return
  }
}

// -----

module {
  func.func @test_vcmp_op_check_f16_1d() {
    // CHECK: call @vcmp_eq_1d_half
    // CHECK: call @vcmp_ne_1d_half
    // CHECK: call @vcmp_lt_1d_half
    // CHECK: call @vcmp_gt_1d_half
    // CHECK: call @vcmp_ge_1d_half
    // CHECK: call @vcmp_le_1d_half
    %src_0 = memref.alloca() : memref<16xf16>
    %src_1 = memref.alloca() : memref<16xf16>
    %dst = memref.alloca() : memref<16xi1>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <eq>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <ne>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <lt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <gt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <ge>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf16>, memref<16xf16>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <le>
    return
  }
}

// -----

module {
  func.func @test_vcmp_op_check_f16_2d() {
    // CHECK: call @vcmp_eq_1d_half
    // CHECK: call @vcmp_ne_1d_half
    // CHECK: call @vcmp_lt_1d_half
    // CHECK: call @vcmp_gt_1d_half
    // CHECK: call @vcmp_ge_1d_half
    // CHECK: call @vcmp_le_1d_half
    %src_0 = memref.alloca() : memref<5x16xf16>
    %src_1 = memref.alloca() : memref<5x16xf16>
    %dst = memref.alloca() : memref<5x16xi1>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst : memref<5x16xi1>)
                  compare_mode = <eq>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst : memref<5x16xi1>)
                  compare_mode = <ne>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst : memref<5x16xi1>)
                  compare_mode = <lt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst : memref<5x16xi1>)
                  compare_mode = <gt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst : memref<5x16xi1>)
                  compare_mode = <ge>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst : memref<5x16xi1>)
                  compare_mode = <le>
    return
  }
}

// -----

module {
  func.func @test_vcmp_op_check_f16_3d() {
    // CHECK: call @vcmp_eq_1d_half
    // CHECK: call @vcmp_ne_1d_half
    // CHECK: call @vcmp_lt_1d_half
    // CHECK: call @vcmp_gt_1d_half
    // CHECK: call @vcmp_ge_1d_half
    // CHECK: call @vcmp_le_1d_half
    %src_0 = memref.alloca() : memref<5x5x16xf16>
    %src_1 = memref.alloca() : memref<5x5x16xf16>
    %dst = memref.alloca() : memref<5x5x16xi1>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x5x16xf16>, memref<5x5x16xf16>)
                  outs(%dst : memref<5x5x16xi1>)
                  compare_mode = <eq>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x5x16xf16>, memref<5x5x16xf16>)
                  outs(%dst : memref<5x5x16xi1>)
                  compare_mode = <ne>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x5x16xf16>, memref<5x5x16xf16>)
                  outs(%dst : memref<5x5x16xi1>)
                  compare_mode = <lt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x5x16xf16>, memref<5x5x16xf16>)
                  outs(%dst : memref<5x5x16xi1>)
                  compare_mode = <gt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x5x16xf16>, memref<5x5x16xf16>)
                  outs(%dst : memref<5x5x16xi1>)
                  compare_mode = <ge>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<5x5x16xf16>, memref<5x5x16xf16>)
                  outs(%dst : memref<5x5x16xi1>)
                  compare_mode = <le>
    return
  }
}

// -----

module {
  func.func @test_vcmp_op_check_f32_1d() {
    // CHECK: call @vcmp_eq_1d_float
    // CHECK: call @vcmp_ne_1d_float
    // CHECK: call @vcmp_lt_1d_float
    // CHECK: call @vcmp_gt_1d_float
    // CHECK: call @vcmp_ge_1d_float
    // CHECK: call @vcmp_le_1d_float
    %src_0 = memref.alloca() : memref<16xf32>
    %src_1 = memref.alloca() : memref<16xf32>
    %dst = memref.alloca() : memref<16xi1>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf32>, memref<16xf32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <eq>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf32>, memref<16xf32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <ne>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf32>, memref<16xf32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <lt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf32>, memref<16xf32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <gt>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf32>, memref<16xf32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <ge>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xf32>, memref<16xf32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <le>
    return
  }
}

// -----

module {
  func.func @test_vcmp_op_check_i32() {
    // CHECK: call @vcmp_eq_1d_int32_t
    %src_0 = memref.alloca() : memref<16xi32>
    %src_1 = memref.alloca() : memref<16xi32>
    %dst = memref.alloca() : memref<16xi1>

    hivm.hir.vcmp ins(%src_0, %src_1 : memref<16xi32>, memref<16xi32>)
                  outs(%dst : memref<16xi1>)
                  compare_mode = <eq>
    return
  }
}

// -----

module {
  func.func @test_vsel_op_check_1d() {
    // CHECK: call @vsel_vv_1d_bool_half
    // CHECK: call @vsel_vv_1d_bool_int16_t
    // CHECK: call @vsel_vv_1d_bool_uint16_t
    // CHECK: call @vsel_vv_1d_bool_float
    // CHECK: call @vsel_vv_1d_bool_int32_t
    // CHECK: call @vsel_vv_1d_bool_uint32_t
    %src_0 = memref.alloca() : memref<16xi1>
    %src_1 = memref.alloca() : memref<16xf16>
    %src_2 = memref.alloca() : memref<16xf16>
    %dst0 = memref.alloca() : memref<16xf16>
    hivm.hir.vsel ins(%src_0, %src_1, %src_2 : memref<16xi1>, memref<16xf16>, memref<16xf16>)
                  outs(%dst0 : memref<16xf16>)

    %src_3 = memref.alloca() : memref<16xi1>
    %src_4 = memref.alloca() : memref<16xi16>
    %src_5 = memref.alloca() : memref<16xi16>
    %dst1 = memref.alloca() : memref<16xi16>
    hivm.hir.vsel ins(%src_3, %src_4, %src_5 : memref<16xi1>, memref<16xi16>, memref<16xi16>)
                  outs(%dst1 : memref<16xi16>)

    %src_6 = memref.alloca() : memref<16xi1>
    %src_7 = memref.alloca() : memref<16xui16>
    %src_8 = memref.alloca() : memref<16xui16>
    %dst2 = memref.alloca() : memref<16xui16>
    hivm.hir.vsel ins(%src_6, %src_7, %src_8 : memref<16xi1>, memref<16xui16>, memref<16xui16>)
                  outs(%dst2 : memref<16xui16>)

    %src_9 = memref.alloca() : memref<16xi1>
    %src_10 = memref.alloca() : memref<16xf32>
    %src_11 = memref.alloca() : memref<16xf32>
    %dst3 = memref.alloca() : memref<16xf32>
    hivm.hir.vsel ins(%src_9, %src_10, %src_11 : memref<16xi1>, memref<16xf32>, memref<16xf32>)
                  outs(%dst3 : memref<16xf32>)

    %src_12 = memref.alloca() : memref<16xi1>
    %src_13 = memref.alloca() : memref<16xi32>
    %src_14 = memref.alloca() : memref<16xi32>
    %dst4 = memref.alloca() : memref<16xi32>
    hivm.hir.vsel ins(%src_12, %src_13, %src_14 : memref<16xi1>, memref<16xi32>, memref<16xi32>)
                  outs(%dst4 : memref<16xi32>)

    %src_15 = memref.alloca() : memref<16xi1>
    %src_16 = memref.alloca() : memref<16xui32>
    %src_17 = memref.alloca() : memref<16xui32>
    %dst5 = memref.alloca() : memref<16xui32>
    hivm.hir.vsel ins(%src_15, %src_16, %src_17 : memref<16xi1>, memref<16xui32>, memref<16xui32>)
                  outs(%dst5 : memref<16xui32>)

    return
  }
}

// -----

module {
  func.func @test_vsel_op_check_2d() {
    // CHECK: call @vsel_vv_1d_bool_half
    // CHECK: call @vsel_vv_1d_bool_int16_t
    // CHECK: call @vsel_vv_1d_bool_uint16_t
    // CHECK: call @vsel_vv_1d_bool_float
    // CHECK: call @vsel_vv_1d_bool_int32_t
    // CHECK: call @vsel_vv_1d_bool_uint32_t
    %src_0 = memref.alloca() : memref<5x16xi1>
    %src_1 = memref.alloca() : memref<5x16xf16>
    %src_2 = memref.alloca() : memref<5x16xf16>
    %dst0 = memref.alloca() : memref<5x16xf16>
    hivm.hir.vsel ins(%src_0, %src_1, %src_2 : memref<5x16xi1>, memref<5x16xf16>, memref<5x16xf16>)
                  outs(%dst0 : memref<5x16xf16>)

    %src_3 = memref.alloca() : memref<5x16xi1>
    %src_4 = memref.alloca() : memref<5x16xi16>
    %src_5 = memref.alloca() : memref<5x16xi16>
    %dst1 = memref.alloca() : memref<5x16xi16>
    hivm.hir.vsel ins(%src_3, %src_4, %src_5 : memref<5x16xi1>, memref<5x16xi16>, memref<5x16xi16>)
                  outs(%dst1 : memref<5x16xi16>)

    %src_6 = memref.alloca() : memref<5x16xi1>
    %src_7 = memref.alloca() : memref<5x16xui16>
    %src_8 = memref.alloca() : memref<5x16xui16>
    %dst2 = memref.alloca() : memref<5x16xui16>
    hivm.hir.vsel ins(%src_6, %src_7, %src_8 : memref<5x16xi1>, memref<5x16xui16>, memref<5x16xui16>)
                  outs(%dst2 : memref<5x16xui16>)

    %src_9 = memref.alloca() : memref<5x16xi1>
    %src_10 = memref.alloca() : memref<5x16xf32>
    %src_11 = memref.alloca() : memref<5x16xf32>
    %dst3 = memref.alloca() : memref<5x16xf32>
    hivm.hir.vsel ins(%src_9, %src_10, %src_11 : memref<5x16xi1>, memref<5x16xf32>, memref<5x16xf32>)
                  outs(%dst3 : memref<5x16xf32>)

    %src_12 = memref.alloca() : memref<5x16xi1>
    %src_13 = memref.alloca() : memref<5x16xi32>
    %src_14 = memref.alloca() : memref<5x16xi32>
    %dst4 = memref.alloca() : memref<5x16xi32>
    hivm.hir.vsel ins(%src_12, %src_13, %src_14 : memref<5x16xi1>, memref<5x16xi32>, memref<5x16xi32>)
                  outs(%dst4 : memref<5x16xi32>)

    %src_15 = memref.alloca() : memref<5x16xi1>
    %src_16 = memref.alloca() : memref<5x16xui32>
    %src_17 = memref.alloca() : memref<5x16xui32>
    %dst5 = memref.alloca() : memref<5x16xui32>
    hivm.hir.vsel ins(%src_15, %src_16, %src_17 : memref<5x16xi1>, memref<5x16xui32>, memref<5x16xui32>)
                  outs(%dst5 : memref<5x16xui32>)

    return
  }
}

// -----
module {
  func.func @test_reduce_min_with_index() {
    // CHECK: %[[CST:.*]] = arith.constant 0x7F800000 : f32
    // CHECK: call @reduce_min_with_index_left_ar_float({{.*}}, %[[CST:.*]])
    %src = memref.alloc() : memref<1024x8xf32, #hivm.address_space<ub>>
    %ubC = memref.alloc() : memref<1024x1xf32, #hivm.address_space<ub>>
    %ubC_index = memref.alloc() : memref<1024x1xi32, #hivm.address_space<ub>>
    hivm.hir.vreduce <min_with_index>
                     ins(%src : memref<1024x8xf32, #hivm.address_space<ub>>)
                     outs(%ubC, %ubC_index : memref<1024x1xf32, #hivm.address_space<ub>>, memref<1024x1xi32, #hivm.address_space<ub>>)
                     unsigned_src = false
                     tie_break_left = true
                     reduce_dims = [1]
    return
  }
}

// -----
module {
  func.func @test_reduce_min_with_index() {
    // CHECK: call @reduce_min_with_index_right_ar_float
    %src = memref.alloc() : memref<1024x8xf32, #hivm.address_space<ub>>
    %ubC = memref.alloc() : memref<1024x1xf32, #hivm.address_space<ub>>
    %ubC_index = memref.alloc() : memref<1024x1xi32, #hivm.address_space<ub>>
    hivm.hir.vreduce <min_with_index>
                     ins(%src : memref<1024x8xf32, #hivm.address_space<ub>>)
                     outs(%ubC, %ubC_index : memref<1024x1xf32, #hivm.address_space<ub>>, memref<1024x1xi32, #hivm.address_space<ub>>)
                     unsigned_src = false
                     tie_break_left = false
                     reduce_dims = [1]
    return
  }
}

// -----
module {
  func.func @test_reduce_sum_mid_axis() {
    // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: call @reduce_sum_ra_float({{.*}}, %[[CST:.*]])
    %src = memref.alloc() : memref<5x16x16xf32, #hivm.address_space<ub>>
    %ubC = memref.alloc() : memref<5x1x16xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum>
                     ins(%src : memref<5x16x16xf32, #hivm.address_space<ub>>)
                     outs(%ubC : memref<5x1x16xf32, #hivm.address_space<ub>>)
                     unsigned_src = false
                     reduce_dims = [1]
    return
  }
}

// -----
module {
  func.func @test_arange_1d() {
    // CHECK: call @arange_1d_int32_t
    %ub = memref.alloc() : memref<256xi32>
    %c1 = arith.constant 1 : index
    hivm.hir.varange strides[%c1] outs(%ub : memref<256xi32>)
    return
  }
}

// -----
module {
  func.func @test_arange_2d() {
    // CHECK: call @arange_2d_int32_t(
    // CHECK-SAME:                memref<?x?xi32, strided<[?, ?], offset: ?>>,
    // CHECK-SAME:                index,
    // CHECK-SAME:                index) -> ()
    %ub = memref.alloc() : memref<16x16xi32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    hivm.hir.varange strides[%c1, %c2] outs(%ub : memref<16x16xi32>)
    return
  }
}

// -----
module {
  func.func @test_arange_2d() {
    // CHECK: call @arange_2d_int32_t
    %ub = memref.alloc() : memref<16x16xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    hivm.hir.varange offset[%c0] strides[%c1, %c2] outs(%ub : memref<16x16xi32>)
    return
  }
}

// -----
module {
  func.func @test_arange_4d(%dst : memref<16x16x16x16xi32>) {
    // CHECK: scf.for
    // CHECK: call @arange_3d_int32_t
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    hivm.hir.varange offset[%c0] strides[%c1, %c1, %c1, %c1] outs(%dst : memref<16x16x16x16xi32>)
    return
  }
}

// -----
module {
  func.func @test_vbrc_scalar_2d() {
    // CHECK: call @broadcast_scalar_float_to_2d
    %cst  = arith.constant 1.0 : f32
    %ub = memref.alloc() : memref<168x6xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%ub : memref<168x6xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    return
  }
}


// -----
module {
  func.func @test_vbrc_last_axis_b64() {
    // CHECK: call @broadcast_last_axis_align_2d_int64_t
    %cst  = arith.constant 1.0 : f32
    %in = memref.alloc() : memref<64x1xi64, #hivm.address_space<ub>>
    %out = memref.alloc() : memref<64x64xi64, #hivm.address_space<ub>>
    %temp = memref.alloc() : memref<0xi64, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%in : memref<64x1xi64, #hivm.address_space<ub>>)
                  outs(%out : memref<64x64xi64, #hivm.address_space<ub>>)
                  temp_buffer(%temp : memref<0xi64, #hivm.address_space<ub>>) broadcast_dims = [1]
    return
  }
}

// -----
module {
  func.func @test_vbrc_first_axis_align_b64() {
    // CHECK: call @broadcast_first_axis_align_2d_int64_t
    %in = memref.alloc() : memref<1x64xi64, #hivm.address_space<ub>>
    %out = memref.alloc() : memref<64x64xi64, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%in : memref<1x64xi64, #hivm.address_space<ub>>)
                  outs(%out : memref<64x64xi64, #hivm.address_space<ub>>) broadcast_dims = [0]
    return
  }
}

// -----
module {
  func.func @test_vbrc_first_axis_unalign_b64() {
    // CHECK: call @broadcast_first_axis_unalign_2d_int64_t
    %in = memref.alloc() : memref<1x61xi64, #hivm.address_space<ub>>
    %out = memref.alloc() : memref<64x61xi64, #hivm.address_space<ub>>
    %temp = memref.alloc() : memref<4096xi64, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%in : memref<1x61xi64, #hivm.address_space<ub>>)
                  outs(%out : memref<64x61xi64, #hivm.address_space<ub>>)
                  temp_buffer(%temp : memref<4096xi64, #hivm.address_space<ub>>) broadcast_dims = [0]
    return
  }
}

// -----
module {
  // CHECK: llvm.mlir.global private constant @_debug_prefix_0
  // CHECK-LABEL: func.func private @print_1d_float_gm
  // CHECK: func @test_print(%[[ARG_0:.*]]: memref<1024xf32, #hivm.address_space<gm>>)
  func.func @test_print(%arg0: memref<1024xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[C1_I8:.*]] = arith.constant 1 : i8
    // CHECK: %[[C2:.*]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK: call @_mlir_ciface_init_debug
    // CHECK: %[[GLOBAL:.*]] = llvm.getelementptr
    // CHECK: %[[CAST:.*]] = memref.cast %[[ARG_0]]
    // CHECK: call @print_1d_float_gm(%[[GLOBAL]], %[[C2]], %[[CAST]], %[[C1_I8]])
    // CHECK: call @_mlir_ciface_finish_debug
    hivm.hir.init_debug
    hivm.hir.debug {debugtype = "print", finishInserted = 0 : i32, hex = true, memscope = #hivm.address_space<gm>, prefix = " VAL =: ", tcoretype = #hivm.tcore_type<VECTOR>} %arg0 : memref<1024xf32, #hivm.address_space<gm>>
    hivm.hir.finish_debug
    return
  }
}

// -----
module {
  func.func @test_vinterleave_op() {
  // CHECK: scf.for
  // CHECK: call @interleave_1d_half
    %alloc = memref.alloc() : memref<2x16xf16>
    %alloc_0 = memref.alloc() : memref<2x16xf16>
    %alloc_1 = memref.alloc() : memref<2x32xf16>
    %alloc_2 = memref.alloc() : memref<160xf16>
    hivm.hir.vinterleave ins(%alloc, %alloc_0 : memref<2x16xf16>, memref<2x16xf16>)
                         outs(%alloc_1 : memref<2x32xf16>)
                         interleave_channel_nums = 2
                         temp_buffer(%alloc_2 : memref<160xf16>)
    return
  }
}


// -----
module {
  func.func @test_deinterleave_2d() {
    // CHECK: scf.for
    // CHECK: call @deinterleave_channel_0_from_2_channels_1d_half
    // CHECK: scf.for
    // CHECK: call @deinterleave_channel_1_from_2_channels_1d_half
    %input_f16 = memref.alloc() : memref<2x32xf16>
    %output_even_f16 = memref.alloc() : memref<2x16xf16>
    %output_odd_f16 = memref.alloc() : memref<2x16xf16>
    hivm.hir.vdeinterleave ins(%input_f16 : memref<2x32xf16>)
                           outs(%output_even_f16 : memref<2x16xf16>)
                           channel_num = 2
                           index_mode = <CHANNEL_0>
    hivm.hir.vdeinterleave ins(%input_f16 : memref<2x32xf16>)
                           outs(%output_odd_f16 : memref<2x16xf16>)
                           channel_num = 2
                           index_mode = <CHANNEL_1>
    return
  }

  func.func @test_deinterleave_1d() {
    // CHECK: call @deinterleave_channel_0_from_2_channels_1d_half
    %input_f16 = memref.alloc() : memref<32xf16>
    %output_even_f16 = memref.alloc() : memref<16xf16>
    hivm.hir.vdeinterleave ins(%input_f16 : memref<32xf16>)
                           outs(%output_even_f16 : memref<16xf16>)
                           channel_num = 2
                           index_mode = <CHANNEL_0>
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @test_deinterleave_n_to_1_2d_case0
  // CHECK: call @deinterleave_channel_0_from_n_channels_2d_half
  func.func @test_deinterleave_n_to_1_2d_case0() {
    %input = memref.alloc() : memref<4x64xf16, strided<[1024, 16]>>
    %output = memref.alloc() : memref<4x64xf16, strided<[64, 1]>>
    hivm.hir.vdeinterleave ins(%input : memref<4x64xf16, strided<[1024, 16]>>)
                            outs(%output : memref<4x64xf16, strided<[64, 1]>>)
                            channel_num = 16
                            index_mode = <CHANNEL_0>
    return
  }

  // CHECK-LABEL: func.func @test_deinterleave_n_to_1_2d_case1
  // CHECK: call @deinterleave_channel_0_from_n_channels_2d_half
  func.func @test_deinterleave_n_to_1_2d_case1() {
    %input = memref.alloc() : memref<4x64xf16, strided<[64, 1]>>
    %output = memref.alloc() : memref<4x4xf16, strided<[4, 1]>>
    hivm.hir.vdeinterleave ins(%input : memref<4x64xf16, strided<[64, 1]>>)
                          outs(%output : memref<4x4xf16, strided<[4, 1]>>)
                          channel_num = 16
                          index_mode = <CHANNEL_0>
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_atomic_add
  // CHECK: %[[ATOMICKIND:.*]] = arith.constant 1 : i32
  // CHECK: @store_ubuf_to_gm_1d_float(%[[UB_MEMREF:.*]], %[[GM_MEMREF:.*]], %[[ATOMICKIND:.*]])
  func.func @test_atomic_add(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32, #hivm.address_space<ub>> {tt.divisibility = 16 : i32}) {
    %c0_i64 = arith.constant 0 : i64
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: 0>, #hivm.address_space<gm>>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.store ins(%0 : memref<256xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<256xf32, strided<[1], offset: 0>, #hivm.address_space<gm>>) atomic = <add>
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_atomic_add
  // CHECK: %[[ATOMICKIND:.*]] = arith.constant 1 : i32
  // CHECK: @store_ubuf_to_gm_1d_float(%[[UB_MEMREF:.*]], %[[GM_MEMREF:.*]], %[[ATOMICKIND:.*]])
  func.func @test_atomic_add(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xf32, #hivm.address_space<ub>> {tt.divisibility = 16 : i32}) {
    %c0_i64 = arith.constant 0 : i64
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: 0>, #hivm.address_space<gm>>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.store ins(%0 : memref<256xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<256xf32, strided<[1], offset: 0>, #hivm.address_space<gm>>) atomic = <add>
    return
  }
}

// -----
module {
    func.func @test_vflip_op_memref() {
    // CHECK: call @flip_1d_half
      %src = memref.alloc() : memref<2x16xf16>
      %dst = memref.alloc() : memref<2x16xf16>
      hivm.hir.vflip ins(%src : memref<2x16xf16>)
                    outs(%dst : memref<2x16xf16>)
                    flip_axis = 1
      return
    }
}

module {
  func.func @test_vsel() {
    // CHECK: scf.for
    // CHECK: call @vsel_vv_1d_bool_half
    %alloca = memref.alloca() : memref<8x16xi1>
    %alloca_0 = memref.alloca() : memref<8x16xf16>
    %alloca_1 = memref.alloca() : memref<8x16xf16>
    %alloca_2 = memref.alloca() : memref<8x16xf16>
    %alloc = memref.alloc() : memref<8xf16>
    hivm.hir.vsel ins(%alloca, %alloca_0, %alloca_1 : memref<8x16xi1>, memref<8x16xf16>, memref<8x16xf16>) outs(%alloca_2 : memref<8x16xf16>) temp_buffer(%alloc : memref<8xf16>)
    return
  }
}

module {
  func.func @test_vxor_1d() {
    // CHECK: call @vxor_1d_int16_t
    %alloca_0 = memref.alloca() : memref<16xi16>
    %alloca_1 = memref.alloca() : memref<16xi16>
    %alloca_2 = memref.alloca() : memref<16xi16>
    %alloc = memref.alloc() : memref<16xi16>
    hivm.hir.vxor ins(%alloca_0, %alloca_1 : memref<16xi16>, memref<16xi16>) outs(%alloca_2 : memref<16xi16>) temp_buffer(%alloc : memref<16xi16>)
    return
  }
}

module {
  func.func @test_vxor_2d() {
    // CHECK: call @vxor_2d_int16_t
    %alloca_0 = memref.alloca() : memref<8x16xi16>
    %alloca_1 = memref.alloca() : memref<8x16xi16>
    %alloca_2 = memref.alloca() : memref<8x16xi16>
    %alloc = memref.alloc() : memref<128xi16>
    hivm.hir.vxor ins(%alloca_0, %alloca_1 : memref<8x16xi16>, memref<8x16xi16>) outs(%alloca_2 : memref<8x16xi16>) temp_buffer(%alloc : memref<128xi16>)
    return
  }
}

// -----
module {
  // CHECK: call @mul_extended_1d_int16_t
  func.func @test_vmulextended_op() {
    %input_0 = memref.alloc() : memref<32xi16>
    %input_1 = memref.alloc() : memref<32xi16>
    %output_0 = memref.alloc() : memref<32xi16>
    %output_1 = memref.alloc() : memref<32xi16>
    %alloc = memref.alloc() : memref<96xi32>
    hivm.hir.vmulextended ins(%input_0, %input_1 : memref<32xi16>, memref<32xi16>)
                          outs(%output_0, %output_1 : memref<32xi16>, memref<32xi16>)
                          temp_buffer(%alloc : memref<96xi32>)
    return
  }
}

// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>} {
  // CHECK-LABEL: @matmul_Xbias_Xdescale_XtransposeA_transposeB_TAhalf_TBhalf_TChalf_TTint64_t
  // CHECK: hacc.always_inline
  func.func @matmul_tb(%arg0: memref<?x?xf16, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<?x?xf16, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<?x?xf16, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<15xi64, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.block_dim = 20 : i64, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    hivm.hir.set_mask_norm
    hivm.hir.matmul ins(%arg0, %arg1 : memref<?x?xf16, #hivm.address_space<gm>>, memref<?x?xf16, #hivm.address_space<gm>>) outs(%arg2 : memref<?x?xf16, #hivm.address_space<gm>>) tiling_params = %arg3 : memref<15xi64, #hivm.address_space<gm>> b_transpose
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// -----
module {
  // CHECK: call @vsel_vv_1d_bool_int64_t
  func.func @test_vsel_op_i1_i64() {
    %cond = memref.alloc() : memref<1024xi1>
    %src0 = memref.alloc() : memref<1024xi64>
    %src1 = memref.alloc() : memref<1024xi64>
    %dst = memref.alloc() : memref<1024xi64>
    %tmp_buf = memref.alloc() : memref<12xi64>
    hivm.hir.vsel ins(%cond, %src0, %src1 : memref<1024xi1>, memref<1024xi64>, memref<1024xi64>)
                  outs(%dst : memref<1024xi64>) temp_buffer(%tmp_buf : memref<12xi64>)
    return
  }
}

// -----
module {
  // CHECK: call @vsel_vv_1d_int8_t_int64_t
  func.func @test_vsel_op_i8_i64() {
    %cond = memref.alloc() : memref<1024xi8>
    %src0 = memref.alloc() : memref<1024xi64>
    %src1 = memref.alloc() : memref<1024xi64>
    %dst = memref.alloc() : memref<1024xi64>
    %tmp_buf = memref.alloc() : memref<768xi64>
    hivm.hir.vsel ins(%cond, %src0, %src1 : memref<1024xi8>, memref<1024xi64>, memref<1024xi64>)
                  outs(%dst : memref<1024xi64>) temp_buffer(%tmp_buf : memref<768xi64>)
    return
  }
}

// -----
module {
  // CHECK: call @broadcast_first_axis_unalign_3d_float
  func.func @test_vbrc_mid_3d_float() {
    %src = memref.alloc() : memref<4x1x2x20xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<4x16x2x20xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%src : memref<4x1x2x20xf32, #hivm.address_space<ub>>)
                  outs(%dst : memref<4x16x2x20xf32, #hivm.address_space<ub>>)
                  broadcast_dims = [1]
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_vsel_op_int64_ss_1d
  // CHECK: call @vsel_ss_1d_int64_t
  func.func @test_vsel_op_int64_ss_1d() {
    %cond = memref.alloc() : memref<32xi1>
    %c5_i64 = arith.constant 5 : i64
    %c-2_i64 = arith.constant -2 : i64
    %dst = memref.alloc() : memref<32xi64>
    hivm.hir.vsel ins(%cond, %c5_i64, %c-2_i64 : memref<32xi1>, i64, i64)
                  outs(%dst : memref<32xi64>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_vsel_op_int64_ss_2d
  // CHECK: scf.for
  // CHECK: call @vsel_ss_1d_int64_t
  func.func @test_vsel_op_int64_ss_2d() {
    %cond = memref.alloc() : memref<7x32xi1>
    %c5_i64 = arith.constant 5 : i64
    %c-2_i64 = arith.constant -2 : i64
    %dst = memref.alloc() : memref<7x32xi64>
    hivm.hir.vsel ins(%cond, %c5_i64, %c-2_i64 : memref<7x32xi1>, i64, i64)
                  outs(%dst : memref<7x32xi64>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_cumsum_op_i32_2d_dim1
  // CHECK: call @cumsum_2d_int32_t_dim1
  func.func @test_cumsum_op_i32_2d_dim1(%arg0: memref<5x16xi32>, %arg1: memref<5x16xi32>) attributes {hacc.entry} {
    hivm.hir.vcumsum ins(%arg0 : memref<5x16xi32>) outs(%arg1 : memref<5x16xi32>) cum_dims = [1] reverse = false
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_cumprod_op_i32_2d_dim1
  // CHECK: call @cumprod_2d_int32_t_dim1
  func.func @test_cumprod_op_i32_2d_dim1(%arg0: memref<5x16xi32>, %arg1: memref<5x16xi32>) attributes {hacc.entry} {
    hivm.hir.vcumprod ins(%arg0 : memref<5x16xi32>) outs(%arg1 : memref<5x16xi32>) cum_dims = [1] reverse = false
    return
  }
}

// -----
module {
  func.func @test_vbrc_to_set2dl1_float() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %dst = hivm.hir.pointer_cast(%c0_i64) : memref<32768xf32, #hivm.address_space<cbuf>>
    // CHECK: call @set_l1_2d_float
    hivm.hir.vbrc ins(%cst : f32) outs(%dst : memref<32768xf32, #hivm.address_space<cbuf>>)
    return
  }
}

module {
  // CHECK-LABEL: func @test_vtranspose_3d_with_last_two_axis_1_size
  func.func @test_vtranspose_3d_with_last_two_axis_1_size() attributes {hacc.entry} {
    %src = memref.alloc() : memref<216x1x1xi64>
    %dst = memref.alloc() : memref<216x1x1xi64>
    %tmp = memref.alloc() : memref<512xi64>
    // CHECK: scf.for
    // CHECK: %[[subview_0:.*]] = memref.subview
    // CHECK: %[[subview_1:.*]] = memref.subview
    // CHECK: %[[cast_0:.*]] = memref.cast %[[subview_0:.*]] : memref<1x1xi64, strided<[1, 1], offset: ?>> to memref<?x?xi64, strided<[?, ?], offset: ?>>
    // CHECK: %[[cast_1:.*]] = memref.cast %[[subview_1:.*]] : memref<1x1xi64, strided<[1, 1], offset: ?>> to memref<?x?xi64, strided<[?, ?], offset: ?>>
    // CHECK: @transpose_2d_with_last_axis_int64_t
    hivm.hir.vtranspose ins(%src : memref<216x1x1xi64>)
                        outs(%dst : memref<216x1x1xi64>)
                        temp_buffer(%tmp : memref<512xi64>)
                        permutation = [0, 2, 1]
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @test_sync_block_lock
  func.func @test_sync_block_lock() attributes {hacc.entry} {
    %lock = memref.alloc() : memref<1xi64>
    // CHECK: @sync_block_lock
    hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    // CHECK: @sync_block_unlock
    hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
    return
  }
}

// -----
module {
  func.func @test_sort_op_check_1d() {
    // CHECK: @sort_1d_float
    // CHECK: @sort_with_index_1d_float
    // CHECK: @sort_1d_half
    // CHECK: @sort_with_index_1d_half
    // CHECK: @sort_1d_int32_t
    // CHECK: @sort_1d_int64_t
    %src_f32 = memref.alloca() : memref<32xf32>
    %dst_value_f32 = memref.alloca() : memref<32xf32>
    %dst_index = memref.alloca() : memref<32xi32>
    hivm.hir.vsort ins(%src_f32 : memref<32xf32>)
                   outs(%dst_value_f32 : memref<32xf32>)
                   descending = true
                   sort_axis = 0
    hivm.hir.vsort ins(%src_f32 : memref<32xf32>)
                   outs(%dst_value_f32, %dst_index : memref<32xf32>, memref<32xi32>)
                   descending = true
                   sort_axis = 0

    %src_f16 = memref.alloca() : memref<32xf16>
    %dst_value_f16 = memref.alloca() : memref<32xf16>
    hivm.hir.vsort ins(%src_f16 : memref<32xf16>)
                   outs(%dst_value_f16 : memref<32xf16>)
                   descending = false
                   sort_axis = 0
    hivm.hir.vsort ins(%src_f16 : memref<32xf16>)
                   outs(%dst_value_f16, %dst_index : memref<32xf16>, memref<32xi32>)
                   descending = false
                   sort_axis = 0

    %src_i32 = memref.alloca() : memref<32xi32>
    %dst_value_i32 = memref.alloca() : memref<32xi32>
    hivm.hir.vsort ins(%src_i32 : memref<32xi32>)
                   outs(%dst_value_i32 : memref<32xi32>)
                   descending = false
                   sort_axis = 0

    %src_i64 = memref.alloca() : memref<32xi64>
    %dst_value_i64 = memref.alloca() : memref<32xi64>
    hivm.hir.vsort ins(%src_i64 : memref<32xi64>)
                   outs(%dst_value_i64 : memref<32xi64>)
                   descending = false
                   sort_axis = 0
    return
  }
}

// -----
module {
  func.func @test_sort_op_check_2d() {
    // CHECK: @sort_2d_float
    // CHECK: @sort_with_index_2d_float
    // CHECK: @sort_2d_half
    // CHECK: @sort_with_index_2d_half
    %src_f32 = memref.alloca() : memref<8x32xf32>
    %dst_value_f32 = memref.alloca() : memref<8x32xf32>
    %dst_index = memref.alloca() : memref<8x32xi32>
    hivm.hir.vsort ins(%src_f32 : memref<8x32xf32>)
                   outs(%dst_value_f32 : memref<8x32xf32>)
                   descending = true
                   sort_axis = 1
    hivm.hir.vsort ins(%src_f32 : memref<8x32xf32>)
                   outs(%dst_value_f32, %dst_index : memref<8x32xf32>, memref<8x32xi32>)
                   descending = true
                   sort_axis = 1

    %src_f16 = memref.alloca() : memref<8x32xf16>
    %dst_value_f16 = memref.alloca() : memref<8x32xf16>
    hivm.hir.vsort ins(%src_f16 : memref<8x32xf16>)
                   outs(%dst_value_f16 : memref<8x32xf16>)
                   descending = false
                   sort_axis = 1
    hivm.hir.vsort ins(%src_f16 : memref<8x32xf16>)
                   outs(%dst_value_f16, %dst_index : memref<8x32xf16>, memref<8x32xi32>)
                   descending = false
                   sort_axis = 1
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @embeddinggather_test
  // CHECK: func.func private @embedding_gather_2d_float_int64_t
  // CHECK: call @embedding_gather_2d_float_int64_t
  func.func @embeddinggather_test(%arg0 : memref<?xf32, #hivm.address_space<gm>>) attributes {DirectlyUsedGMArgIdxList = [0]} {
    %c0_i64 = arith.constant 0 : i64
    %c3_i64 = arith.constant 3 : i64
    %c38_i64 = arith.constant 38 : i64
    %c128_i64 = arith.constant 128 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c81913_i64 = arith.constant 81913 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<3x40x4x1xi64, #hivm.address_space<ub>>
    %subview = memref.subview %0[0, 0, 0, 0] [3, 38, 1, 1] [1, 1, 1, 1] : memref<3x40x4x1xi64, #hivm.address_space<ub>> to memref<3x38x1xi64, strided<[160, 4, 1]>, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %subview [[0], [1, 2]] : memref<3x38x1xi64, strided<[160, 4, 1]>, #hivm.address_space<ub>> into memref<3x38xi64, strided<[160, 4]>, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c10240_i64) : memref<3x38x128xf32, #hivm.address_space<ub>>
    hivm.hir.embedding_gather ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>, %collapse_shape : memref<3x38xi64, strided<[160, 4]>, #hivm.address_space<ub>>, %c81913_i64 : i64, [%c0_i64, %c0_i64, %c0_i64 : i64, i64, i64], [%c3_i64, %c38_i64, %c128_i64 : i64, i64, i64]) outs(%1 : memref<3x38x128xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: test_nz2nd
  func.func @test_nz2nd() {
    %gmA = memref.alloc() : memref<1024x2048xf16, #hivm.address_space<gm>>
    %gmASubview = memref.subview %gmA[0, 0][256, 128][1, 1]
                         : memref<1024x2048xf16, #hivm.address_space<gm>> to
                           memref<256x128xf16, strided<[2048, 1], offset: 0>, #hivm.address_space<gm>>
    %l1A = memref.alloc() : memref<256x128xf16, #hivm.address_space<cbuf>>
    // CHECK: call @nz2nd_2d_to_2d_half
    hivm.hir.nz2nd
      ins(%l1A : memref<256x128xf16, #hivm.address_space<cbuf>>)
      outs(%gmASubview : memref<256x128xf16, strided<[2048, 1], offset: 0>, #hivm.address_space<gm>>)
    return
 }
}

// -----
// CHECK-LABEL: test_matmul_invalid_input
module {
  func.func @test_matmul_invalid_input() {
    %ma = memref.alloc() : memref<16x16xf16>
    %mb = memref.alloc() : memref<16x16xf16>
    %bias = memref.alloc() : memref<16xf16>
    %mc = memref.alloc() : memref<16x16xf32>
    %c16 = arith.constant 16 : index
    %init = arith.constant 1 : i1
    // CHECK: call @mma_tile_half_to_float_hf32
    hivm.hir.mmadL1{enable_HF32} ins(%ma, %mb, %init, %c16, %c16, %c16: memref<16x16xf16>, memref<16x16xf16>, i1, index, index, index)
                         outs(%mc : memref<16x16xf32>)
    return
  }
}

// -----
// CHECK-LABEL: func @test_indirect_load
module {
  func.func @test_indirect_load(%arg0_f: memref<?xf32>, %arg0_i: memref<?xi32>, %arg1_i32: memref<4x32xi32>, %arg1_i64: memref<4x32xi64>, %arg2: memref<4x32xi1>, %mask_f: memref<4x32xf32>, %mask_i: memref<4x32xi32>) {
    %dst = memref.alloc() : memref<4x32xf32>
    // CHECK: call @indirect_load_2d_float_int32_t
    hivm.hir.indirect_load ins(%arg0_f : memref<?xf32>, %arg1_i32 : memref<4x32xi32>, %arg2 : memref<4x32xi1>, %mask_f : memref<4x32xf32>) outs(%dst : memref<4x32xf32>)
    // CHECK: call @indirect_load_2d_float_int64_t
    hivm.hir.indirect_load ins(%arg0_f : memref<?xf32>, %arg1_i64 : memref<4x32xi64>, %arg2 : memref<4x32xi1>, %mask_f : memref<4x32xf32>) outs(%dst : memref<4x32xf32>)
    %dst_i = memref.alloc() : memref<4x32xi32>
    // CHECK: call @indirect_load_2d_int32_t_int32_t
    hivm.hir.indirect_load ins(%arg0_i : memref<?xi32>, %arg1_i32 : memref<4x32xi32>, %arg2 : memref<4x32xi1>, %mask_i : memref<4x32xi32>) outs(%dst_i : memref<4x32xi32>)
    // CHECK: call @indirect_load_2d_int32_t_int64_t
    hivm.hir.indirect_load ins(%arg0_i : memref<?xi32>, %arg1_i64 : memref<4x32xi64>, %arg2 : memref<4x32xi1>, %mask_i : memref<4x32xi32>) outs(%dst_i : memref<4x32xi32>)
    // CHECK: call @indirect_load_nonvolatile_2d_float_int32_t
    hivm.hir.indirect_load ins(%arg0_f : memref<?xf32>, %arg1_i32 : memref<4x32xi32>, %arg2 : memref<4x32xi1>, %mask_f : memref<4x32xf32>) outs(%dst : memref<4x32xf32>) {isVolatile = false}
    return
  }
}


// -----
// CHECK-LABEL: func @test_stride_load
module {
  func.func @test_stride_load(%src: memref<?xf32>) {
    %offset = arith.constant 0 : i64
    %stride = arith.constant 2 : i64
    %numel = arith.constant 8 : i64
    %other = arith.constant 0.000000e+00 : f32
    %dst = memref.alloc() : memref<8xf32>
    // CHECK: call @stride_load_1d_float_int64_t
    hivm.hir.stride_load
      ins(%src : memref<?xf32>)
      outs(%dst : memref<8xf32>)
      offset(%offset : i64)
      other(%other : f32)
      strides([%stride : i64])
      numels([%numel : i64])
    %offset_i32 = arith.constant 0 : i32
    %stride_i32 = arith.constant 3 : i32
    %numel0_i32 = arith.constant 4 : i32
    %numel1_i32 = arith.constant 8 : i32
    %dst2d = memref.alloc() : memref<4x8xf32>
    // CHECK: call @stride_load_2d_float_int32_t
    hivm.hir.stride_load
      ins(%src : memref<?xf32>)
      outs(%dst2d : memref<4x8xf32>)
      offset(%offset_i32 : i32)
      other(%other : f32)
      strides([%stride_i32, %stride_i32 : i32, i32])
      numels([%numel0_i32, %numel1_i32 : i32, i32])
    %numel2_i32 = arith.constant 2 : i32
    %dst3d = memref.alloc() : memref<2x4x8xf32>
    // CHECK: call @stride_load_3d_float_int32_t
    hivm.hir.stride_load
      ins(%src : memref<?xf32>)
      outs(%dst3d : memref<2x4x8xf32>)
      offset(%offset_i32 : i32)
      other(%other : f32)
      strides([%stride_i32, %stride_i32, %stride_i32 : i32, i32, i32])
      numels([%numel2_i32, %numel0_i32, %numel1_i32 : i32, i32, i32])
    return
  }
}


// -----
// CHECK-LABEL: func @test_stride_store
module {
  func.func @test_stride_store(%dst: memref<?xf32>, %src1d: memref<8xf32>, %src2d: memref<4x8xf32>, %src3d: memref<2x4x8xf32>) {
    %offset = arith.constant 0 : i64
    %stride = arith.constant 2 : i64
    %numel = arith.constant 8 : i64
    // CHECK: call @stride_store_1d_float_int64_t
    hivm.hir.stride_store
      ins(%src1d : memref<8xf32>)
      outs(%dst : memref<?xf32>)
      offset(%offset : i64)
      strides([%stride : i64])
      numels([%numel : i64])
    %offset_i32 = arith.constant 0 : i32
    %stride_i32 = arith.constant 3 : i32
    %numel0_i32 = arith.constant 4 : i32
    %numel1_i32 = arith.constant 8 : i32
    // CHECK: call @stride_store_2d_float_int32_t
    hivm.hir.stride_store
      ins(%src2d : memref<4x8xf32>)
      outs(%dst : memref<?xf32>)
      offset(%offset_i32 : i32)
      strides([%stride_i32, %stride_i32 : i32, i32])
      numels([%numel0_i32, %numel1_i32 : i32, i32])
    %numel2_i32 = arith.constant 2 : i32
    // CHECK: call @stride_store_3d_float_int32_t
    hivm.hir.stride_store
      ins(%src3d : memref<2x4x8xf32>)
      outs(%dst : memref<?xf32>)
      offset(%offset_i32 : i32)
      strides([%stride_i32, %stride_i32, %stride_i32 : i32, i32, i32])
      numels([%numel2_i32, %numel0_i32, %numel1_i32 : i32, i32, i32])
    return
  }
}


// -----
// CHECK-LABEL: func @test_indirect_store
module {
  func.func @test_indirect_store(%arg0: memref<?xf32>, %arg1: memref<2x32xi32>, %arg2: memref<4x32xf32>, %arg3: memref<2x32xi8>) attributes {DirectlyUsedGMArgIdxList = [2, 2, 2]} {
    // CHECK: call @indirect_store_no_mask_2d_float_int32_t
    hivm.hir.indirect_store ins(%arg2 : memref<4x32xf32>, %arg1 : memref<2x32xi32>) outs(%arg0 : memref<?xf32>)
    // CHECK: call @indirect_store_2d_float_int32_t
    hivm.hir.indirect_store ins(%arg2 : memref<4x32xf32>, %arg1 : memref<2x32xi32>, %arg3 : memref<2x32xi8>) outs(%arg0 : memref<?xf32>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func @gather_out_to_ub_test
  // CHECK: call @gather_out_to_ub_2d_float_int64_t
  func.func @gather_out_to_ub_test(%arg0: memref<?xf32>, %arg1: memref<2x2xi64>) attributes {DirectlyUsedGMArgIdxList = [0]} {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<0.000000e+00> : memref<2x2xf32>
    hivm.hir.gatherT ins(%arg0 : memref<?xf32>, %arg1 : memref<2x2xi64>, %c4_i64 : i64, %c0_i32 : i32, [%c2_i64, %c1_i64 : i64, i64], [%c2_i32, %c2_i32 : i32, i32], [%c0_i32, %c0_i32 : i32, i32]) outs(%cst : memref<2x2xf32>)
    return
  }
}

// -----
// CHECK: func.func private @index_put_2d_float_int64_t
// CHECK-LABEL: func @index_put_test
// CHECK: call @index_put_2d_float_int64_t
module {
  func.func @index_put_test(%arg0: memref<?xf32>, %arg1: memref<2xi64>, %arg2: memref<2x2xf32>) attributes {DirectlyUsedGMArgIdxList = [0]} {
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    hivm.hir.index_put ins(%arg0 : memref<?xf32>, %arg1 : memref<2xi64>, %arg2 : memref<2x2xf32>, %c0_i32 : i32, %c2_i64 : i64, [%c4_i64, %c2_i64 : i64, i64], [%c0_i64, %c0_i64 : i64, i64], [%c0_i64, %c0_i64 : i64, i64])
    return
  }
}

// -----
// CHECK-LABEL: func @scatterT_test
// CHECK: call @scatter_ub_to_out_2d_float_int32_t
module {
  func.func @scatterT_test(%arg0: memref<4x32xf32>, %arg1: tensor<4x32xf32>, %arg2: tensor<4x32xi32>) attributes {DirectlyUsedGMArgIdxList = [0], hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    hivm.hir.scatterT ins(%arg0 : memref<4x32xf32>, %arg1 : tensor<4x32xf32>, %arg2 : tensor<4x32xi32>, %c32_i32 : i32, %c1_i32 : i32, [%c32_i32, %c1_i32 : i32, i32], [%c4_i32, %c32_i32 : i32, i32], [%c0_i32, %c0_i32 : i32, i32])
    return
  }
}

// -----

// CHECK: func.func @fix_pipe_quant_scale(%[[_:.*]], %[[QUANT_SCALE:.*]]: f32)
// CHECK: %[[QF322F32_PRE:.*]] = arith.constant 15 : i64
// CHECK: call @fixpipe_nz2nd_dual_float_to_float_2d_to_2d_ubuf(%[[_:.*]], %[[QF322F32_PRE]], %[[QUANT_SCALE]]
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @fix_pipe_quant_scale(%arg0: tensor<64x64xf32>, %arg1: memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: f32) {
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>} ins(%arg0 : tensor<64x64xf32>) outs(%arg1 : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) quant_scale = %arg2 : f32 dual_dst_mode = <ROW_SPLIT>
    return
  }
}

// -----

// CHECK: func.func @fix_pipe_quant_scale(%[[_:.*]], %[[QUANT_SCALE:.*]]: f32)
// CHECK: %[[QF322F32_PRE:.*]] = arith.constant 15 : i64
// CHECK: call @fixpipe_nz2nd_dual_float_to_float_2d_to_2d_ubuf(%[[_:.*]], %[[QF322F32_PRE]], %[[QUANT_SCALE]]
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @fix_pipe_quant_scale(%arg0: tensor<64x64xf32>, %arg1: memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, %arg2: f32) {
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>} ins(%arg0 : tensor<64x64xf32>) outs(%arg1 : memref<32x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) quant_scale = %arg2 : f32 dual_dst_mode = <ROW_SPLIT>
    return
  }
}

// -----
module {
  // CHECK-LABEL: @copyop1d_without_eviction_policy
  func.func @copyop1d_without_eviction_policy() {
    %src = memref.alloc() : memref<16xi8, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<16xi8, #hivm.address_space<ub>>

    // CHECK: %[[LEFT_PAD:.*]] = arith.constant 0 : index
    // CHECK: %[[PAD_VALUE:.*]] = arith.constant 0 : i8
    // CHECK: %[[EVICT:.*]] = arith.constant 0 : i32
    // CHECK: call @load_gm_to_ubuf_1d_int8_t(
    // CHECK-SAME: %{{.*}}, %{{.*}}, %[[EVICT]], %[[PAD_VALUE]], %[[LEFT_PAD]], %[[EVICT]]
    hivm.hir.load ins(%src : memref<16xi8, #hivm.address_space<gm>>)
                  outs(%dst : memref<16xi8, #hivm.address_space<ub>>)

    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func @test_fixpipe_nz2nd_ub_sub_block_1
  func.func @test_fixpipe_nz2nd_ub_sub_block_1() {
    %ubC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    //   CHECK-DAG: %[[SB1:.*]] = arith.constant true
    //       CHECK: call @fixpipe_nz2nd_float_to_half_4d_to_2d_ubuf({{.*}}, %[[SB1]]) :
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, sub_block_idx = #hivm.fixpipe_sub_block<sub_block_1>}
      ins(%l0c : memref<2x2x16x16xf32, #hivm.address_space<cc>>)
      outs(%ubC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    return
 }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func @test_fixpipe_nz2nd_ub_sub_block_default
  func.func @test_fixpipe_nz2nd_ub_sub_block_default() {
    %ubC = memref.alloc() : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>
    %l0c = memref.alloc() : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    //   CHECK-NOT: arith.constant true
    //       CHECK: call @fixpipe_nz2nd_float_to_half_4d_to_2d_ubuf(
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%l0c : memref<2x2x16x16xf32, #hivm.address_space<cc>>)
      outs(%ubC : memref<32x32xf16, strided<[32, 1], offset: 0>, #hivm.address_space<ub>>)
    return
 }
}
