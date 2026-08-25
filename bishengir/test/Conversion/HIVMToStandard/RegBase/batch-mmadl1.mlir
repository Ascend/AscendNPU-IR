// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @lower_batch_mmad_l1
  // CHECK: call @batch_mma_tile_half_to_float
  // CHECK-NOT: call @mma_tile_half_to_float
  // CHECK-NOT: scf.for
  // CHECK: call @fixpipe_nz2nd_float_to_half_5d_to_3d_gm
  // CHECK-NOT: hivm.hir.batchMmadL1
  func.func @lower_batch_mmad_l1() {
    %a = memref.alloc() : memref<2x4x4x16x16xf16, #hivm.address_space<cbuf>>
    %b = memref.alloc() : memref<2x4x4x16x16xf16, #hivm.address_space<cbuf>>
    %c = memref.alloc() : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>
    %dst = memref.alloc() : memref<2x64x64xf16, #hivm.address_space<gm>>
    %true = arith.constant true
    %m = arith.constant 64 : index
    %k = arith.constant 64 : index
    %n = arith.constant 64 : index

    hivm.hir.batchMmadL1 {already_set_real_mkn}
      ins(%a, %b, %true, %m, %k, %n :
          memref<2x4x4x16x16xf16, #hivm.address_space<cbuf>>,
          memref<2x4x4x16x16xf16, #hivm.address_space<cbuf>>,
          i1, index, index, index)
      outs(%c : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%c : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<2x64x64xf16, #hivm.address_space<gm>>)
    return
  }

  // Even when the nd strides overrun ND_PARA's 16-bit fields (src 65536, dst
  // 1048576) the batch stays a single call: splitting it is the library's job,
  // so that the unit flag keeps exactly one consumer.
  // CHECK-LABEL: func.func @lower_batch_fixpipe_nd_overflow
  // CHECK-NOT: scf.for
  // CHECK: call @fixpipe_nz2nd_float_to_half_5d_to_3d_gm
  // CHECK-NOT: call @fixpipe_nz2nd_float_to_half_4d_to_2d_gm
  func.func @lower_batch_fixpipe_nd_overflow() {
    %c = memref.alloc() : memref<2x64x64x16x16xf32, #hivm.address_space<cc>>
    %dst = memref.alloc() : memref<2x1024x1024xf16, #hivm.address_space<gm>>

    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%c : memref<2x64x64x16x16xf32, #hivm.address_space<cc>>)
      outs(%dst : memref<2x1024x1024xf16, #hivm.address_space<gm>>)
    return
  }

  // CHECK-LABEL: func.func @lower_batch_mmad_l1_bf16
  // CHECK: call @batch_mma_tile_bfloat16_t_to_float
  // CHECK-NOT: call @mma_tile_bfloat16_t_to_float
  // CHECK-NOT: hivm.hir.batchMmadL1
  func.func @lower_batch_mmad_l1_bf16() {
    %a = memref.alloc() : memref<2x4x4x16x16xbf16, #hivm.address_space<cbuf>>
    %b = memref.alloc() : memref<2x4x4x16x16xbf16, #hivm.address_space<cbuf>>
    %c = memref.alloc() : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>
    %true = arith.constant true
    %m = arith.constant 64 : index
    %k = arith.constant 64 : index
    %n = arith.constant 64 : index

    hivm.hir.batchMmadL1 {already_set_real_mkn}
      ins(%a, %b, %true, %m, %k, %n :
          memref<2x4x4x16x16xbf16, #hivm.address_space<cbuf>>,
          memref<2x4x4x16x16xbf16, #hivm.address_space<cbuf>>,
          i1, index, index, index)
      outs(%c : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>)
    return
  }

  // CHECK-LABEL: func.func @lower_batch_mmad_l1_f32
  // CHECK: call @batch_mma_tile_float_to_float
  // CHECK-NOT: call @mma_tile_float_to_float
  // CHECK-NOT: hivm.hir.batchMmadL1
  func.func @lower_batch_mmad_l1_f32() {
    %a = memref.alloc() : memref<2x4x4x16x16xf32, #hivm.address_space<cbuf>>
    %b = memref.alloc() : memref<2x4x4x16x16xf32, #hivm.address_space<cbuf>>
    %c = memref.alloc() : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>
    %true = arith.constant true
    %m = arith.constant 64 : index
    %k = arith.constant 64 : index
    %n = arith.constant 64 : index

    hivm.hir.batchMmadL1 {already_set_real_mkn}
      ins(%a, %b, %true, %m, %k, %n :
          memref<2x4x4x16x16xf32, #hivm.address_space<cbuf>>,
          memref<2x4x4x16x16xf32, #hivm.address_space<cbuf>>,
          i1, index, index, index)
      outs(%c : memref<2x4x4x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}
