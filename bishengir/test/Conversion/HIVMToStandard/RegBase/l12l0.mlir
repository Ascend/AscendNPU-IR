// RUN: bishengir-opt %s -convert-hivm-to-std -cse | FileCheck %s --implicit-check-not=hivm.hir.l12l0 --implicit-check-not=scf.for

// Physical rank-4 descriptors, not logical 2-D shapes. A5 L0A=zN, L0B=nZ.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // CHECK-LABEL: func.func @a_nonzero_k
  // CHECK-SAME: %[[SRC:.*]]: memref<6x2x16x16xf16, #hivm.address_space<cbuf>>, %[[DST:.*]]: memref<2x2x16x16xf16, #hivm.address_space<ca>>
  // CHECK-DAG: %[[IDX:.*]] = arith.constant 1 : i64
  // CHECK-DAG: %[[PART:.*]] = arith.constant 32 : i64
  // CHECK-DAG: %[[LOOPS:.*]] = arith.constant 3 : i64
  // CHECK-DAG: %[[S:.*]] = memref.cast %[[SRC]]
  // CHECK-DAG: %[[D:.*]] = memref.cast %[[DST]]
  // CHECK: call @l12l0_cbuf_to_ca_half(%[[S]], %[[D]], %[[IDX]], %[[PART]], %[[PART]], %[[LOOPS]], %[[PART]], %[[PART]]) : (memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, i64, i64, i64, i64, i64, i64) -> ()
  func.func @a_nonzero_k(%src: memref<6x2x16x16xf16, #hivm.address_space<cbuf>>, %dst: memref<2x2x16x16xf16, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<6x2x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x2x16x16xf16, #hivm.address_space<ca>>)
      {k_part_idx = 1, k_part = 32, k_part_ceil = 32, k_part_loop = 3, k_part_actual = 32, m = 32}
    return
  }

  // N=48, K=96; the B-path m parameter denotes N, not K.
  // CHECK-LABEL: func.func @b_nonzero_k
  // CHECK: arith.constant 48 : i64
  // CHECK: call @l12l0_cbuf_to_cb_half
  func.func @b_nonzero_k(%src: memref<3x6x16x16xf16, #hivm.address_space<cbuf>>, %dst: memref<2x3x16x16xf16, #hivm.address_space<cb>>) {
    hivm.hir.l12l0 ins(%src : memref<3x6x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x3x16x16xf16, #hivm.address_space<cb>>)
      {k_part_idx = 1, k_part = 32, k_part_ceil = 32, k_part_loop = 3, k_part_actual = 32, m = 48}
    return
  }

  // Transposed A: physical nZ represents logical M=32, K=96.
  // CHECK-LABEL: func.func @a_transpose_bf16
  // CHECK: call @l12l0_cbuf_to_ca_bfloat16_t_transpose
  func.func @a_transpose_bf16(%src: memref<2x6x16x16xbf16, #hivm.address_space<cbuf>>, %dst: memref<2x2x16x16xbf16, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<2x6x16x16xbf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x2x16x16xbf16, #hivm.address_space<ca>>)
      {k_part_idx = 2, k_part = 32, k_part_ceil = 32, k_part_loop = 3, k_part_actual = 32, m = 32, is_transposed = true}
    return
  }

  // F32 has C0=8. Transposed B: logical K=96, N=48.
  // CHECK-LABEL: func.func @b_transpose_f32
  // CHECK: call @l12l0_cbuf_to_cb_float_transpose
  func.func @b_transpose_f32(%src: memref<12x3x16x8xf32, #hivm.address_space<cbuf>>, %dst: memref<4x3x16x8xf32, #hivm.address_space<cb>>) {
    hivm.hir.l12l0 ins(%src : memref<12x3x16x8xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<4x3x16x8xf32, #hivm.address_space<cb>>)
      {k_part_idx = 1, k_part = 32, k_part_ceil = 32, k_part_loop = 3, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // Parent stride must survive canonicalization into the runtime descriptor.
  // Nonzero src/dst offsets are applied once, in the template, independently
  // of the K partition coordinate. The last K block is 13 padded to 16.
  // CHECK-LABEL: func.func @b_tail_parent_views
  // CHECK-DAG: %[[SV:.*]] = memref.subview
  // CHECK-DAG: %[[DV:.*]] = memref.subview
  // CHECK-DAG: %[[TI:.*]] = arith.constant 2 : i64
  // CHECK-DAG: %[[TP:.*]] = arith.constant 32 : i64
  // CHECK-DAG: %[[TC:.*]] = arith.constant 16 : i64
  // CHECK-DAG: %[[TL:.*]] = arith.constant 3 : i64
  // CHECK-DAG: %[[TA:.*]] = arith.constant 13 : i64
  // CHECK-DAG: %[[TN:.*]] = arith.constant 48 : i64
  // CHECK-DAG: %[[TS:.*]] = memref.cast %[[SV]] : memref<3x5x16x16xf16, strided<[2048, 256, 16, 1], offset: 256>, #hivm.address_space<cbuf>>
  // CHECK-DAG: %[[TD:.*]] = memref.cast %[[DV]] : memref<2x3x16x16xf16, strided<[1024, 256, 16, 1], offset: 256>, #hivm.address_space<cb>>
  // CHECK: call @l12l0_cbuf_to_cb_half(%[[TS]], %[[TD]], %[[TI]], %[[TP]], %[[TC]], %[[TL]], %[[TA]], %[[TN]])
  func.func @b_tail_parent_views(%src: memref<3x8x16x16xf16, #hivm.address_space<cbuf>>, %dst: memref<2x4x16x16xf16, #hivm.address_space<cb>>) {
    %s = memref.subview %src[0, 1, 0, 0] [3, 5, 16, 16] [1, 1, 1, 1] : memref<3x8x16x16xf16, #hivm.address_space<cbuf>> to memref<3x5x16x16xf16, strided<[2048, 256, 16, 1], offset: 256>, #hivm.address_space<cbuf>>
    %d = memref.subview %dst[0, 1, 0, 0] [2, 3, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16, #hivm.address_space<cb>> to memref<2x3x16x16xf16, strided<[1024, 256, 16, 1], offset: 256>, #hivm.address_space<cb>>
    hivm.hir.l12l0 ins(%s : memref<3x5x16x16xf16, strided<[2048, 256, 16, 1], offset: 256>, #hivm.address_space<cbuf>>)
      outs(%d : memref<2x3x16x16xf16, strided<[1024, 256, 16, 1], offset: 256>, #hivm.address_space<cb>>)
      {k_part_idx = 2, k_part = 32, k_part_ceil = 16, k_part_loop = 3, k_part_actual = 13, m = 48}
    return
  }

  // Single tail whose nominal partition is larger than the source K extent.
  // Dynamic offsets are preserved, and guarded in the template.
  // CHECK-LABEL: func.func @a_single_tail_dynamic_offset
  // CHECK-DAG: arith.constant 64 : i64
  // CHECK-DAG: arith.constant 13 : i64
  // CHECK: call @l12l0_cbuf_to_ca_half
  func.func @a_single_tail_dynamic_offset(%src: memref<1x2x16x16xf16, strided<[768, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>, %dst: memref<2x2x16x16xf16, strided<[768, 256, 16, 1], offset: ?>, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<1x2x16x16xf16, strided<[768, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x2x16x16xf16, strided<[768, 256, 16, 1], offset: ?>, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 64, k_part_ceil = 16, k_part_loop = 1, k_part_actual = 13, m = 32}
    return
  }

  // CHECK-LABEL: func.func @a_transposed_f16_export
  // CHECK: call @l12l0_cbuf_to_ca_half_transpose
  func.func @a_transposed_f16_export(%src: memref<2x4x16x16xf16, #hivm.address_space<cbuf>>, %dst: memref<1x2x16x16xf16, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x2x16x16xf16, #hivm.address_space<ca>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32, is_transposed = true}
    return
  }

  // CHECK-LABEL: func.func @b_transposed_f16_export
  // CHECK: call @l12l0_cbuf_to_cb_half_transpose
  func.func @b_transposed_f16_export(%src: memref<4x2x16x16xf16, #hivm.address_space<cbuf>>, %dst: memref<1x2x16x16xf16, #hivm.address_space<cb>>) {
    hivm.hir.l12l0 ins(%src : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x2x16x16xf16, #hivm.address_space<cb>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32, is_transposed = true}
    return
  }

  // CHECK-LABEL: func.func @a_plain_bf16_export
  // CHECK: call @l12l0_cbuf_to_ca_bfloat16_t
  func.func @a_plain_bf16_export(%src: memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>, %dst: memref<1x2x16x16xbf16, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x2x16x16xbf16, #hivm.address_space<ca>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32}
    return
  }

  // CHECK-LABEL: func.func @b_plain_bf16_export
  // CHECK: call @l12l0_cbuf_to_cb_bfloat16_t
  func.func @b_plain_bf16_export(%src: memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>, %dst: memref<1x2x16x16xbf16, #hivm.address_space<cb>>) {
    hivm.hir.l12l0 ins(%src : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x2x16x16xbf16, #hivm.address_space<cb>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32}
    return
  }

  // CHECK-LABEL: func.func @b_transposed_bf16_export
  // CHECK: call @l12l0_cbuf_to_cb_bfloat16_t_transpose
  func.func @b_transposed_bf16_export(%src: memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>, %dst: memref<1x2x16x16xbf16, #hivm.address_space<cb>>) {
    hivm.hir.l12l0 ins(%src : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>)
      outs(%dst : memref<1x2x16x16xbf16, #hivm.address_space<cb>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32, is_transposed = true}
    return
  }

  // CHECK-LABEL: func.func @a_plain_f32_export
  // CHECK: call @l12l0_cbuf_to_ca_float
  func.func @a_plain_f32_export(%src: memref<8x2x16x8xf32, #hivm.address_space<cbuf>>, %dst: memref<2x2x16x8xf32, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<8x2x16x8xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x2x16x8xf32, #hivm.address_space<ca>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32}
    return
  }

  // CHECK-LABEL: func.func @a_transposed_f32_export
  // CHECK: call @l12l0_cbuf_to_ca_float_transpose
  func.func @a_transposed_f32_export(%src: memref<4x4x16x8xf32, #hivm.address_space<cbuf>>, %dst: memref<2x2x16x8xf32, #hivm.address_space<ca>>) {
    hivm.hir.l12l0 ins(%src : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x2x16x8xf32, #hivm.address_space<ca>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32, is_transposed = true}
    return
  }

  // CHECK-LABEL: func.func @b_plain_f32_export
  // CHECK: call @l12l0_cbuf_to_cb_float
  func.func @b_plain_f32_export(%src: memref<4x4x16x8xf32, #hivm.address_space<cbuf>>, %dst: memref<2x2x16x8xf32, #hivm.address_space<cb>>) {
    hivm.hir.l12l0 ins(%src : memref<4x4x16x8xf32, #hivm.address_space<cbuf>>)
      outs(%dst : memref<2x2x16x8xf32, #hivm.address_space<cb>>)
      {k_part_idx = 1, k_part = 16, k_part_ceil = 16, k_part_loop = 4, k_part_actual = 16, m = 32}
    return
  }
}
