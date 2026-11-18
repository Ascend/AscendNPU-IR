// RUN: bishengir-opt %s -hivm-infer-data-layout -canonicalize -cse | FileCheck %s --check-prefix=LAYOUT
// RUN: bishengir-opt %s -hivm-infer-data-layout -canonicalize -cse -hivm-flatten-ops -convert-hivm-to-std -cse | FileCheck %s --check-prefix=LOWER
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-infer-data-layout -canonicalize -cse -hivm-plan-memory -convert-hivm-to-std -cse | FileCheck %s --check-prefix=LOWER --implicit-check-not=memref.alloc

// Frontend-validated whole views use the unchanged partition protocol.
// MN=48, K=32 distinguish the two matrix axes. No tail support is added.
// Initialize each L1 allocation from GM so memory planning sees its first write.
// Compile-only coverage: these examples do not provide device synchronization.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // LAYOUT-LABEL: func.func @ca_f16_plain
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.nd2nz
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @ca_f16_plain
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_half(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @ca_f16_plain(%gm: memref<48x32xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<48x32xf16, #hivm.address_space<gm>> to memref<48x32xf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @ca_f16_transpose
  // LAYOUT-DAG: memref.alloc() : memref<3x2x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
  // LOWER-LABEL: func.func @ca_f16_transpose
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_half_transpose(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @ca_f16_transpose(%gm: memref<32x48xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<32x48xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<32x48xf16, #hivm.address_space<gm>> to memref<32x48xf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<32x48xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // LAYOUT-LABEL: func.func @cb_f16_plain
  // LAYOUT-DAG: memref.alloc() : memref<3x2x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<cb>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @cb_f16_plain
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_cb_half(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @cb_f16_plain(%gm: memref<32x48xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<32x48xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<32x48xf16, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<32x48xf16, #hivm.address_space<gm>> to memref<32x48xf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<32x48xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<32x48xf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @cb_f16_transpose
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<cb>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
  // LOWER-LABEL: func.func @cb_f16_transpose
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_cb_half_transpose(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @cb_f16_transpose(%gm: memref<48x32xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<32x48xf16, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<48x32xf16, #hivm.address_space<gm>> to memref<48x32xf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<32x48xf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // LAYOUT-LABEL: func.func @ca_bf16_plain
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xbf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xbf16, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @ca_bf16_plain
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_bfloat16_t(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @ca_bf16_plain(%gm: memref<48x32xbf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xbf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xbf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<48x32xbf16, #hivm.address_space<gm>> to memref<48x32xbf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xbf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xbf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @ca_bf16_transpose
  // LAYOUT-DAG: memref.alloc() : memref<3x2x16x16xbf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xbf16, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
  // LOWER-LABEL: func.func @ca_bf16_transpose
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_bfloat16_t_transpose(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @ca_bf16_transpose(%gm: memref<32x48xbf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<32x48xbf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xbf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<32x48xbf16, #hivm.address_space<gm>> to memref<32x48xbf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<32x48xbf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xbf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // LAYOUT-LABEL: func.func @cb_bf16_plain
  // LAYOUT-DAG: memref.alloc() : memref<3x2x16x16xbf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xbf16, #hivm.address_space<cb>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @cb_bf16_plain
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_cb_bfloat16_t(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @cb_bf16_plain(%gm: memref<32x48xbf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<32x48xbf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<32x48xbf16, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<32x48xbf16, #hivm.address_space<gm>> to memref<32x48xbf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<32x48xbf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<32x48xbf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @cb_bf16_transpose
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xbf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xbf16, #hivm.address_space<cb>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
  // LOWER-LABEL: func.func @cb_bf16_transpose
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_cb_bfloat16_t_transpose(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @cb_bf16_transpose(%gm: memref<48x32xbf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xbf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<32x48xbf16, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<48x32xbf16, #hivm.address_space<gm>> to memref<48x32xbf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xbf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<32x48xbf16, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // LAYOUT-LABEL: func.func @ca_f32_plain
  // LAYOUT-DAG: memref.alloc() : memref<4x3x16x8xf32, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<4x3x16x8xf32, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @ca_f32_plain
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_float(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @ca_f32_plain(%gm: memref<48x32xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xf32, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf32, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<48x32xf32, #hivm.address_space<gm>> to memref<48x32xf32, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xf32, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf32, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @ca_f32_transpose
  // LAYOUT-DAG: memref.alloc() : memref<6x2x16x8xf32, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<4x3x16x8xf32, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
  // LOWER-LABEL: func.func @ca_f32_transpose
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_float_transpose(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @ca_f32_transpose(%gm: memref<32x48xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<32x48xf32, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf32, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<32x48xf32, #hivm.address_space<gm>> to memref<32x48xf32, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<32x48xf32, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf32, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // LAYOUT-LABEL: func.func @cb_f32_plain
  // LAYOUT-DAG: memref.alloc() : memref<6x2x16x8xf32, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<4x3x16x8xf32, #hivm.address_space<cb>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @cb_f32_plain
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_cb_float(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @cb_f32_plain(%gm: memref<32x48xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<32x48xf32, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<32x48xf32, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<32x48xf32, #hivm.address_space<gm>> to memref<32x48xf32, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<32x48xf32, #hivm.address_space<cbuf>>)
      outs(%d : memref<32x48xf32, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @cb_f32_transpose
  // LAYOUT-DAG: memref.alloc() : memref<4x3x16x8xf32, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<4x3x16x8xf32, #hivm.address_space<cb>>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
  // LOWER-LABEL: func.func @cb_f32_transpose
  // LOWER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // LOWER-DAG: %[[K:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[ONE:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_cb_float_transpose(
  // LOWER-SAME: %[[ZERO]], %[[K]], %[[K]], %[[ONE]], %[[K]], %[[MN]])
  func.func @cb_f32_transpose(%gm: memref<48x32xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xf32, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<32x48xf32, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<48x32xf32, #hivm.address_space<gm>> to memref<48x32xf32, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xf32, #hivm.address_space<cbuf>>)
      outs(%d : memref<32x48xf32, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }


  // The source hint means its logical axes are already normalized.
  // LAYOUT-LABEL: func.func @transpose_hint
  // LAYOUT-DAG: %[[HINT_SRC:.*]] = memref.alloc() : memref<3x2x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: %[[HINT_CAST:.*]] = memref.cast %[[HINT_SRC]] : memref<3x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: %[[HINT_DST:.*]] = memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.nd2nz
  // LAYOUT-SAME: ins(%{{.*}} : memref<32x48xf16, #hivm.address_space<gm>>)
  // LAYOUT-SAME: outs(%[[HINT_CAST]] : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>)
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: ins(%[[HINT_SRC]] : memref<3x2x16x16xf16, #hivm.address_space<cbuf>>)
  // LAYOUT-SAME: outs(%[[HINT_DST]] : memref<2x3x16x16xf16, #hivm.address_space<ca>>)
  // LAYOUT-SAME: k_part_idx = 0, k_part = 32
  // LOWER-LABEL: func.func @transpose_hint
  // LOWER: call @nd2nz_half
  // LOWER: call @l12l0_cbuf_to_ca_half_transpose
  func.func @transpose_hint(%gm: memref<32x48xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() {hivm.fractal_layout = "nZ"} : memref<48x32xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf16, #hivm.address_space<ca>>
    // GM uses [K, M]; the hint describes L1 using normalized [M, K] axes.
    // After layout inference, ND2NZ writes the physical [3, 2, 16, 16] buffer.
    hivm.hir.nd2nz {dst_continuous}
      ins(%gm : memref<32x48xf16, #hivm.address_space<gm>>)
      outs(%s : memref<48x32xf16, #hivm.address_space<cbuf>>)
    hivm.hir.l12l0 ins(%s : memref<48x32xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48, is_transposed = true}
    return
  }

  // Both origins live in subviews. The partition coordinate stays zero.
  // These are interior aligned views, not copies of the whole allocation.
  // LAYOUT-LABEL: func.func @a_parent_views
  // LAYOUT-DAG: memref.alloc() : memref<6x4x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<4x6x16x16xf16, #hivm.address_space<ca>>
  // LAYOUT: memref.subview
  // LAYOUT-SAME: strided<[1024, 256, 16, 1], offset: 2304>
  // LAYOUT: memref.subview
  // LAYOUT-SAME: strided<[1536, 256, 16, 1], offset: 2048>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @a_parent_views
  // LOWER: call @l12l0_cbuf_to_ca_half
  func.func @a_parent_views(%gm: memref<64x96xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<64x96xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<96x64xf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<64x96xf16, #hivm.address_space<gm>> to memref<64x96xf16, #hivm.address_space<cbuf>>
    %sv = memref.subview %s[16, 32] [48, 32] [1, 1] : memref<64x96xf16, #hivm.address_space<cbuf>> to memref<48x32xf16, strided<[96, 1], offset: 1568>, #hivm.address_space<cbuf>>
    %dv = memref.subview %d[32, 16] [48, 32] [1, 1] : memref<96x64xf16, #hivm.address_space<ca>> to memref<48x32xf16, strided<[64, 1], offset: 2064>, #hivm.address_space<ca>>
    hivm.hir.l12l0 ins(%sv : memref<48x32xf16, strided<[96, 1], offset: 1568>, #hivm.address_space<cbuf>>)
      outs(%dv : memref<48x32xf16, strided<[64, 1], offset: 2064>, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @b_parent_views
  // LAYOUT-DAG: memref.alloc() : memref<6x6x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<4x8x16x16xf16, #hivm.address_space<cb>>
  // LAYOUT: memref.subview
  // LAYOUT-SAME: strided<[1536, 256, 16, 1], offset: 2048>
  // LAYOUT: memref.subview
  // LAYOUT-SAME: strided<[2048, 256, 16, 1], offset: 2560>
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @b_parent_views
  // LOWER: call @l12l0_cbuf_to_cb_half
  func.func @b_parent_views(%gm: memref<96x96xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<96x96xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<64x128xf16, #hivm.address_space<cb>>
    memref.copy %gm, %s : memref<96x96xf16, #hivm.address_space<gm>> to memref<96x96xf16, #hivm.address_space<cbuf>>
    %sv = memref.subview %s[32, 16] [32, 48] [1, 1] : memref<96x96xf16, #hivm.address_space<cbuf>> to memref<32x48xf16, strided<[96, 1], offset: 3088>, #hivm.address_space<cbuf>>
    %dv = memref.subview %d[16, 32] [32, 48] [1, 1] : memref<64x128xf16, #hivm.address_space<cb>> to memref<32x48xf16, strided<[128, 1], offset: 2080>, #hivm.address_space<cb>>
    hivm.hir.l12l0 ins(%sv : memref<32x48xf16, strided<[96, 1], offset: 3088>, #hivm.address_space<cbuf>>)
      outs(%dv : memref<32x48xf16, strided<[128, 1], offset: 2080>, #hivm.address_space<cb>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }

  // Existing partition IR must retain its coordinate and loop count.
  // LAYOUT-LABEL: func.func @legacy_partition
  // LAYOUT-DAG: memref.alloc() : memref<6x3x16x16xf16, #hivm.address_space<cbuf>>
  // LAYOUT-DAG: memref.alloc() : memref<2x3x16x16xf16, #hivm.address_space<ca>>
  // LAYOUT: hivm.hir.nd2nz
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 1, k_part = 32, k_part_ceil = 32, k_part_loop = 3, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @legacy_partition
  // LOWER-DAG: %[[IDX:.*]] = arith.constant 1 : i64
  // LOWER-DAG: %[[PART:.*]] = arith.constant 32 : i64
  // LOWER-DAG: %[[LOOPS:.*]] = arith.constant 3 : i64
  // LOWER-DAG: %[[MN:.*]] = arith.constant 48 : i64
  // LOWER: call @l12l0_cbuf_to_ca_half(
  // LOWER-SAME: %[[IDX]], %[[PART]], %[[PART]], %[[LOOPS]], %[[PART]], %[[MN]])
  func.func @legacy_partition(%gm: memref<48x96xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x96xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<48x96xf16, #hivm.address_space<gm>> to memref<48x96xf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x96xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf16, #hivm.address_space<ca>>)
      {k_part_idx = 1, k_part = 32, k_part_ceil = 32, k_part_loop = 3, k_part_actual = 32, m = 48}
    return
  }

  // LAYOUT-LABEL: func.func @gm_to_l1_to_l0
  // LAYOUT: hivm.hir.nd2nz
  // LAYOUT: hivm.hir.l12l0
  // LAYOUT-SAME: {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
  // LOWER-LABEL: func.func @gm_to_l1_to_l0
  // LOWER: call @l12l0_cbuf_to_ca_half
  func.func @gm_to_l1_to_l0(%gm: memref<48x32xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %s = memref.alloc() : memref<48x32xf16, #hivm.address_space<cbuf>>
    %d = memref.alloc() : memref<48x32xf16, #hivm.address_space<ca>>
    memref.copy %gm, %s : memref<48x32xf16, #hivm.address_space<gm>> to memref<48x32xf16, #hivm.address_space<cbuf>>
    hivm.hir.l12l0 ins(%s : memref<48x32xf16, #hivm.address_space<cbuf>>)
      outs(%d : memref<48x32xf16, #hivm.address_space<ca>>)
      {k_part_idx = 0, k_part = 32, k_part_ceil = 32, k_part_loop = 1, k_part_actual = 32, m = 48}
    return
  }
}
