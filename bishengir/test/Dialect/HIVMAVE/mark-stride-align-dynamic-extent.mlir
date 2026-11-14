// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// A UB DMA operand whose sub-low stride is already hardware-aligned needs no
// stride alignment: every row starts on an aligned boundary regardless of how
// many elements a row carries. The lowest dimension may therefore be dynamic.

// CHECK-LABEL: func.func @dyn_extent_aligned_stride
// CHECK-NOT: hivm.stride_align_dims

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dyn_extent_aligned_stride(
      %gm: memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<?x?xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : bf16
    hivm.hir.load
        ins(%gm : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<?x?xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : bf16
    return
  }
}

// -----

// Same for rank > 2, where isSubLowDimStrideAlign guards the stride above the
// sub-low one: 32xf32 = 128B and 96xf32 = 384B are both aligned.

// CHECK-LABEL: func.func @dyn_extent_aligned_stride_rank3
// CHECK-NOT: hivm.stride_align_dims

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dyn_extent_aligned_stride_rank3(
      %gm: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<?x?x?xf32, strided<[96, 32, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : f32
    hivm.hir.load
        ins(%gm : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<?x?x?xf32, strided<[96, 32, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : f32
    return
  }
}

// -----

// A dynamic lowest dimension over an unaligned sub-low stride stays marked:
// 24xbf16 = 48B, so the rows do land off the 32B boundary.

// CHECK-LABEL: func.func @dyn_extent_unaligned_stride
// CHECK: annotation.mark %arg1 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dyn_extent_unaligned_stride(
      %gm: memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<?x?xbf16, strided<[24, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : bf16
    hivm.hir.load
        ins(%gm : memref<?x?xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<?x?xbf16, strided<[24, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : bf16
    return
  }
}

// -----

// An unknown sub-low stride cannot be proven aligned. The kDynamic sentinel
// must not be fed to the modulo, which would wrap to zero and read as aligned.

// CHECK-LABEL: func.func @dynamic_stride
// CHECK: annotation.mark %arg1 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dynamic_stride(
      %gm: memref<2x8xbf16, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<2x8xbf16, strided<[?, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : bf16
    hivm.hir.load
        ins(%gm : memref<2x8xbf16, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<2x8xbf16, strided<[?, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : bf16
    return
  }
}

// -----

// The stride above the sub-low one is checked by isSubLowDimStrideAlign, which
// this change leaves alone. An unknown one there is the sub-low stride times a
// dim extent, so it is a multiple of a stride this test already requires to be
// a multiple of the hardware alignment, and no alignment is requested here
// either.

// CHECK-LABEL: func.func @rank3_dynamic_outer_stride
// CHECK-NOT: hivm.stride_align_dims

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @rank3_dynamic_outer_stride(
      %gm: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<?x?x?xf32, strided<[?, 32, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : f32
    hivm.hir.load
        ins(%gm : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<?x?x?xf32, strided<[?, 32, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : f32
    return
  }
}

// -----

// Unchanged: a static lowest dimension under an aligned sub-low stride is
// already skipped today, and the lowest dimension need not match the stride.

// CHECK-LABEL: func.func @static_extent_aligned_stride
// CHECK-NOT: hivm.stride_align_dims

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @static_extent_aligned_stride(
      %gm: memref<?x8xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<?x8xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : bf16
    hivm.hir.load
        ins(%gm : memref<?x8xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<?x8xbf16, strided<[16, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : bf16
    return
  }
}

// -----

// Unchanged: a tightly packed row keeps its skip even when the resulting row
// stride is unaligned, which is what the DMA template's compact mode handles.

// CHECK-LABEL: func.func @packed_unaligned_stride
// CHECK-NOT: hivm.stride_align_dims

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @packed_unaligned_stride(
      %gm: memref<2x24xbf16, strided<[24, 1], offset: ?>, #hivm.address_space<gm>>,
      %ub: memref<2x24xbf16, strided<[24, 1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %cst = arith.constant 0.000000e+00 : bf16
    hivm.hir.load
        ins(%gm : memref<2x24xbf16, strided<[24, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%ub : memref<2x24xbf16, strided<[24, 1]>, #hivm.address_space<ub>>)
        pad_mode = <PadValue> pad_value = %cst : bf16
    return
  }
}
