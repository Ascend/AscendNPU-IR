// RUN: bishengir-opt %s -hivm-mark-stride-align -split-input-file | FileCheck %s --check-prefixes=COMMON,NEW
// RUN: bishengir-opt %s -pass-pipeline="builtin.module(func.func(hivm-mark-stride-align{align-packed-unaligned=false}))" -split-input-file | FileCheck %s --check-prefixes=COMMON,LEGACY

// A tightly packed layout whose row stride is NOT hardware aligned
// (200 * 16bit = 400B, not a multiple of 32B) must still be marked for
// stride alignment. Tightly packed (dimSize == stride) must not be treated
// as a reason to skip alignment (issue #14, 45x slowdown on UBDMA).
// Disabling align-packed-unaligned restores the legacy packed-skip
// behavior, so no mark is emitted.
// COMMON-LABEL: func.func @ubdma_packed_unaligned_stride
// NEW: annotation.mark %{{.+}} {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x200xf16, #hivm.address_space<ub>>
// LEGACY-NOT: annotation.mark
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @ubdma_packed_unaligned_stride() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %src = memref.alloc() : memref<4x200xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<4x200xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4x200xf16, #hivm.address_space<gm>>) outs(%dst : memref<4x200xf16, #hivm.address_space<ub>>)
  return
}
}

// -----

// A tightly packed layout whose row stride is already hardware aligned
// (128 * 16bit = 256B) needs no stride-align mark in either mode.
// COMMON-LABEL: func.func @ubdma_packed_aligned_stride
// COMMON-NOT: annotation.mark
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @ubdma_packed_aligned_stride() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %src = memref.alloc() : memref<4x128xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<4x128xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4x128xf16, #hivm.address_space<gm>>) outs(%dst : memref<4x128xf16, #hivm.address_space<ub>>)
  return
}
}

// -----

// The same applies to hivm.hir.store (copy_ub_to_gm): a tightly packed
// layout with a non-aligned row stride must still be marked, unless
// align-packed-unaligned is disabled.
// COMMON-LABEL: func.func @ubdma_store_packed_unaligned_stride
// NEW: annotation.mark %{{.+}} {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x200xf16, #hivm.address_space<ub>>
// LEGACY-NOT: annotation.mark
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @ubdma_store_packed_unaligned_stride() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %src = memref.alloc() : memref<4x200xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<4x200xf16, #hivm.address_space<gm>>
  hivm.hir.store ins(%src : memref<4x200xf16, #hivm.address_space<ub>>) outs(%dst : memref<4x200xf16, #hivm.address_space<gm>>)
  return
}
}
