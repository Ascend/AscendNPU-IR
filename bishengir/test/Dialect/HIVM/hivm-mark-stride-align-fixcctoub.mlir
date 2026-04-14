// RUN: bishengir-opt %s -hivm-mark-stride-align -split-input-file | FileCheck %s -check-prefix=AIC

// Test 1: NSize - B16, CS_N, ND_N, LE_N, N -> nSizeStride=16, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b16_cs_n_nd_n_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b16_cs_n_nd_n_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xf16, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 2: NSize - B16, CS_N, ND_N, LE_Y, N -> nSizeStride=16, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b16_cs_n_nd_n_le_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b16_cs_n_nd_n_le_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xf16, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 3: NSize - B16, CS_N, ND_Y, LE_N, N -> nSizeStride=16, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b16_cs_n_nd_y_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b16_cs_n_nd_y_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xf16, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 4: NSize - B16, CS_N, ND_Y, LE_Y, N -> nSizeStride=16, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b16_cs_n_nd_y_le_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b16_cs_n_nd_y_le_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xf16, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 5: NSize - B8, CS_N, ND_N, LE_N, N -> nSizeStride=16, alignBytes=16
// AIC-LABEL: func.func @test_nsize_b8_cs_n_nd_n_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 16>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b8_cs_n_nd_n_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi8, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi8, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xi8, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi8, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 6: NSize - B8, CS_N, ND_N, LE_Y, N -> nSizeStride=16, alignBytes=16
// AIC-LABEL: func.func @test_nsize_b8_cs_n_nd_n_le_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 16>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b8_cs_n_nd_n_le_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi8, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi8, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xi8, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi8, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 7: NSize - B8, CS_N, ND_Y, LE_N, N -> nSizeStride=32, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b8_cs_n_nd_y_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b8_cs_n_nd_y_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi8, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi8, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xi8, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi8, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 8: NSize - B8, CS_N, ND_Y, LE_Y, N -> nSizeStride=16, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b8_cs_n_nd_y_le_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b8_cs_n_nd_y_le_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi8, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi8, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xi8, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi8, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 9: NSize - B4, CS_N, ND_N, LE_N, N -> nSizeStride=16, alignBytes=8
// AIC-LABEL: func.func @test_nsize_b4_cs_n_nd_n_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 8>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b4_cs_n_nd_n_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi4, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi4, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi4, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xi4, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi4, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 10: NSize - B4, CS_N, ND_Y, LE_N, N -> nSizeStride=64, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b4_cs_n_nd_y_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b4_cs_n_nd_y_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi4, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi4, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi4, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xi4, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi4, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 11: NSize - B32, CS_N, ND_N, LE_N, N -> nSizeStride=16, alignBytes=64
// AIC-LABEL: func.func @test_nsize_b32_cs_n_nd_n_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 64>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b32_cs_n_nd_n_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 12: NSize - B32, CS_N, ND_Y, LE_N, N -> nSizeStride=8, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b32_cs_n_nd_y_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b32_cs_n_nd_y_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 13: NSize - B32, CS_Y, ND_N, LE_N, N -> nSizeStride=8, alignBytes=32
// AIC-LABEL: func.func @test_nsize_b32_cs_y_nd_n_le_n
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b32_cs_y_nd_n_le_n() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe {channel_split = true} ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 14: NSize - B32, CS_Y, ND_N, LE_N, SplitN -> nSizeStride=8, alignBytes=32 (tiling_dim = rank-2)
// AIC-LABEL: func.func @test_nsize_b32_cs_y_nd_n_le_n_splitn
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b32_cs_y_nd_n_le_n_splitn() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = -1 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe {channel_split = true} ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 15: NSize - B32, CS_Y, ND_Y, LE_N, SplitM -> MSizeStride=2, alignBytes=8
// AIC-LABEL: func.func @test_nsize_b32_cs_y_nd_y_le_n_splitm
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 8>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_nsize_b32_cs_y_nd_y_le_n_splitm() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe {channel_split = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 16: MSize - B16, DN_Y, N -> mSizeStride=16, alignBytes=32
// AIC-LABEL: func.func @test_msize_b16_dn_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_msize_b16_dn_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%alloc_0 : memref<16x16xf16, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 17: MSize - B8, DN_Y, N -> mSizeStride=32, alignBytes=32
// AIC-LABEL: func.func @test_msize_b8_dn_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_msize_b8_dn_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi8, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi8, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%alloc_0 : memref<16x16xi8, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi8, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 18: MSize - B4, DN_Y, N -> mSizeStride=64, alignBytes=32
// AIC-LABEL: func.func @test_msize_b4_dn_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_msize_b4_dn_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xi4, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xi4, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi4, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%alloc_0 : memref<16x16xi4, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xi4, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 19: MSize - B32, DN_N, SplitN -> nSizeStride=32, alignBytes=128
// AIC-LABEL: func.func @test_msize_b32_dn_n_splitn
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 128>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_msize_b32_dn_n_splitn() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : i32} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}

// -----//

// Test 20: MSize - B32, DN_Y, N -> mSizeStride=8, alignBytes=32
// AIC-LABEL: func.func @test_msize_b32_dn_y
// AIC: annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_msize_b32_dn_y() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2dn>} ins(%alloc_0 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}
}