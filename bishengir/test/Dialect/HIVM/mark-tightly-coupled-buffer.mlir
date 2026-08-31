// RUN: bishengir-opt %s -hivm-mark-tightly-coupled-buffer -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_copy_to_cbuf
  func.func @mark_copy_to_cbuf() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %cbuf = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
    %ub_marked = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %ub_marked {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<128x128xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>}
    // CHECK: %{{.*}} = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    // CHECK-NEXT: %{{.*}} = memref.memory_space_cast
    %ub_scalar = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %ub_scalar : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
    %tensor_cbuf = tensor.empty() : tensor<8x8x16x16xf16>
    %tensor_scalar = tensor.empty() : tensor<128xf32>
    hivm.hir.copy ins(%tensor_cbuf : tensor<8x8x16x16xf16>) outs(%cbuf : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
    hivm.hir.copy ins(%tensor_scalar : tensor<128xf32>) outs(%cast : memref<128xf32>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_fixpipe_to_ub
  func.func @mark_fixpipe_to_ub() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %cbuf = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %cbuf {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %ub0 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %ub0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<128x128xf32, #hivm.address_space<ub>>
    %ub1 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %ub1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<128x128xf32, #hivm.address_space<ub>>
    %tensor = tensor.empty() : tensor<128x128xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%tensor : tensor<128x128xf32>) outs(%ub0 : memref<128x128xf32, #hivm.address_space<ub>>)
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%tensor : tensor<128x128xf32>) outs(%ub1 : memref<128x128xf32, #hivm.address_space<ub>>)
    // CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>}
    // CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>}
    // CHECK-NOT: tightly_coupled_buffer<1>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_new_candidates
  func.func @mark_new_candidates() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %cbuf = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[CBUF:.*]] = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: annotation.mark %[[CBUF]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
    %ub = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    // CHECK: %[[UB:.*]] = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[UB]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
    %tensor_cbuf = tensor.empty() : tensor<8x8x16x16xf16>
    %tensor_ub = tensor.empty() : tensor<128x128xf32>
    hivm.hir.copy ins(%tensor_cbuf : tensor<8x8x16x16xf16>) outs(%cbuf : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%tensor_ub : tensor<128x128xf32>) outs(%ub : memref<128x128xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_fixpipe_scf_if_dual_branch
  func.func @mark_fixpipe_scf_if_dual_branch() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %cond = arith.constant true
    %ub_then = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: %[[UB_THEN:.*]] = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[UB_THEN]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
    %ub_else = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: %[[UB_ELSE:.*]] = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[UB_ELSE]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
    %tensor = tensor.empty() : tensor<128x128xf32>
    %selected = scf.if %cond -> (tensor<64x128xf32>) {
      %cast_then = memref.memory_space_cast %ub_then : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
      %t_then = bufferization.to_tensor %cast_then restrict writable : memref<64x128xf32>
      scf.yield %t_then : tensor<64x128xf32>
    } else {
      %cast_else = memref.memory_space_cast %ub_else : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
      %t_else = bufferization.to_tensor %cast_else restrict writable : memref<64x128xf32>
      scf.yield %t_else : tensor<64x128xf32>
    }
    %dst = bufferization.to_memref %selected : memref<64x128xf32>
    %dst_ub = memref.memory_space_cast %dst : memref<64x128xf32> to memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%tensor : tensor<128x128xf32>) outs(%dst_ub : memref<64x128xf32, #hivm.address_space<ub>>)
    // CHECK-NOT: tightly_coupled_buffer<2>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @no_mark_non_950
  func.func @no_mark_non_950() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %cbuf = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    %tensor = tensor.empty() : tensor<8x8x16x16xf16>
    hivm.hir.copy ins(%tensor : tensor<8x8x16x16xf16>) outs(%cbuf : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
    // CHECK: memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK-NOT: tightly_coupled_buffer
    return
  }
}

// -----

// Device-print shape produced by insert-l12ub-for-debug: the l12ub (CUBE side)
// writes a UB alloc through a subview, and the debug op (VECTOR side) reads it
// after split-mix-kernel. The subview dst must be traced back to the alloc so
// both split sides inherit the same tightly-coupled id.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_l12ub_subview_dst
  func.func @mark_l12ub_subview_dst() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %cbuf = memref.alloc() : memref<1x15x16x8xf32, #hivm.address_space<cbuf>>
    // CHECK: %[[UB:.*]] = memref.alloc() : memref<234x8x1xf32, #hivm.address_space<ub>>
    %ub = memref.alloc() : memref<234x8x1xf32, #hivm.address_space<ub>>
    // CHECK-NEXT: annotation.mark %[[UB]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<234x8x1xf32, #hivm.address_space<ub>>
    %subview = memref.subview %ub[0, 0, 0] [234, 1, 1] [1, 1, 1] : memref<234x8x1xf32, #hivm.address_space<ub>> to memref<234x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.l12ub ins(%cbuf : memref<1x15x16x8xf32, #hivm.address_space<cbuf>>) outs(%subview : memref<234x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    // The l12ub src cbuf alloc is not a dst candidate and must stay unmarked.
    // CHECK-NOT: annotation.mark %{{.*}} : memref<1x15x16x8xf32, #hivm.address_space<cbuf>>
    return
  }
}

// -----

// An l12ub UB dst alloc gets a fresh id that skips ids already reserved by
// other marked candidates in the same function.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @mark_l12ub_skips_reserved_id
  func.func @mark_l12ub_skips_reserved_id() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %cbuf = memref.alloc() : memref<1x15x16x8xf32, #hivm.address_space<cbuf>>
    %ub = memref.alloc() : memref<234x1xf32, #hivm.address_space<ub>>
    %ub_fixpipe = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    annotation.mark %ub_fixpipe {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x128xf32, #hivm.address_space<ub>>
    %tensor = tensor.empty() : tensor<128x128xf32>
    %cbuf_cast = memref.cast %cbuf : memref<1x15x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
    hivm.hir.l12ub ins(%cbuf_cast : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) outs(%ub : memref<234x1xf32, #hivm.address_space<ub>>)
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%tensor : tensor<128x128xf32>) outs(%ub_fixpipe : memref<128x128xf32, #hivm.address_space<ub>>)
    // CHECK: annotation.mark %{{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<234x1xf32, #hivm.address_space<ub>>
    return
  }
}
