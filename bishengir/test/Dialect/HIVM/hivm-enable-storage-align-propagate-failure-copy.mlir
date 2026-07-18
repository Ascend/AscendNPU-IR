// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// Test case where unrealized conversion cast propagation fails and CopyOps are inserted.
// This happens when a memref-producing operation (like memref.reinterpret_cast) cannot 
// have the conversion cast pushed past it, so the failure handling inserts CopyOps to 
// maintain data consistency between original and aligned buffers.

// CHECK-LABEL: func @propagate_failure_reinterpret_cast
func.func @propagate_failure_reinterpret_cast() {
  // Original allocation with non-aligned sizes
  %0 = memref.alloc() : memref<13x13x13xf32, #hivm.address_space<ub>>
  
  // Annotate for stride alignment on dim 1 (require 32-byte alignment = 8 f32 elements)
  // This forces alloc expansion, and since reinterpret_cast cannot have the conversion
  // cast pushed past it, CopyOps are inserted to sync data.
  // CHECK: memref.alloc() : memref<13x13x16x1xf32, #hivm.address_space<ub>>
  // CHECK: memref.subview %[[EXPANDED:.*]][0, 0, 0, 0] [13, 13, 13, 1] [1, 1, 1, 1]
  // CHECK: memref<13x13x13xf32, strided<[208, 16, 1]>
  annotation.mark %0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<13x13x13xf32, #hivm.address_space<ub>>
  
  // CopyOp is inserted to copy from aligned buffer to flat buffer before reinterpret_cast
  // CHECK: hivm.hir.copy ins(%{{.*}} : memref<13x13x13xf32, strided<[208, 16, 1]>
  // CHECK: memref.reinterpret_cast
  // CHECK: hivm.hir.vexp
  // CHECK: hivm.hir.copy
  // CHECK-NOT: unrealized_conversion_cast
  
  %c0 = arith.constant 0 : index
  %1 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [169], strides: [13] : memref<13x13x13xf32, #hivm.address_space<ub>> to memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>
  
  hivm.hir.vexp ins(%1 : memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>) outs(%1 : memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>)
  return
}

// -----

// Test case: write to reinterpret_cast result after CopyOp insertion
// Tests the sync-back path: after the initial CopyOp, a load writes to the new buffer,
// requiring a sync CopyOp back to the original buffer to maintain data consistency.
// hivm.hir.load reads from GM and writes to the reinterpret_cast result (newDst).

// CHECK-LABEL: func @propagate_failure_write_after_copy
func.func @propagate_failure_write_after_copy(%arg0: memref<169xf32, #hivm.address_space<gm>>) {
  %0 = memref.alloc() : memref<13x13x13xf32, #hivm.address_space<ub>>
  
  // Force alignment expansion and trigger propagate failure
  // CHECK-DAG: memref.alloc() : memref<13x13x16x1xf32, #hivm.address_space<ub>>
  // CHECK-DAG: memref.subview
  // CHECK-DAG: memref<13x13x13xf32, strided<[208, 16, 1]>
  annotation.mark %0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<13x13x13xf32, #hivm.address_space<ub>>
  
  %c0 = arith.constant 0 : index
  
  // Initial CopyOp from aligned subview to flat alloc_0, then memref.reinterpret_cast
  // CHECK: hivm.hir.copy ins(%{{.*}} : memref<13x13x13xf32, strided<[208, 16, 1]>
  // CHECK: memref.reinterpret_cast
  
  %1 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [169], strides: [13] : memref<13x13x13xf32, #hivm.address_space<ub>> to memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>
  
  // Load writes to reinterpret_cast result (new buffer)
  // This triggers a sync CopyOp back to original buffer after the load
  // CHECK: hivm.hir.load
  // CHECK: hivm.hir.copy
  
  hivm.hir.load ins(%arg0 : memref<169xf32, #hivm.address_space<gm>>) outs(%1 : memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>)
  return
}

// -----

// Test case: multiple writes with vexp (reads AND writes to buffer)
// Tests sync-back for operations that both read and write to the new buffer.

// CHECK-LABEL: func @propagate_failure_vexp_write
func.func @propagate_failure_vexp_write() {
  %0 = memref.alloc() : memref<13x13x13xf32, #hivm.address_space<ub>>
  
  // Force alignment expansion
  // CHECK: memref.alloc() : memref<13x13x16x1xf32, #hivm.address_space<ub>>
  // CHECK: memref.subview
  annotation.mark %0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<13x13x13xf32, #hivm.address_space<ub>>
  
  %c0 = arith.constant 0 : index
  %1 = memref.reinterpret_cast %0 to offset: [%c0], sizes: [169], strides: [13] : memref<13x13x13xf32, #hivm.address_space<ub>> to memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>
  
  // vexp reads AND writes to reinterpret_cast (new buffer)
  // Multiple sync CopyOps should be inserted after vexp to sync back to src
  // CHECK: hivm.hir.copy
  // CHECK: memref.reinterpret_cast
  // CHECK: hivm.hir.vexp
  // CHECK: hivm.hir.copy
  // CHECK: hivm.hir.copy
  // CHECK: return
  
  hivm.hir.vexp ins(%1 : memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>) outs(%1 : memref<169xf32, strided<[13], offset: ?>, #hivm.address_space<ub>>)
  return
}

// -----

// Test case: simple stride alignment scenario that succeeds without CopyOps
// CHECK-LABEL: func @stride_align_success_no_copy
func.func @stride_align_success_no_copy() {
  // CHECK: memref.alloc() : memref<16x16x8xf32, #hivm.address_space<ub>>
  // CHECK: memref.subview
  %0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  annotation.mark %0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<16x16xf32, #hivm.address_space<ub>>
  
  // CHECK-NOT: hivm.hir.copy
  // CHECK: hivm.hir.vexp
  hivm.hir.vexp ins(%0 : memref<16x16xf32, #hivm.address_space<ub>>) outs(%0 : memref<16x16xf32, #hivm.address_space<ub>>)
  return
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @producer_fixpipe() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %cc = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    %ub = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<4x4xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe ins(%cc : memref<4x4xf32, #hivm.address_space<cc>>) outs(%ub : memref<4x4xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
    return
  }

  // CHECK-LABEL: func.func @consumer_materialized_copy
  // CHECK: %[[ALIGNED_A:.*]] = memref.alloc() : memref<4x8x1xf32, #hivm.address_space<ub>>
  // CHECK: %[[SUBVIEW_A:.*]] = memref.subview %[[ALIGNED_A]]
  // CHECK: %[[CONTIG_A:.*]] = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy ins(%[[SUBVIEW_A]] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%[[CONTIG_A]] : memref<4x4xf32, #hivm.address_space<ub>>)
  // CHECK: %[[COLLAPSED_A:.*]] = memref.collapse_shape %[[CONTIG_A]]
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
  // CHECK: hivm.hir.copy ins(%[[SUBVIEW_A]] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%[[CONTIG_A]] : memref<4x4xf32, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.copy ins(%[[COLLAPSED_A]] : memref<16xf32, #hivm.address_space<ub>>)
  func.func @consumer_materialized_copy() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ub = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<4x4xf32, #hivm.address_space<ub>>
    %collapsed = memref.collapse_shape %ub [[0, 1]] : memref<4x4xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
    hivm.hir.copy ins(%collapsed : memref<16xf32, #hivm.address_space<ub>>) outs(%dst : memref<16xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @producer_fixpipe_with_multiple_syncs() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %cc0 = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    %ub0 = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<4x4xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe ins(%cc0 : memref<4x4xf32, #hivm.address_space<cc>>) outs(%ub0 : memref<4x4xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
    %cc1 = memref.alloc() : memref<4x4xf32, #hivm.address_space<cc>>
    %ub1 = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<4x4xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe ins(%cc1 : memref<4x4xf32, #hivm.address_space<cc>>) outs(%ub1 : memref<4x4xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_V>] flag = 3
    return
  }

  // CHECK-LABEL: func.func @consumer_skip_non_matching_wait
  // CHECK: %[[SUBVIEW_B:.*]] = memref.subview
  // CHECK: %[[CONTIG_B:.*]] = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy ins(%[[SUBVIEW_B]] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%[[CONTIG_B]] : memref<4x4xf32, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_V>] flag = 3
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
  // CHECK: hivm.hir.copy ins(%[[SUBVIEW_B]] : memref<4x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%[[CONTIG_B]] : memref<4x4xf32, #hivm.address_space<ub>>)
  // CHECK: hivm.hir.copy
  func.func @consumer_skip_non_matching_wait() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %ub = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %ub {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<4x4xf32, #hivm.address_space<ub>>
    %collapsed = memref.collapse_shape %ub [[0, 1]] : memref<4x4xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_V>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
    hivm.hir.copy ins(%collapsed : memref<16xf32, #hivm.address_space<ub>>) outs(%dst : memref<16xf32, #hivm.address_space<ub>>)
    return
  }
}