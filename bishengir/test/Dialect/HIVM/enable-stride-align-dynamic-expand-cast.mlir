// RUN: bishengir-opt %s -hivm-enable-stride-align -split-input-file | FileCheck %s

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
// A shape-refining cast after a dynamic expand must propagate the aligned
// layout, rather than materialize a temporary with dynamic layout symbols.
// Check 32-byte row padding (4 * sizeof(i64)), unchanged logical extents,
// and direct use of the aligned window with no fallback or reshape chain.
// CHECK-LABEL: func.func @dynamic_expand_cast_alignment(
// CHECK-SAME: %[[M:[^:]+]]: index, %[[N:[^:]+]]: index)
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<1024x4x1xi64, #hivm.address_space<ub>>
// CHECK-NEXT: %[[TILE:.*]] = memref.subview %[[ALLOC]][0, 0, 0] [1024, 2, 1] [1, 1, 1]
// CHECK-SAME: to memref<1024x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>
// CHECK-NEXT: %[[WINDOW:.*]] = memref.subview %[[TILE]][0, 0] [%[[M]], %[[N]]] [1, 1]
// CHECK-SAME: to memref<?x?xi64, strided<[4, 1]>, #hivm.address_space<ub>>
// CHECK-NEXT: hivm.hir.copy ins(%[[WINDOW]] : memref<?x?xi64, strided<[4, 1]>, #hivm.address_space<ub>>) outs(%[[WINDOW]] : memref<?x?xi64, strided<[4, 1]>, #hivm.address_space<ub>>)
// CHECK-NEXT: return
func.func @dynamic_expand_cast_alignment(%m: index, %n: index) {
 %a = memref.alloc() : memref<1024x2xi64, #hivm.address_space<ub>>
 annotation.mark %a {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1024x2xi64, #hivm.address_space<ub>>
 %s = memref.subview %a[0, 0] [%m, %n] [1, 1] : memref<1024x2xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>
 %e = memref.expand_shape %s [[0, 1], [2]] output_shape [1, %m, %n] : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>> into memref<1x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>>
 %c = memref.cast %e : memref<1x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>> to memref<1x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>
 %r = memref.collapse_shape %c [[0, 1], [2]] : memref<1x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>> into memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>
 hivm.hir.copy ins(%r : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>) outs(%s : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>)
 return
}

}

// -----

// Exercise failure materialization directly. The pre-existing dim use lies
// between failed consumers in the conversion use-list. Shared storage preserves
// aliasing between consumers. Allocation sizing introduces
// additional dim queries; all must use the aligned source without skipping
// either consumer or its copy-out synchronization.
// CHECK-LABEL: func.func @multiple_failed_users
// CHECK-SAME: %[[SRC:.*]]: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[SIZE0:.*]] = memref.dim %[[SRC]],
// CHECK: %[[BUF0:.*]] = memref.alloc(%[[SIZE0]])
// CHECK: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[BUF0]] :
// CHECK: call @consume(%[[BUF0]])
// CHECK: hivm.hir.copy ins(%[[BUF0]] : {{.*}}) outs(%[[SRC]] :
// CHECK: %[[N:.*]] = memref.dim %[[SRC]],
// CHECK-NOT: memref.alloc
// CHECK: call @consume(%[[BUF0]])
// CHECK: hivm.hir.copy ins(%[[BUF0]] : {{.*}}) outs(%[[SRC]] :
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return %[[N]] : index
func.func private @consume(memref<?x2xi64, #hivm.address_space<ub>>)
func.func @multiple_failed_users(%src: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) -> index {
  %c0 = arith.constant 0 : index
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x2xi64, #hivm.address_space<ub>> {PropagateFailure}
  func.call @consume(%dst) {PropagateFailure} : (memref<?x2xi64, #hivm.address_space<ub>>) -> ()
  %n = memref.dim %dst, %c0 : memref<?x2xi64, #hivm.address_space<ub>>
  func.call @consume(%dst) {PropagateFailure} : (memref<?x2xi64, #hivm.address_space<ub>>) -> ()
  return %n : index
}

// -----

// Dynamic layout entries need symbol operands on each fallback allocation.
// CHECK-LABEL: func.func @fallback_dynamic_stride
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[SHARED_STRIDE:.*]] = memref.alloc({{.*}})[{{.+}}]
// CHECK: call @consume_dynamic_stride(%[[SHARED_STRIDE]])
// CHECK-NOT: memref.alloc
// CHECK: call @consume_dynamic_stride(%[[SHARED_STRIDE]])
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
func.func private @consume_dynamic_stride(memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>>)
func.func @fallback_dynamic_stride(%src: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) -> index {
  %c0 = arith.constant 0 : index
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>> {PropagateFailure}
  func.call @consume_dynamic_stride(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>>) -> ()
  %n = memref.dim %dst, %c0 : memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>>
  func.call @consume_dynamic_stride(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, 1]>, #hivm.address_space<ub>>) -> ()
  return %n : index
}


// -----

// Dynamic layout entries need symbol operands on each fallback allocation.
// CHECK-LABEL: func.func @fallback_dynamic_offset
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[SHARED_OFFSET:.*]] = memref.alloc({{.*}})[{{.+}}]
// CHECK: call @consume_dynamic_offset(%[[SHARED_OFFSET]])
// CHECK-NOT: memref.alloc
// CHECK: call @consume_dynamic_offset(%[[SHARED_OFFSET]])
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
func.func private @consume_dynamic_offset(memref<?x2xi64, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>)
func.func @fallback_dynamic_offset(%src: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) -> index {
  %c0 = arith.constant 0 : index
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[2, 1], offset: ?>, #hivm.address_space<ub>> {PropagateFailure}
  func.call @consume_dynamic_offset(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
  %n = memref.dim %dst, %c0 : memref<?x2xi64, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
  func.call @consume_dynamic_offset(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
  return %n : index
}


// -----

// Dynamic layout entries need symbol operands on each fallback allocation.
// CHECK-LABEL: func.func @fallback_dynamic_layout
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[SHARED_LAYOUT:.*]] = memref.alloc({{.*}})[{{.+}}]
// CHECK: call @consume_dynamic_layout(%[[SHARED_LAYOUT]])
// CHECK-NOT: memref.alloc
// CHECK: call @consume_dynamic_layout(%[[SHARED_LAYOUT]])
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
func.func private @consume_dynamic_layout(memref<?x2xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>)
func.func @fallback_dynamic_layout(%src: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) -> index {
  %c0 = arith.constant 0 : index
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>> {PropagateFailure}
  func.call @consume_dynamic_layout(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
  %n = memref.dim %dst, %c0 : memref<?x2xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
  func.call @consume_dynamic_layout(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
  return %n : index
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {

// Translate the non-unit outer stride using the expand reassociation:
// 2048 / 2 * 4 = 4096. An external interface then needs its declared layout.
// CHECK-LABEL: func.func @nonunit_outer_stride_refinement
// CHECK: %[[EXPAND:.*]] = memref.expand_shape
// CHECK-SAME: strided<[?, 4, 1]>
// CHECK: %[[CAST:.*]] = memref.cast %[[EXPAND]]
// CHECK-SAME: to memref<2x?x?xi64, strided<[4096, 4, 1]>, #hivm.address_space<ub>>
// CHECK: %[[BUF:.*]] = memref.alloc({{.*}}) : memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>
// CHECK: hivm.hir.copy ins(%[[CAST]] : {{.*}}) outs(%[[BUF]] :
// CHECK: call @consume_nonunit(%[[BUF]])
// CHECK: hivm.hir.copy ins(%[[BUF]] : {{.*}}) outs(%[[CAST]] :
// CHECK: return
func.func private @consume_nonunit(memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>)
func.func @nonunit_outer_stride_refinement(%rows: index, %m: index, %n: index) {
  %a = memref.alloc() : memref<2048x2xi64, #hivm.address_space<ub>>
  annotation.mark %a {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2048x2xi64, #hivm.address_space<ub>>
  %s = memref.subview %a[0, 0] [%rows, %n] [1, 1] : memref<2048x2xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>
  %e = memref.expand_shape %s [[0, 1], [2]] output_shape [2, %m, %n] : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>> into memref<2x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>>
  %c = memref.cast %e : memref<2x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>> to memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>
  func.call @consume_nonunit(%c) : (memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>) -> ()
  return
}

// A consumer that accepts the aligned layout needs no fallback allocation.
// CHECK-LABEL: func.func @nonunit_outer_stride_direct
// CHECK: memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK: %[[DIRECT_EXPAND:.*]] = memref.expand_shape
// Canonicalization can fold the cast into a layout-polymorphic copy.
// CHECK-NOT: memref.alloc
// CHECK-NOT: unrealized_conversion_cast
// CHECK: hivm.hir.copy ins(%[[DIRECT_EXPAND]] : {{.*}}) outs(%[[DIRECT_EXPAND]] :
// CHECK-NEXT: return
func.func @nonunit_outer_stride_direct(%rows: index, %m: index, %n: index) {
  %a = memref.alloc() : memref<2048x2xi64, #hivm.address_space<ub>>
  annotation.mark %a {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2048x2xi64, #hivm.address_space<ub>>
  %s = memref.subview %a[0, 0] [%rows, %n] [1, 1] : memref<2048x2xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>
  %e = memref.expand_shape %s [[0, 1], [2]] output_shape [2, %m, %n] : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>> into memref<2x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>>
  %c = memref.cast %e : memref<2x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>> to memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%c : memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>) outs(%c : memref<2x?x?xi64, strided<[2048, 2, 1]>, #hivm.address_space<ub>>)
  return
}

}

// -----

// Delay dynamic fallback storage until consumption, after a source write and
// a dimension query. Sizes must come from the real source, not the conversion.
// CHECK-LABEL: func.func @delayed_dynamic_fallback
// CHECK-SAME: %[[SRC:[^:]+]]:
// CHECK-NOT: memref.alloc
// CHECK: call @update_aligned(%[[SRC]])
// CHECK: %[[SIZE:.*]] = memref.dim %[[SRC]],
// CHECK: %[[ALLOC_SIZE:.*]] = memref.dim %[[SRC]],
// CHECK: %[[BUF:.*]] = memref.alloc(%[[ALLOC_SIZE]])[{{.+}}]
// CHECK-NEXT: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[BUF]] :
// CHECK-NEXT: call @consume_delayed(%[[BUF]])
// CHECK-NEXT: hivm.hir.copy ins(%[[BUF]] : {{.*}}) outs(%[[SRC]] :
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return %[[SIZE]] : index
func.func private @update_aligned(memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>)
func.func private @consume_delayed(memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>)
func.func @delayed_dynamic_fallback(%src: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) -> index {
  %c0 = arith.constant 0 : index
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> {PropagateFailure}
  func.call @update_aligned(%src) : (memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) -> ()
  %size = memref.dim %dst, %c0 : memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
  func.call @consume_delayed(%dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
  return %size : index
}

// -----

// A shared dynamic-layout allocation stays outside the loop, but TCB data is
// seeded inside it after the completion wait. Duplicate operands on one call
// need one seed; the second call needs its own seed. Each write is copied back.
// CHECK-LABEL: func.func @tcb_dynamic_fallback_loop
// CHECK-SAME: %[[SRC:[^:]+]]:
// CHECK: %[[SIZE:.*]] = memref.dim %[[SRC]],
// CHECK: %[[BUF:.*]] = memref.alloc(%[[SIZE]])[{{.+}}]
// CHECK-NOT: hivm.hir.copy
// CHECK: scf.for
// CHECK-NOT: memref.alloc
// CHECK: hivm.hir.sync_block_wait
// CHECK-NEXT: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[BUF]] :
// CHECK-NEXT: call @consume_tcb_pair(%[[BUF]], %[[BUF]])
// CHECK-NEXT: hivm.hir.copy ins(%[[BUF]] : {{.*}}) outs(%[[SRC]] :
// CHECK-NEXT: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[BUF]] :
// CHECK-NEXT: call @consume_tcb_pair(%[[BUF]], %[[BUF]])
// CHECK-NEXT: hivm.hir.copy ins(%[[BUF]] : {{.*}}) outs(%[[SRC]] :
// CHECK-NEXT: }
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NEXT: return
func.func private @consume_tcb_pair(memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>)
func.func @tcb_dynamic_fallback_loop(%src: memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>, %iterations: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  annotation.mark %src {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>
  %dst = builtin.unrealized_conversion_cast %src : memref<?x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> {PropagateFailure}
  scf.for %i = %c0 to %iterations step %c1 {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 0
    func.call @consume_tcb_pair(%dst, %dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
    func.call @consume_tcb_pair(%dst, %dst) {PropagateFailure} : (memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, memref<?x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
  }
  return
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
// The source's leading extent is one even though the cast makes it dynamic.
// Its stride refinement is irrelevant and needs no expand-based translation
// through the intervening subview. The non-unit row stride must still align.
// CHECK-LABEL: func.func @source_unit_dim_dynamic_result
// CHECK: memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK-NOT: unrealized_conversion_cast
// CHECK: hivm.hir.copy ins({{.*}}strided<[?, 8, 1]>{{.*}}) outs(
// CHECK-NEXT: return
func.func @source_unit_dim_dynamic_result(%m: index, %n: index, %m2: index, %n2: index) {
  %a = memref.alloc() : memref<1024x2xi64, #hivm.address_space<ub>>
  annotation.mark %a {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1024x2xi64, #hivm.address_space<ub>>
  %s = memref.subview %a[0, 0] [%m, %n] [1, 1] : memref<1024x2xi64, #hivm.address_space<ub>> to memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>>
  %e = memref.expand_shape %s [[0, 1], [2]] output_shape [1, %m, %n] : memref<?x?xi64, strided<[2, 1]>, #hivm.address_space<ub>> into memref<1x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>>
  %sub = memref.subview %e[0, 0, 0] [1, %m2, %n2] [1, 2, 1] : memref<1x?x?xi64, strided<[?, 2, 1]>, #hivm.address_space<ub>> to memref<1x?x?xi64, strided<[?, 4, 1]>, #hivm.address_space<ub>>
  %c = memref.cast %sub : memref<1x?x?xi64, strided<[?, 4, 1]>, #hivm.address_space<ub>> to memref<?x?x?xi64, strided<[8, 4, 1]>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%c : memref<?x?x?xi64, strided<[8, 4, 1]>, #hivm.address_space<ub>>) outs(%c : memref<?x?x?xi64, strided<[8, 4, 1]>, #hivm.address_space<ub>>)
  return
}
}

// -----

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
// Remaining unmarked conversions reach the regbase materializer. Static
// extents do not remove the need for dynamic stride and offset symbols.
// CHECK-LABEL: func.func @regbase_static_shape_dynamic_layout
// CHECK-SAME: %[[SRC:[^:]+]]:
// CHECK-NOT: memref.dim
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[BUF:.*]] = memref.alloc()[%[[ZERO]], %{{.*}}] : memref<4x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK-NEXT: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[BUF]] :
// CHECK-NEXT: call @consume_regbase_static(%[[BUF]])
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NEXT: return
func.func private @consume_regbase_static(memref<4x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>)
func.func @regbase_static_shape_dynamic_layout(%src: memref<4x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) {
  %dst = builtin.unrealized_conversion_cast %src : memref<4x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<4x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
  func.call @consume_regbase_static(%dst) : (memref<4x2xi64, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
  return
}
}
