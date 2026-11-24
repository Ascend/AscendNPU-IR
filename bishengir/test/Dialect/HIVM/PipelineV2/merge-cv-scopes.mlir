// RUN: bishengir-opt -merge-cv-scopes -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @merge_carried_pair
// CHECK:         scf.for
// CHECK:           %[[MERGED:.*]] = scope.scope : () -> f32 {
// CHECK:             %[[ADD:.*]] = arith.addf
// CHECK:             %[[MUL:.*]] = arith.mulf %[[ADD]], %[[ADD]]
// CHECK:             scope.return %[[MUL]]
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:       cv_merged_scope
// CHECK:           scf.yield %[[MERGED]]
func.func @merge_carried_pair(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %1 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %1 : f32
  } {hivm.cv_pipelined_loop}
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @merge_mixed_cores
// CHECK:         scope.scope : () -> f32 {
// CHECK:           scope.scope : () -> f32 {
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:           scope.scope : () -> f32 {
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:           scope.scope : () -> f32 {
// CHECK:           } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:           scope.return
// CHECK:         } {cv_merged_scope}
// CHECK-NOT:     hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>
func.func @merge_mixed_cores(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %1 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
    %2 = scope.scope : () -> f32 {
      %c = arith.subf %1, %1 : f32
      scope.return %c : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %2 : f32
  } {hivm.cv_pipelined_loop}
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @merge_escaping_results
// CHECK:         %[[MERGED:.*]]:2 = scope.scope : () -> (f32, f32) {
// CHECK:           %[[A:.*]] = arith.addf
// CHECK:           %[[B:.*]] = arith.mulf %[[A]], %[[A]]
// CHECK:           scope.return %[[A]], %[[B]]
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:     cv_merged_scope
// CHECK:         scf.yield %[[MERGED]]#1, %[[MERGED]]#0
func.func @merge_escaping_results(%init: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r:2 = scf.for %i = %c0 to %c16 step %c1
      iter_args(%acc = %init, %other = %init) -> (f32, f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %1 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %1, %0 : f32, f32
  } {hivm.cv_pipelined_loop}
  return %r#0, %r#1 : f32, f32
}

// -----

// Rule 1 moves the independent consumer after the recurrence group; Rule 4
// subsequently merges it into the collapsed VECTOR scope.
// CHECK-LABEL: func.func @merge_consumer_moved_after
// CHECK:         %[[MERGED:.*]] = scope.scope : () -> f32 {
// CHECK:           %[[A:.*]] = arith.addf
// CHECK:           %[[B:.*]] = arith.mulf %[[A]], %[[A]]
// CHECK-NEXT:      {{.*}} = arith.negf %[[A]]
// CHECK-NEXT:      scope.return %[[B]]
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:     cv_merged_scope
// CHECK-NEXT:    scf.yield %[[MERGED]]
func.func @merge_consumer_moved_after(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %n = arith.negf %0 : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %2 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %2 : f32
  } {hivm.cv_pipelined_loop}
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @merge_no_recurrence
// CHECK-NOT:     cv_merged_scope
func.func @merge_no_recurrence(%a: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %0 = scope.scope : () -> f32 {
      %x = arith.addf %a, %a : f32
      scope.return %x : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %y = arith.mulf %0, %0 : f32
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  } {hivm.cv_pipelined_loop}
  return
}

// -----

// CHECK-LABEL: func.func @merge_not_outlined
// CHECK-NOT:     cv_merged_scope
func.func @merge_not_outlined(%init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %init) -> (f32) {
    %0 = scope.scope : () -> f32 {
      %a = arith.addf %acc, %acc : f32
      scope.return %a : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %1 = scope.scope : () -> f32 {
      %b = arith.mulf %0, %0 : f32
      scope.return %b : f32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %1 : f32
  }
  return %r : f32
}

// -----

// Verify top-level backward merging from `copy -> cbuf`:
// - Merges VECTOR (`S2`, `S3`, `S5`) and CUBE_AND_VECTOR (`S1`) into one VECTOR scope.
// - Includes `S2` in the merged scope but stops before its CUBE dependency `S0`.
// - Rule 4 also merges intervening VECTOR consumer `S4`, preserving its uses.
// - Stops at CUBE `fixpipe -> ub` scope `S6` because its dependency (`S5`) is VECTOR.
//
// CHECK-LABEL: func.func @merge_backward_from_copy_cbuf_and_preserve_dominance
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}, %[[CNT:.*]] = {{.*}}) -> (tensor<16x16xf32>, i32) : i32 {
// CHECK:   %[[S0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.mmadL1
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[MERGED:.*]]:3 = scope.scope : () -> (i32, tensor<1x1x16x16xf16>, tensor<16x16xf32>) {
// CHECK:     %[[NEXT_CNT:.*]] = arith.addi %[[CNT]], {{.*}} : i32
// CHECK:     %[[VEXP:.*]] = hivm.hir.vexp ins(%[[S0]] : tensor<16x16xf32>)
// CHECK:     %[[VADD:.*]] = hivm.hir.vadd ins(%[[VEXP]], {{.*}} : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     hivm.hir.vcast ins(%[[VADD]] : tensor<16x16xf32>)
// CHECK:     hivm.hir.copy
// CHECK:     %[[RELU:.*]] = hivm.hir.vrelu ins(%[[VADD]] : tensor<16x16xf32>)
// CHECK:     scope.return %[[NEXT_CNT]], {{.*}}, %[[RELU]] : i32, tensor<1x1x16x16xf16>, tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[S6:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.mmadL1 ins(%[[MERGED]]#1, {{.*}})
// CHECK:     hivm.hir.fixpipe
// CHECK:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   %[[S7:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     %[[OUT:.*]] = hivm.hir.vadd ins(%[[MERGED]]#2, %[[S6]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:     scope.return %[[OUT]] : tensor<16x16xf32>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   scf.yield %[[S7]], %[[MERGED]]#0 : tensor<16x16xf32>, i32
// CHECK: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @merge_backward_from_copy_cbuf_and_preserve_dominance(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %ub: i32) -> (tensor<16x16xf32>, i32)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res:2 = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init, %cnt = %c0) -> (tensor<16x16xf32>, i32) : i32 {
      %s0 = scope.scope : () -> tensor<16x16xf32> {
        %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot0 = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s1 = scope.scope : () -> i32 {
        %next_cnt = arith.addi %cnt, %c1 : i32
        scope.return %next_cnt : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}

      %s2 = scope.scope : () -> tensor<16x16xf32> {
        %p_init = tensor.empty() : tensor<16x16xf32>
        %p = hivm.hir.vexp ins(%s0 : tensor<16x16xf32>) outs(%p_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %p : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s3 = scope.scope : () -> tensor<16x16xf32> {
        %idx_keep = arith.addi %s1, %c0 : i32
        %v_init = tensor.empty() : tensor<16x16xf32>
        %v = hivm.hir.vadd ins(%s2, %init : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %v : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s4 = scope.scope : () -> tensor<16x16xf32> {
        %r_init = tensor.empty() : tensor<16x16xf32>
        %r = hivm.hir.vrelu ins(%s3 : tensor<16x16xf32>) outs(%r_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %r : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s5 = scope.scope : () -> tensor<1x1x16x16xf16> {
        %f16_init = tensor.empty() : tensor<16x16xf16>
        %f16 = hivm.hir.vcast ins(%s3 : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %t_cbuf : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s6 = scope.scope : () -> tensor<16x16xf32> {
        %dot1_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot1 = hivm.hir.mmadL1 ins(%s5, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub1_cast = memref.memory_space_cast %ub1 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t1 = bufferization.to_tensor %ub1_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot1 : tensor<1x1x16x16xf32>) outs(%ub1 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t1 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s7 = scope.scope : () -> tensor<16x16xf32> {
        %out_init = tensor.empty() : tensor<16x16xf32>
        %out = hivm.hir.vadd ins(%s4, %s6 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %out : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %s7, %s1 : tensor<16x16xf32>, i32
    } {hivm.cv_pipelined_loop}
    return %res#0, %res#1 : tensor<16x16xf32>, i32
  }
}

// -----

// Verify Rule 1 (loop-carried dependence grouping) followed by Rule 2
// (cross-core transfer backward merging):
// - Rule 1 wraps `S3..S7` (connected via `%acc` recurrence) into `{cv_merged_scope}`,
//   leaving `S0`, `S1`, `S2` outside in the `scf.for` body.
// - Rule 2 merges `S3`, `S4`, and `S5` (`VECTOR`) INSIDE `{cv_merged_scope}`,
//   without pulling `S1` or `S2` from outside `{cv_merged_scope}`, while keeping
//   `S6` (`CUBE`) and `S7` (`VECTOR`, depends on `S6`) separate.
//
// CHECK-LABEL: func.func @lcd_grouping_then_cross_core_backward_merge
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}, %[[CNT:.*]] = {{.*}}) -> (tensor<16x16xf32>, i32) : i32 {
// CHECK-NEXT:   %[[S0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          hivm.hir.mmadL1
// CHECK:          hivm.hir.fixpipe
// CHECK:          scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:   %[[S1:.*]] = scope.scope : () -> i32 {
// CHECK:          %[[NEXT_CNT:.*]] = arith.addi %[[CNT]], {{.*}} : i32
// CHECK-NEXT:     scope.return %[[NEXT_CNT]] : i32
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}
// CHECK-NEXT:   %[[S2:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          %[[VEXP:.*]] = hivm.hir.vexp ins(%[[S0]] : tensor<16x16xf32>)
// CHECK-NEXT:     scope.return %[[VEXP]] : tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:   %[[LCD:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK-NEXT:     %[[INNER_MERGED:.*]]:2 = scope.scope : () -> (tensor<16x16xf32>, tensor<1x1x16x16xf16>) {
// CHECK:            arith.addi %[[S1]]
// CHECK:            %[[VADD:.*]] = hivm.hir.vadd ins(%[[S2]], %[[ACC]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:            %[[RELU:.*]] = hivm.hir.vrelu ins(%[[VADD]] : tensor<16x16xf32>)
// CHECK:            hivm.hir.vcast ins(%[[VADD]] : tensor<16x16xf32>)
// CHECK:            hivm.hir.copy
// CHECK-NEXT:       scope.return %[[RELU]], {{.*}} : tensor<16x16xf32>, tensor<1x1x16x16xf16>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:     %[[S6:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:            hivm.hir.mmadL1 ins(%[[INNER_MERGED]]#1, {{.*}})
// CHECK:            hivm.hir.fixpipe
// CHECK-NEXT:       scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:     %[[S7:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:            %[[OUT:.*]] = hivm.hir.vadd ins(%[[INNER_MERGED]]#0, %[[S6]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NEXT:       scope.return %[[OUT]] : tensor<16x16xf32>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:     scope.return %[[S7]] : tensor<16x16xf32>
// CHECK-NEXT:   } {cv_merged_scope}
// CHECK-NEXT:   scf.yield %[[LCD]], %[[S1]] : tensor<16x16xf32>, i32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @lcd_grouping_then_cross_core_backward_merge(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %ub: i32) -> (tensor<16x16xf32>, i32)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res:2 = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init, %cnt = %c0) -> (tensor<16x16xf32>, i32) : i32 {
      %s0 = scope.scope : () -> tensor<16x16xf32> {
        %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot0 = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s1 = scope.scope : () -> i32 {
        %next_cnt = arith.addi %cnt, %c1 : i32
        scope.return %next_cnt : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}

      %s2 = scope.scope : () -> tensor<16x16xf32> {
        %p_init = tensor.empty() : tensor<16x16xf32>
        %p = hivm.hir.vexp ins(%s0 : tensor<16x16xf32>) outs(%p_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %p : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s3 = scope.scope : () -> tensor<16x16xf32> {
        %idx_keep = arith.addi %s1, %c0 : i32
        %v_init = tensor.empty() : tensor<16x16xf32>
        %v = hivm.hir.vadd ins(%s2, %acc : tensor<16x16xf32>, tensor<16x16xf32>) outs(%v_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %v : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s4 = scope.scope : () -> tensor<16x16xf32> {
        %r_init = tensor.empty() : tensor<16x16xf32>
        %r = hivm.hir.vrelu ins(%s3 : tensor<16x16xf32>) outs(%r_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %r : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s5 = scope.scope : () -> tensor<1x1x16x16xf16> {
        %f16_init = tensor.empty() : tensor<16x16xf16>
        %f16 = hivm.hir.vcast ins(%s3 : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %t_cbuf : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %s6 = scope.scope : () -> tensor<16x16xf32> {
        %dot1_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot1 = hivm.hir.mmadL1 ins(%s5, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub1_cast = memref.memory_space_cast %ub1 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t1 = bufferization.to_tensor %ub1_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot1 : tensor<1x1x16x16xf32>) outs(%ub1 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t1 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s7 = scope.scope : () -> tensor<16x16xf32> {
        %out_init = tensor.empty() : tensor<16x16xf32>
        %out = hivm.hir.vadd ins(%s4, %s6 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %out : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %s7, %s1 : tensor<16x16xf32>, i32
    } {hivm.cv_pipelined_loop}
    return %res#0, %res#1 : tensor<16x16xf32>, i32
  }
}

// -----

// Rule 3 stops backward traversal at an incompatible dependency. Rule 4 can
// still merge the CUBE pair because the VECTOR input does not lie on a path
// between them; it remains outside and dominates the merged scope.
//
// CHECK-LABEL: func.func @stop_when_any_dep_incompatible
// CHECK: scf.for
// CHECK:   %[[V0:.*]] = scope.scope : () -> tensor<1x1x16x16xf16>
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:   %[[C1:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:     hivm.hir.nd2nz
// CHECK:     %[[C0:.*]] = bufferization.to_tensor
// CHECK:     hivm.hir.mmadL1 ins(%[[C0]], %[[V0]], {{.*}})
// CHECK:     hivm.hir.fixpipe
// CHECK:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK:   scf.yield %[[C1]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @stop_when_any_dep_incompatible(
      %k_gm: memref<16x16xf16>,
      %x: tensor<16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %c_dep = scope.scope : () -> tensor<1x1x16x16xf16> {
        %k_alloc = memref.alloc() : memref<1x1x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%k_gm : memref<16x16xf16>) outs(%k_alloc : memref<1x1x16x16xf16>)
        %k = bufferization.to_tensor %k_alloc restrict writable : memref<1x1x16x16xf16>
        scope.return %k : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %v_dep = scope.scope : () -> tensor<1x1x16x16xf16> {
        %x_4d = tensor.expand_shape %x [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        scope.return %x_4d : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %seed = scope.scope : () -> tensor<16x16xf32> {
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%c_dep, %v_dep, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub_alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub_cast = memref.memory_space_cast %ub_alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t = bufferization.to_tensor %ub_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      scf.yield %seed : tensor<16x16xf32>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<16x16xf32>
  }
}

// -----

// Verify backward merging from `fixpipe -> ub` across CUBE and CUBE_OR_VECTOR.
//
// CHECK-LABEL: func.func @merge_cube_backward_from_fixpipe_ub
// CHECK: scf.for
// CHECK-NEXT: %[[MERGED_CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        hivm.hir.nd2nz
// CHECK:        hivm.hir.mmadL1
// CHECK:        hivm.hir.fixpipe
// CHECK:        scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.yield %[[MERGED_CUBE]] : tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @merge_cube_backward_from_fixpipe_ub(
      %a: tensor<1x1x16x16xf16>,
      %k_gm: memref<16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %s0 = scope.scope : () -> index {
        %m = arith.addi %c16, %c16 : index
        %half = arith.divui %m, %c16 : index
        %k_dim = arith.muli %half, %c16 : index
        scope.return %c16 : index
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_OR_VECTOR>}

      %s1 = scope.scope : () -> tensor<1x1x16x16xf32> {
        %k_alloc = memref.alloc() : memref<1x1x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%k_gm : memref<16x16xf16>) outs(%k_alloc : memref<1x1x16x16xf16>)
        %k = bufferization.to_tensor %k_alloc restrict writable : memref<1x1x16x16xf16>
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %k, %true, %s0, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        scope.return %dot : tensor<1x1x16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s2 = scope.scope : () -> tensor<16x16xf32> {
        %ub_alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub_cast = memref.memory_space_cast %ub_alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t = bufferization.to_tensor %ub_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%s1 : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      scf.yield %s2 : tensor<16x16xf32>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<16x16xf32>
  }
}

// -----

// Verify that scopes without `hivm.loop_core_type` are neither used as merge
// seeds nor added into `mergeSet` as dependencies.
//
// CHECK-LABEL: func.func @skip_scopes_without_core_type
// CHECK: scf.for
// CHECK-NEXT: %[[UNTYPED_DEP:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        hivm.hir.vabs
// CHECK-NEXT:   scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT: }
// CHECK-NOT:  hivm.loop_core_type
// CHECK-NEXT: %[[MERGED_VEC:.*]]:2 = scope.scope : () -> (tensor<1x1x16x16xf16>, tensor<16x16xf16>) {
// CHECK:        %[[VADD:.*]] = hivm.hir.vadd ins(%[[UNTYPED_DEP]], {{.*}})
// CHECK:        hivm.hir.vcast ins(%[[VADD]] : tensor<16x16xf32>)
// CHECK:        hivm.hir.copy
// CHECK:        hivm.hir.vrelu
// CHECK-NEXT:   scope.return {{.*}} : tensor<1x1x16x16xf16>, tensor<16x16xf16>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[UNTYPED_SEED:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:        tensor.expand_shape %[[MERGED_VEC]]#1
// CHECK:        hivm.hir.copy
// CHECK-NEXT:   scope.return {{.*}} : tensor<1x1x16x16xf16>
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[MERGED_VEC]]#0, %[[UNTYPED_SEED]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @skip_scopes_without_core_type(
      %x: tensor<16x16xf32>,
      %y: tensor<16x16xf16>,
      %ub: i32) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init0 = tensor.empty() : tensor<1x1x16x16xf16>
    %init1 = tensor.empty() : tensor<1x1x16x16xf16>
    %res:2 = scf.for %i = %c0 to %ub step %c1 iter_args(%acc0 = %init0, %acc1 = %init1) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) : i32 {
      %untyped_dep = scope.scope : () -> tensor<16x16xf32> {
        %abs_init = tensor.empty() : tensor<16x16xf32>
        %abs = hivm.hir.vabs ins(%x : tensor<16x16xf32>) outs(%abs_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %abs : tensor<16x16xf32>
      }

      %vec_dep = scope.scope : () -> tensor<16x16xf32> {
        %add_init = tensor.empty() : tensor<16x16xf32>
        %add = hivm.hir.vadd ins(%untyped_dep, %x : tensor<16x16xf32>, tensor<16x16xf32>) outs(%add_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %add : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %typed_seed = scope.scope : () -> tensor<1x1x16x16xf16> {
        %f16_init = tensor.empty() : tensor<16x16xf16>
        %f16 = hivm.hir.vcast ins(%vec_dep : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %t_cbuf : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %pre_untyped = scope.scope : () -> tensor<16x16xf16> {
        %r_init = tensor.empty() : tensor<16x16xf16>
        %r = hivm.hir.vrelu ins(%y : tensor<16x16xf16>) outs(%r_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        scope.return %r : tensor<16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %untyped_seed = scope.scope : () -> tensor<1x1x16x16xf16> {
        %y_4d = tensor.expand_shape %pre_untyped [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf2 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf2 = bufferization.to_tensor %cbuf2 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%y_4d : tensor<1x1x16x16xf16>) outs(%cbuf2 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %t_cbuf2 : tensor<1x1x16x16xf16>
      }

      scf.yield %typed_seed, %untyped_seed : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
    } {hivm.cv_pipelined_loop}
    return %res#0, %res#1 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
  }
}

// -----

// Verify that when a CUBE_OR_VECTOR seed depends on both a CUBE scope and a
// VECTOR scope, `resolveMergedCoreType` returns failure() cleanly (instead of
// asserting) once `targetCore` narrows to CUBE on the first dependency.
//
// CHECK-LABEL: func.func @conflicting_deps_on_cube_or_vector_seed
// CHECK: scf.for
// CHECK-NEXT: %[[V_DEP:.*]] = scope.scope : () -> index {
// CHECK:        scope.return {{.*}} : index
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[MERGED_CUBE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        hivm.hir.mmadL1
// CHECK:        hivm.hir.fixpipe
// CHECK:        arith.addi %[[V_DEP]]
// CHECK-NEXT:   scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT: scf.yield %[[MERGED_CUBE]] : tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @conflicting_deps_on_cube_or_vector_seed(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %c_dep = scope.scope : () -> tensor<1x1x16x16xf32> {
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        scope.return %dot : tensor<1x1x16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %v_dep = scope.scope : () -> index {
        %idx = arith.addi %c16, %c16 : index
        scope.return %idx : index
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %seed = scope.scope : () -> tensor<16x16xf32> {
        %ub_alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub_cast = memref.memory_space_cast %ub_alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t = bufferization.to_tensor %ub_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%c_dep : tensor<1x1x16x16xf32>) outs(%ub_alloc : memref<16x16xf32, #hivm.address_space<ub>>)
        %keep_v = arith.addi %v_dep, %c16 : index
        scope.return %t : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_OR_VECTOR>}

      scf.yield %seed : tensor<16x16xf32>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<16x16xf32>
  }
}

// -----

// Verify that scopes are only merged when they reside in the same parent block:
// - `%outer_dep` (in `scf.for` body) is NOT merged into `%then_seed` or `%else_seed`.
// - `%then_dep` and `%then_seed` (both in `scf.if` then-block) DO merge together.
// - `%else_seed` (in `scf.if` else-block) stays in the else-block unmerged with then-block scopes.
//
// CHECK-LABEL: func.func @only_merge_scopes_in_same_parent_block
// CHECK: scf.for
// CHECK-NEXT: %[[OUTER_DEP:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:        hivm.hir.vexp
// CHECK-NEXT:   scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT: %[[IF_RES:.*]] = scf.if %{{.*}} -> (tensor<1x1x16x16xf16>) {
// CHECK-NEXT:   %[[MERGED_THEN:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:          %[[VADD:.*]] = hivm.hir.vadd ins(%[[OUTER_DEP]], {{.*}})
// CHECK:          hivm.hir.vcast ins(%[[VADD]] : tensor<16x16xf32>)
// CHECK:          hivm.hir.copy
// CHECK-NEXT:     scope.return {{.*}} : tensor<1x1x16x16xf16>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:   scf.yield %[[MERGED_THEN]] : tensor<1x1x16x16xf16>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[ELSE_SCOPE:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:          hivm.hir.vcast ins(%[[OUTER_DEP]] : tensor<16x16xf32>)
// CHECK:          hivm.hir.copy
// CHECK-NEXT:     scope.return {{.*}} : tensor<1x1x16x16xf16>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:   scf.yield %[[ELSE_SCOPE]] : tensor<1x1x16x16xf16>
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield %[[IF_RES]] : tensor<1x1x16x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @only_merge_scopes_in_same_parent_block(
      %x: tensor<16x16xf32>,
      %cond: i1,
      %ub: i32) -> tensor<1x1x16x16xf16>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = tensor.empty() : tensor<1x1x16x16xf16>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<1x1x16x16xf16>) : i32 {
      %outer_dep = scope.scope : () -> tensor<16x16xf32> {
        %e_init = tensor.empty() : tensor<16x16xf32>
        %e = hivm.hir.vexp ins(%x : tensor<16x16xf32>) outs(%e_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %e : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %if_res = scf.if %cond -> (tensor<1x1x16x16xf16>) {
        %then_dep = scope.scope : () -> tensor<16x16xf32> {
          %a_init = tensor.empty() : tensor<16x16xf32>
          %a = hivm.hir.vadd ins(%outer_dep, %x : tensor<16x16xf32>, tensor<16x16xf32>) outs(%a_init : tensor<16x16xf32>) -> tensor<16x16xf32>
          scope.return %a : tensor<16x16xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

        %then_seed = scope.scope : () -> tensor<1x1x16x16xf16> {
          %f16_init = tensor.empty() : tensor<16x16xf16>
          %f16 = hivm.hir.vcast ins(%then_dep : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
          %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
          %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
          %t_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
          scope.return %t_cbuf : tensor<1x1x16x16xf16>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

        scf.yield %then_seed : tensor<1x1x16x16xf16>
      } else {
        %else_seed = scope.scope : () -> tensor<1x1x16x16xf16> {
          %f16_init = tensor.empty() : tensor<16x16xf16>
          %f16 = hivm.hir.vcast ins(%outer_dep : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
          %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
          %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
          %t_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
          scope.return %t_cbuf : tensor<1x1x16x16xf16>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

        scf.yield %else_seed : tensor<1x1x16x16xf16>
      }

      scf.yield %if_res : tensor<1x1x16x16xf16>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<1x1x16x16xf16>
  }
}

// -----

// Verify that when Rule 1 wraps same-core LCD scopes (`%m_new` and `%alpha`)
// into `{cv_merged_scope}`, Rule 2 merges them and collapses `{cv_merged_scope}`
// into a single VECTOR scope, allowing Rule 3 (`copy -> cbuf` backward merge)
// to merge the downstream `copy -> cbuf` scope (`%p_copy`) with the collapsed
// LCD scope (`%m_new` + `%alpha`) and its upstream VECTOR dependency (`%s_mask`).
//
// CHECK-LABEL: func.func @lcd_collapse_then_cross_core_backward_merge
// CHECK: scf.for {{.*}} iter_args(%[[M_PREV:.*]] = {{.*}}) -> (tensor<16x16xf32>) : i32 {
// CHECK-NEXT:   %[[CUBE_S:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          hivm.hir.mmadL1
// CHECK:          hivm.hir.fixpipe
// CHECK-NEXT:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:   %[[VEC_MERGED:.*]]:3 = scope.scope : () -> (tensor<16x16xf32>, tensor<16x16xf32>, tensor<1x1x16x16xf16>) {
// CHECK:          %[[MASK:.*]] = hivm.hir.vexp ins(%[[CUBE_S]] : tensor<16x16xf32>)
// CHECK:          %[[M_NEW:.*]] = hivm.hir.vmax ins(%[[M_PREV]], %[[MASK]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:          %[[ALPHA:.*]] = hivm.hir.vsub ins(%[[M_PREV]], %[[M_NEW]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:          hivm.hir.vcast ins(%[[M_NEW]] : tensor<16x16xf32>)
// CHECK:          hivm.hir.copy
// CHECK-NEXT:     scope.return %[[M_NEW]], %[[ALPHA]], {{.*}} : tensor<16x16xf32>, tensor<16x16xf32>, tensor<1x1x16x16xf16>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:    cv_merged_scope
// CHECK-NEXT:   %[[OUT_SCOPE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          hivm.hir.mmadL1 ins(%[[VEC_MERGED]]#2, {{.*}})
// CHECK:          hivm.hir.fixpipe
// CHECK-NEXT:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:   %[[FINAL_VEC:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          hivm.hir.vadd ins(%[[VEC_MERGED]]#1, %[[OUT_SCOPE]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NEXT:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:   scf.yield %[[VEC_MERGED]]#0 : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @lcd_collapse_then_cross_core_backward_merge(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%m_prev = %init) -> (tensor<16x16xf32>) : i32 {
      %cube_s = scope.scope : () -> tensor<16x16xf32> {
        %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot0 = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s_mask = scope.scope : () -> tensor<16x16xf32> {
        %e_init = tensor.empty() : tensor<16x16xf32>
        %e = hivm.hir.vexp ins(%cube_s : tensor<16x16xf32>) outs(%e_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %e : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %m_new = scope.scope : () -> tensor<16x16xf32> {
        %m_init = tensor.empty() : tensor<16x16xf32>
        %m = hivm.hir.vmax ins(%m_prev, %s_mask : tensor<16x16xf32>, tensor<16x16xf32>) outs(%m_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %m : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %alpha = scope.scope : () -> tensor<16x16xf32> {
        %a_init = tensor.empty() : tensor<16x16xf32>
        %al = hivm.hir.vsub ins(%m_prev, %m_new : tensor<16x16xf32>, tensor<16x16xf32>) outs(%a_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %al : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %p_copy = scope.scope : () -> tensor<1x1x16x16xf16> {
        %f16_init = tensor.empty() : tensor<16x16xf16>
        %f16 = hivm.hir.vcast ins(%m_new : tensor<16x16xf32>) outs(%f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %f16_4d = tensor.expand_shape %f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf = bufferization.to_tensor %cbuf restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%f16_4d : tensor<1x1x16x16xf16>) outs(%cbuf : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %t_cbuf : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %cube_pv = scope.scope : () -> tensor<16x16xf32> {
        %dot1_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot1 = hivm.hir.mmadL1 ins(%p_copy, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub1_cast = memref.memory_space_cast %ub1 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t1 = bufferization.to_tensor %ub1_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot1 : tensor<1x1x16x16xf32>) outs(%ub1 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t1 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %final_vec = scope.scope : () -> tensor<16x16xf32> {
        %out_init = tensor.empty() : tensor<16x16xf32>
        %out = hivm.hir.vadd ins(%alpha, %cube_pv : tensor<16x16xf32>, tensor<16x16xf32>) outs(%out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %out : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %m_new : tensor<16x16xf32>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<16x16xf32>
  }
}

// -----

// Verify that `applyMergedScopeSameCoreRule` merges `CUBE + CUBE_AND_VECTOR`
// and `VECTOR + CUBE_AND_VECTOR` scopes inside `{cv_merged_scope}` (connected
// only by loop-carried counter recurrences) and collapses each wrapper into a
// single flat `CUBE` or `VECTOR` scope.
//
// CHECK-LABEL: func.func @merge_cube_and_vector_counter_in_merged_scope
// CHECK: scf.for {{.*}} iter_args(%[[CNT_C:.*]] = {{.*}}, %[[CNT_V:.*]] = {{.*}}) -> (i32, i32) : i32 {
// CHECK-NEXT:   %[[CUBE_MERGED:.*]]:2 = scope.scope : () -> (tensor<16x16xf32>, i32) {
// CHECK:          arith.index_cast %[[CNT_C]] : i32 to index
// CHECK:          hivm.hir.mmadL1
// CHECK:          hivm.hir.fixpipe
// CHECK:          %[[NEXT_C:.*]] = arith.addi %[[CNT_C]], {{.*}} : i32
// CHECK-NEXT:     scope.return {{.*}}, %[[NEXT_C]] : tensor<16x16xf32>, i32
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:   %[[VEC_MERGED:.*]] = scope.scope : () -> i32 {
// CHECK:          %[[NEXT_V:.*]] = arith.addi %[[CNT_V]], {{.*}} : i32
// CHECK:          arith.index_cast %[[CNT_V]] : i32 to index
// CHECK:          hivm.hir.vexp ins(%[[CUBE_MERGED]]#0 : tensor<16x16xf32>)
// CHECK:          hivm.hir.store
// CHECK-NEXT:     scope.return %[[NEXT_V]] : i32
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:    cv_merged_scope
// CHECK-NEXT:   scf.yield %[[CUBE_MERGED]]#1, %[[VEC_MERGED]] : i32, i32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @merge_cube_and_vector_counter_in_merged_scope(
      %a: tensor<1x1x16x16xf16>,
      %k_gm: memref<?xf16>,
      %out_gm: memref<?xf32>,
      %ub: i32) -> (i32, i32)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %res:2 = scf.for %i = %c0 to %ub step %c1 iter_args(%cnt_c = %c0, %cnt_v = %c0) -> (i32, i32) : i32 {
      %s_cube = scope.scope : () -> tensor<16x16xf32> {
        %off_c = arith.index_cast %cnt_c : i32 to index
        %rc_k = memref.reinterpret_cast %k_gm to offset: [%off_c], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
        %k_alloc = memref.alloc() : memref<1x1x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%rc_k : memref<16x16xf16, strided<[16, 1], offset: ?>>) outs(%k_alloc : memref<1x1x16x16xf16>)
        %k = bufferization.to_tensor %k_alloc restrict writable : memref<1x1x16x16xf16>
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %k, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %s_cnt_v = scope.scope : () -> i32 {
        %next_v = arith.addi %cnt_v, %c16_i32 : i32
        scope.return %next_v : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}

      %s_cnt_c = scope.scope : () -> i32 {
        %next_c = arith.addi %cnt_c, %c16_i32 : i32
        scope.return %next_c : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}

      scope.scope : () -> () {
        %off_v = arith.index_cast %cnt_v : i32 to index
        %rc_out = memref.reinterpret_cast %out_gm to offset: [%off_v], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
        %e_init = tensor.empty() : tensor<16x16xf32>
        %e = hivm.hir.vexp ins(%s_cube : tensor<16x16xf32>) outs(%e_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        hivm.hir.store ins(%e : tensor<16x16xf32>) outs(%rc_out : memref<16x16xf32, strided<[16, 1], offset: ?>>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %s_cnt_c, %s_cnt_v : i32, i32
    } {hivm.cv_pipelined_loop}
    return %res#0, %res#1 : i32, i32
  }
}

// -----

// Verify that a scalar-only `CUBE_AND_VECTOR` scope producing a loop-carried
// scalar value used by both `CUBE` and `VECTOR` scopes is cloned/merged into
// each user scope before loop-carried grouping, erasing the scalar-only scope
// and giving each merged scope its own loop `iter_arg`.
//
// CHECK-LABEL: func.func @clone_shared_scalar_lcd_cube_and_vector_scope_to_users
// CHECK: scf.for {{.*}} iter_args(%[[CNT0:.*]] = %[[C0:.*]], %[[CNT1:.*]] = %[[C0]]) -> (i32, i32) : i32 {
// CHECK-NEXT:   %[[CUBE_MERGED:.*]]:2 = scope.scope : () -> (i32, tensor<16x16xf32>) {
// CHECK-NEXT:     %[[NEXT0:.*]] = arith.addi %[[CNT0]], {{.*}} : i32
// CHECK:          arith.index_cast %[[CNT0]] : i32 to index
// CHECK:          hivm.hir.mmadL1
// CHECK:          hivm.hir.fixpipe
// CHECK-NEXT:     scope.return %[[NEXT0]], {{.*}} : i32, tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:   %[[VEC_MERGED:.*]] = scope.scope : () -> i32 {
// CHECK-NEXT:     %[[NEXT1:.*]] = arith.addi %[[CNT1]], {{.*}} : i32
// CHECK:          arith.index_cast %[[CNT1]] : i32 to index
// CHECK:          hivm.hir.vexp ins(%[[CUBE_MERGED]]#1 : tensor<16x16xf32>)
// CHECK:          hivm.hir.store
// CHECK-NEXT:     scope.return %[[NEXT1]] : i32
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:    cv_merged_scope
// CHECK-NEXT:   scf.yield %[[CUBE_MERGED]]#0, %[[VEC_MERGED]] : i32, i32
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @clone_shared_scalar_lcd_cube_and_vector_scope_to_users(
      %a: tensor<1x1x16x16xf16>,
      %k_gm: memref<?xf16>,
      %out_gm: memref<?xf32>,
      %ub: i32) -> i32
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%cnt = %c0) -> (i32) : i32 {
      %s_cnt = scope.scope : () -> i32 {
        %next = arith.addi %cnt, %c16_i32 : i32
        scope.return %next : i32
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE_AND_VECTOR>}

      %s_cube = scope.scope : () -> tensor<16x16xf32> {
        %off_c = arith.index_cast %cnt : i32 to index
        %rc_k = memref.reinterpret_cast %k_gm to offset: [%off_c], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
        %k_alloc = memref.alloc() : memref<1x1x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%rc_k : memref<16x16xf16, strided<[16, 1], offset: ?>>) outs(%k_alloc : memref<1x1x16x16xf16>)
        %k = bufferization.to_tensor %k_alloc restrict writable : memref<1x1x16x16xf16>
        %dot_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %k, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      scope.scope : () -> () {
        %off_v = arith.index_cast %cnt : i32 to index
        %rc_out = memref.reinterpret_cast %out_gm to offset: [%off_v], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
        %e_init = tensor.empty() : tensor<16x16xf32>
        %e = hivm.hir.vexp ins(%s_cube : tensor<16x16xf32>) outs(%e_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        hivm.hir.store ins(%e : tensor<16x16xf32>) outs(%rc_out : memref<16x16xf32, strided<[16, 1], offset: ?>>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %s_cnt : i32
    } {hivm.cv_pipelined_loop}
    return %res : i32
  }
}

// -----

// Verify `applySmallVectorScopeMergeRule` (running right before
// `applyGraphSameCoreRule`):
// - `%v_pre` (< 3 CoreOps: load + vcast, 0 input, 1 output) merges into
//   connected VECTOR scope `%v_mid` inside `{cv_merged_scope}`.
// - `%v_store0` and `%v_store1` (< 3 CoreOps: store, 1 input, 0 output) merge
//   into their respective connected VECTOR scopes (`%v_head` and `%v_mid`)
//   inside `{cv_merged_scope}` instead of merging together outside via wrapper
//   fan-out, pruning dead `{cv_merged_scope}` and inner scope results.
//
// CHECK-LABEL: func.func @merge_small_vector_scopes_into_connected_vector_scopes
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>) : i32 {
// CHECK-NEXT:   %[[LCD:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK-NEXT:     %[[V0:.*]]:2 = scope.scope : () -> (tensor<16x16xf32>, tensor<1x1x16x16xf16>) {
// CHECK:            hivm.hir.vexp ins(%[[ACC]] : tensor<16x16xf32>)
// CHECK:            %[[H_F16:.*]] = hivm.hir.vcast
// CHECK:            hivm.hir.copy
// CHECK:            hivm.hir.store ins(%[[H_F16]] : tensor<16x16xf16>) outs(%arg2 : memref<16x16xf16>)
// CHECK-NEXT:       scope.return {{.*}}, {{.*}} : tensor<16x16xf32>, tensor<1x1x16x16xf16>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:     %[[C0:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:            hivm.hir.mmadL1 ins(%[[V0]]#1, {{.*}})
// CHECK:            hivm.hir.fixpipe
// CHECK-NEXT:       scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:     %[[V1:.*]] = scope.scope : () -> tensor<1x1x16x16xf16> {
// CHECK:            hivm.hir.load ins(%arg1 : memref<16x16xf16>)
// CHECK:            %[[PRE_F32:.*]] = hivm.hir.vcast
// CHECK:            hivm.hir.vsub ins(%[[PRE_F32]], %[[C0]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK:            %[[M_F16:.*]] = hivm.hir.vcast
// CHECK:            hivm.hir.copy
// CHECK:            hivm.hir.store ins(%[[M_F16]] : tensor<16x16xf16>) outs(%arg3 : memref<16x16xf16>)
// CHECK-NEXT:       scope.return {{.*}} : tensor<1x1x16x16xf16>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:     %[[C1:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:            hivm.hir.mmadL1 ins(%[[V1]], {{.*}})
// CHECK:            hivm.hir.fixpipe
// CHECK-NEXT:       scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NEXT:     %[[V2:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:            %[[OUT:.*]] = hivm.hir.vadd ins(%[[V0]]#0, %[[C1]] : tensor<16x16xf32>, tensor<16x16xf32>)
// CHECK-NEXT:       scope.return %[[OUT]] : tensor<16x16xf32>
// CHECK-NEXT:     } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:     scope.return %[[V2]] : tensor<16x16xf32>
// CHECK-NEXT:   } {cv_merged_scope}
// CHECK-NEXT:   scf.yield %[[LCD]] : tensor<16x16xf32>
// CHECK-NEXT: } {hivm.cv_pipelined_loop}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @merge_small_vector_scopes_into_connected_vector_scopes(
      %b: tensor<1x1x16x16xf16>,
      %in_gm: memref<16x16xf16>,
      %out0_gm: memref<16x16xf16>,
      %out1_gm: memref<16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %v_pre = scope.scope : () -> tensor<16x16xf32> {
        %ld_ub = memref.alloc() : memref<16x16xf16>
        hivm.hir.load ins(%in_gm : memref<16x16xf16>) outs(%ld_ub : memref<16x16xf16>)
        %ld_t = bufferization.to_tensor %ld_ub restrict writable : memref<16x16xf16>
        %f32_init = tensor.empty() : tensor<16x16xf32>
        %f32 = hivm.hir.vcast ins(%ld_t : tensor<16x16xf16>) outs(%f32_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %f32 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %v_head:3 = scope.scope : () -> (tensor<16x16xf32>, tensor<16x16xf16>, tensor<1x1x16x16xf16>) {
        %e_init = tensor.empty() : tensor<16x16xf32>
        %e = hivm.hir.vexp ins(%acc : tensor<16x16xf32>) outs(%e_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        %h_init = tensor.empty() : tensor<16x16xf16>
        %h_f16 = hivm.hir.vcast ins(%acc : tensor<16x16xf32>) outs(%h_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %h_4d = tensor.expand_shape %h_f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf0 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf0 = bufferization.to_tensor %cbuf0 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%h_4d : tensor<1x1x16x16xf16>) outs(%cbuf0 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %e, %h_f16, %t_cbuf0 : tensor<16x16xf32>, tensor<16x16xf16>, tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %c_mid = scope.scope : () -> tensor<16x16xf32> {
        %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot0 = hivm.hir.mmadL1 ins(%v_head#2, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %v_mid:2 = scope.scope : () -> (tensor<16x16xf16>, tensor<1x1x16x16xf16>) {
        %sub_init = tensor.empty() : tensor<16x16xf32>
        %sub = hivm.hir.vsub ins(%v_pre, %c_mid : tensor<16x16xf32>, tensor<16x16xf32>) outs(%sub_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        %m_init = tensor.empty() : tensor<16x16xf16>
        %m_f16 = hivm.hir.vcast ins(%sub : tensor<16x16xf32>) outs(%m_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %m_4d = tensor.expand_shape %m_f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf1 = bufferization.to_tensor %cbuf1 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%m_4d : tensor<1x1x16x16xf16>) outs(%cbuf1 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %m_f16, %t_cbuf1 : tensor<16x16xf16>, tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %c_tail = scope.scope : () -> tensor<16x16xf32> {
        %dot1_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot1 = hivm.hir.mmadL1 ins(%v_mid#1, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub1_cast = memref.memory_space_cast %ub1 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t1 = bufferization.to_tensor %ub1_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot1 : tensor<1x1x16x16xf32>) outs(%ub1 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t1 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %v_tail = scope.scope : () -> tensor<16x16xf32> {
        %out_init = tensor.empty() : tensor<16x16xf32>
        %out = hivm.hir.vadd ins(%v_head#0, %c_tail : tensor<16x16xf32>, tensor<16x16xf32>) outs(%out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %out : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scope.scope : () -> () {
        hivm.hir.store ins(%v_head#1 : tensor<16x16xf16>) outs(%out0_gm : memref<16x16xf16>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scope.scope : () -> () {
        hivm.hir.store ins(%v_mid#0 : tensor<16x16xf16>) outs(%out1_gm : memref<16x16xf16>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %v_tail : tensor<16x16xf32>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<16x16xf32>
  }
}

// -----

// Verify that a VECTOR scope with >= 3 CoreOps is NOT merged by
// `applySmallVectorScopeMergeRule` into `{cv_merged_scope}`.
//
// CHECK-LABEL: func.func @do_not_merge_vector_scope_with_three_core_ops
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (tensor<16x16xf32>) : i32 {
// CHECK-NEXT:   %[[PRE:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          hivm.hir.load
// CHECK:          hivm.hir.vcast
// CHECK:          hivm.hir.vabs
// CHECK-NEXT:     scope.return {{.*}} : tensor<16x16xf32>
// CHECK-NEXT:   } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NEXT:   %[[LCD:.*]] = scope.scope : () -> tensor<16x16xf32> {
// CHECK:          hivm.hir.vadd ins(%[[PRE]], {{.*}})
// CHECK:        } {cv_merged_scope}
// CHECK-NEXT:   scf.yield %[[LCD]] : tensor<16x16xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @do_not_merge_vector_scope_with_three_core_ops(
      %a: tensor<1x1x16x16xf16>,
      %b: tensor<1x1x16x16xf16>,
      %in_gm: memref<16x16xf16>,
      %ub: i32) -> tensor<16x16xf32>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %init = tensor.empty() : tensor<16x16xf32>
    %res = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<16x16xf32>) : i32 {
      %v_pre_3ops = scope.scope : () -> tensor<16x16xf32> {
        %ld_ub = memref.alloc() : memref<16x16xf16>
        hivm.hir.load ins(%in_gm : memref<16x16xf16>) outs(%ld_ub : memref<16x16xf16>)
        %ld_t = bufferization.to_tensor %ld_ub restrict writable : memref<16x16xf16>
        %f32_init = tensor.empty() : tensor<16x16xf32>
        %f32 = hivm.hir.vcast ins(%ld_t : tensor<16x16xf16>) outs(%f32_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        %abs_init = tensor.empty() : tensor<16x16xf32>
        %abs = hivm.hir.vabs ins(%f32 : tensor<16x16xf32>) outs(%abs_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %abs : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %v_head = scope.scope : () -> tensor<1x1x16x16xf16> {
        %h_init = tensor.empty() : tensor<16x16xf16>
        %h_f16 = hivm.hir.vcast ins(%acc : tensor<16x16xf32>) outs(%h_init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %h_4d = tensor.expand_shape %h_f16 [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %cbuf0 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %t_cbuf0 = bufferization.to_tensor %cbuf0 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%h_4d : tensor<1x1x16x16xf16>) outs(%cbuf0 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        scope.return %t_cbuf0 : tensor<1x1x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      %c_mid = scope.scope : () -> tensor<16x16xf32> {
        %dot0_init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot0 = hivm.hir.mmadL1 ins(%v_head, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        %ub0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
        %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
        %t0 = bufferization.to_tensor %ub0_cast restrict writable : memref<16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot0 : tensor<1x1x16x16xf32>) outs(%ub0 : memref<16x16xf32, #hivm.address_space<ub>>)
        scope.return %t0 : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

      %v_tail = scope.scope : () -> tensor<16x16xf32> {
        %out_init = tensor.empty() : tensor<16x16xf32>
        %out = hivm.hir.vadd ins(%v_pre_3ops, %c_mid : tensor<16x16xf32>, tensor<16x16xf32>) outs(%out_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scope.return %out : tensor<16x16xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

      scf.yield %v_tail : tensor<16x16xf32>
    } {hivm.cv_pipelined_loop}
    return %res : tensor<16x16xf32>
  }
}
