// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// Test that a fixpipe is not inserted redundantly when an MMAD result flows
// through extract_slice to fixpipes in both scf.if branches.
// CHECK-LABEL: func.func @no_redundant_fixpipe_for_extract_slice_to_scf_if
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[MMAD]]
// CHECK: tensor.extract_slice %[[MMAD]]
func.func @no_redundant_fixpipe_for_extract_slice_to_scf_if(
    %arg0: memref<1x128xbf16>,
    %arg1: memref<128x128xbf16>,
    %arg2: memref<1x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %a = bufferization.to_tensor %arg0 restrict writable : memref<1x128xbf16>
  %b = bufferization.to_tensor %arg1 restrict writable : memref<128x128xbf16>
  %c = tensor.empty() : tensor<16x128xf32>
  %mmad = hivm.hir.mmadL1
      ins(%a, %b, %true, %c16, %c128, %c128 : tensor<1x128xbf16>, tensor<128x128xbf16>, i1, index, index, index)
      outs(%c : tensor<16x128xf32>) -> tensor<16x128xf32>
  %extracted = tensor.extract_slice %mmad[0, 0] [1, 128] [1, 1]
      : tensor<16x128xf32> to tensor<1x128xf32>
  %dst0 = tensor.empty() : tensor<1x128xf32>
  %dst1 = tensor.empty() : tensor<1x128xf32>
  %cond = arith.cmpi eq, %c0_i32, %c1_i32 : i32
  %result = scf.if %cond -> tensor<1x128xf32> {
    %fix0 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%extracted : tensor<1x128xf32>) outs(%dst0 : tensor<1x128xf32>) -> tensor<1x128xf32>
    scf.yield %fix0 : tensor<1x128xf32>
  } else {
    %fix1 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%extracted : tensor<1x128xf32>) outs(%dst1 : tensor<1x128xf32>) -> tensor<1x128xf32>
    scf.yield %fix1 : tensor<1x128xf32>
  }
  hivm.hir.store ins(%result : tensor<1x128xf32>) outs(%arg2 : memref<1x128xf32>)
  return
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @inline_fixpipe_for_iter_arg_mmad
// CHECK: hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true}
// CHECK: scf.yield
func.func @inline_fixpipe_for_iter_arg_mmad(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
    %arg5: i32, %arg6: i32, %arg7: i32) {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  // TODO: Restore `eviction_policy = <EvictFirst>` from the register-based
  // test once A3 LoadOp supports the same assembly syntax.
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  %alloc = memref.alloc() : memref<16x16xf32>
  hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc : memref<16x16xf32>)
  %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  %alloc_1 = memref.alloc() : memref<16x16xf32>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf32>)
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32>
  %4 = scf.for %arg8 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %2) -> (tensor<16x16xf32>) : i32 {
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins(%arg9, %3, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %6 : tensor<16x16xf32>
  }
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
  hivm.hir.store ins(%4 : tensor<16x16xf32>) outs(%reinterpret_cast_2 : memref<16x16xf32, strided<[16, 1]>>)
  hivm.hir.set_ctrl true at ctrl[60]
  return
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @while_for_l0c_fixpipe_before_store
// CHECK: %[[FOR:.*]] = scf.for
// CHECK:   %[[WHILE:.*]] = scf.while
// CHECK:     %[[IF:.*]] = scf.if
// CHECK:       %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield %[[MMAD]]
// CHECK:     } else {
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield
// CHECK:     }
// CHECK:     scf.yield %[[IF]]
// CHECK:   }
// CHECK:   scf.yield %[[WHILE]]
// CHECK: }
// CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[MMAD]]
// CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[IF]]
// CHECK-NOT: hivm.hir.fixpipe {{.*}} ins(%[[WHILE]]
// CHECK: %[[FP:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[FOR]]
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[FP]]
// CHECK: hivm.hir.store ins(%[[SLICE]]
func.func @while_for_l0c_fixpipe_before_store(
    %cond: i1,
    %a: tensor<16x16xf16>,
    %b: tensor<16x16xf16>,
    %out: memref<?x?xf32, strided<[16, 1], offset: ?>>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %init = tensor.empty() : tensor<16x16xf32>
  %for = scf.for %i = %c0 to %c2 step %c1
      iter_args(%acc = %init) -> (tensor<16x16xf32>) {
    %while = scf.while (%wacc = %acc) : (tensor<16x16xf32>) -> (tensor<16x16xf32>) {
      scf.condition(%cond) %wacc : tensor<16x16xf32>
    } do {
    ^bb0(%wacc: tensor<16x16xf32>):
      %if = scf.if %cond -> (tensor<16x16xf32>) {
        %mmad = hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c = [0 : i32], normalized_in_L0C}
            ins(%a, %b, %true, %c16, %c16, %c16
                : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
            outs(%wacc : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %mmad : tensor<16x16xf32>
      } else {
        scf.yield %wacc : tensor<16x16xf32>
      } {hivm.remain_in_l0c = [0 : i32]}
      scf.yield %if : tensor<16x16xf32>
    } attributes {hivm.remain_in_l0c = [0 : i32]}
    scf.yield %while : tensor<16x16xf32>
  } {hivm.remain_in_l0c = [0 : i32], normalized_in_L0C = [0 : i32]}
  %slice = tensor.extract_slice %for[0, 0] [%c16, %c16] [1, 1]
      : tensor<16x16xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%slice : tensor<?x?xf32>)
      outs(%out : memref<?x?xf32, strided<[16, 1], offset: ?>>)
  return
}
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// Two L0C results, matching the carried pair that is sliced and stored after
// the outer scf.for. Each fixpipe reads one for-result; neither yield is
// rewritten.
// CHECK-LABEL: func.func @while_for_two_l0c_fixpipes_before_stores
// CHECK: %[[FOR:.*]]:2 = scf.for
// CHECK:   %[[WHILE:.*]]:2 = scf.while
// CHECK:     %[[IF1:.*]] = scf.if
// CHECK:       %[[MMAD1:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield %[[MMAD1]]
// CHECK:     } else {
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield
// CHECK:     }
// CHECK:     %[[IF0:.*]] = scf.if
// CHECK:       %[[MMAD0:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield %[[MMAD0]]
// CHECK:     } else {
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:       scf.yield
// CHECK:     }
// CHECK:     scf.yield %[[IF0]], %[[IF1]]
// CHECK:   }
// CHECK:   scf.yield %[[WHILE]]#0, %[[WHILE]]#1
// CHECK: }
// CHECK-DAG: %[[FP0:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[FOR]]#0
// CHECK-DAG: %[[FP1:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[FOR]]#1
// CHECK: %[[SLICE1:.*]] = tensor.extract_slice %[[FP1]]
// CHECK: hivm.hir.store ins(%[[SLICE1]]
// CHECK: %[[SLICE0:.*]] = tensor.extract_slice %[[FP0]]
// CHECK: hivm.hir.store ins(%[[SLICE0]]
func.func @while_for_two_l0c_fixpipes_before_stores(
    %cond: i1,
    %a: tensor<16x16xf16>,
    %b: tensor<16x16xf16>,
    %out0: memref<?x?xf32, strided<[16, 1], offset: ?>>,
    %out1: memref<?x?xf32, strided<[16, 1], offset: ?>>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %init0 = tensor.empty() : tensor<16x16xf32>
  %init1 = tensor.empty() : tensor<16x16xf32>
  // Result 0 accumulates in the second if; result 1 accumulates in the first.
  %for:2 = scf.for %i = %c0 to %c2 step %c1
      iter_args(%acc0 = %init0, %acc1 = %init1) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
    %while:2 = scf.while (%w0 = %acc0, %w1 = %acc1)
        : (tensor<16x16xf32>, tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
      scf.condition(%cond) %w0, %w1 : tensor<16x16xf32>, tensor<16x16xf32>
    } do {
    ^bb0(%w0: tensor<16x16xf32>, %w1: tensor<16x16xf32>):
      %if1 = scf.if %cond -> (tensor<16x16xf32>) {
        %mmad1 = hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c = [0 : i32], normalized_in_L0C}
            ins(%a, %b, %true, %c16, %c16, %c16
                : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
            outs(%w1 : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %mmad1 : tensor<16x16xf32>
      } else {
        scf.yield %w1 : tensor<16x16xf32>
      } {hivm.remain_in_l0c = [0 : i32]}
      %if0 = scf.if %cond -> (tensor<16x16xf32>) {
        %mmad0 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, hivm.remain_in_l0c = [0 : i32], normalized_in_L0C}
            ins(%a, %b, %true, %c16, %c16, %c16
                : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
            outs(%w0 : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %mmad0 : tensor<16x16xf32>
      } else {
        scf.yield %w0 : tensor<16x16xf32>
      } {hivm.remain_in_l0c = [0 : i32]}
      scf.yield %if0, %if1 : tensor<16x16xf32>, tensor<16x16xf32>
    } attributes {hivm.remain_in_l0c = [0 : i32, 1 : i32]}
    scf.yield %while#0, %while#1 : tensor<16x16xf32>, tensor<16x16xf32>
  } {hivm.remain_in_l0c = [0 : i32, 1 : i32], normalized_in_L0C = [1 : i32, 0 : i32]}
  %slice1 = tensor.extract_slice %for#1[0, 0] [%c16, %c16] [1, 1]
      : tensor<16x16xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%slice1 : tensor<?x?xf32>)
      outs(%out1 : memref<?x?xf32, strided<[16, 1], offset: ?>>)
  %slice0 = tensor.extract_slice %for#0[0, 0] [%c16, %c16] [1, 1]
      : tensor<16x16xf32> to tensor<?x?xf32>
  hivm.hir.store ins(%slice0 : tensor<?x?xf32>)
      outs(%out0 : memref<?x?xf32, strided<[16, 1], offset: ?>>)
  return
}
}
