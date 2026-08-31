// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// Test 1: The upstream `scf.for` yields multiple results (an mmad result in L0C + auxiliary tensor + scalar counter).
// These results are passed as `iter_args` to the SAME downstream `scf.for` loop.
// The `mmad` inside the downstream `scf.if` accumulates directly on the cascaded L0C tensor.
// Check that NO fixpipe is inserted inside the first loop or between the two loops, and a single
// fixpipe is bypassed at the end with the loop attributes marked properly.

// CHECK-LABEL: func.func @cascade_mmad_multiple_results_to_same_downstream_loop
// CHECK: %[[FOR1:.*]]:3 = scf.for
// CHECK:   %[[IF1:.*]]:2 = scf.if
// CHECK:     %[[MMAD1:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:     scf.yield %[[MMAD1]]
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield %[[IF1]]#0, %{{.*}}, %[[IF1]]#1
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C = [0 : i32]}
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: %[[FOR2:.*]]:3 = scf.for {{.*}} iter_args(%[[ITER_ACC:.*]] = %[[FOR1]]#0, %[[ITER_AUX:.*]] = %[[FOR1]]#1, %[[ITER_CNT:.*]] = %[[FOR1]]#2)
// CHECK:   %[[IF2:.*]]:2 = scf.if
// CHECK:     %[[MMAD2:.*]] = hivm.hir.mmadL1 {{.*}} outs(%[[ITER_ACC]]
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:     scf.yield %[[MMAD2]]
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield %[[IF2]]#0, %[[ITER_AUX]], %[[IF2]]#1
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true}
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: hivm.hir.store ins(%[[FOR2]]#0
func.func @cascade_mmad_multiple_results_to_same_downstream_loop(
    %cond: i1,
    %arg1: memref<128x128xf16>,
    %arg2: memref<128x128xf16>,
    %arg3: memref<128x128xf16>,
    %arg4: memref<128x128xf32>,
    %arg5: memref<128x128xf32, #hivm.address_space<ub>>) {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %0 = bufferization.to_tensor %arg1 restrict writable : memref<128x128xf16>
  %1 = bufferization.to_tensor %arg2 restrict writable : memref<128x128xf16>
  %2 = bufferization.to_tensor %arg3 restrict writable : memref<128x128xf16>
  %3 = tensor.empty() : tensor<128x128xf32>
  %4 = tensor.empty() : tensor<128x128xf32>

  // First loop: yields (accumulated_mmad, aux_tensor, counter)
  %for1_res:3 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%acc = %3, %aux_arg = %4, %cnt = %c0_i32) -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) {
    %if_res:2 = scf.if %cond -> (tensor<128x128xf32>, i32) {
      %mmad1 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
          ins(%0, %1, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
          outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
      %next_cnt = arith.addi %cnt, %c1_i32 : i32
      scf.yield %mmad1, %next_cnt : tensor<128x128xf32>, i32
    } else {
      scf.yield %acc, %cnt : tensor<128x128xf32>, i32
    }
    scf.yield %if_res#0, %aux_arg, %if_res#1 : tensor<128x128xf32>, tensor<128x128xf32>, i32
  } {hivm.remain_in_l0c, normalized_in_L0C = [0 : i32]}

  // Second loop: consumes all 3 return values as initial iter_args
  %for2_res:3 = scf.for %j = %c0 to %c10 step %c1
      iter_args(%acc2 = %for1_res#0, %aux2 = %for1_res#1, %cnt2 = %for1_res#2) -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) {
    %if_res2:2 = scf.if %cond -> (tensor<128x128xf32>, i32) {
      %mmad2 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
          ins(%0, %2, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
          outs(%acc2 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %next_cnt2 = arith.addi %cnt2, %c1_i32 : i32
      scf.yield %mmad2, %next_cnt2 : tensor<128x128xf32>, i32
    } else {
      scf.yield %acc2, %cnt2 : tensor<128x128xf32>, i32
    }
    scf.yield %if_res2#0, %aux2, %if_res2#1 : tensor<128x128xf32>, tensor<128x128xf32>, i32
  }

  // Store final result to memory
  hivm.hir.store ins(%for2_res#0 : tensor<128x128xf32>) outs(%arg4 : memref<128x128xf32>) atomic = <add>
  return
}

}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// Test 2: The upstream `scf.for` produces MULTIPLE mmad results (normalized_in_L0C = [0 : i32, 1 : i32]).
// Result 0 cascades to downstream Loop A (`%forA`), and Result 1 cascades to downstream Loop B (`%forB`).
// Check that NO fixpipe is inserted inside the first loop or immediately after it, and both downstream
// loops are correctly marked as having fixpipes bypassed.

// CHECK-LABEL: func.func @cascade_mmad_multiple_results_to_different_downstream_loops
// CHECK: %[[FOR1:.*]]:2 = scf.for
// CHECK:   %[[IF1_A:.*]] = scf.if
// CHECK:     %[[MMAD1_A:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:     scf.yield %[[MMAD1_A]]
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   %[[IF1_B:.*]] = scf.if
// CHECK:     %[[MMAD1_B:.*]] = hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:     scf.yield %[[MMAD1_B]]
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield %[[IF1_A]], %[[IF1_B]]
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C = [0 : i32, 1 : i32]}
// CHECK-NOT: hivm.hir.fixpipe
//
// CHECK: %[[FOR_A:.*]] = scf.for {{.*}} iter_args(%[[ITER_A:.*]] = %[[FOR1]]#0)
// CHECK:   %[[IF_A:.*]] = scf.if
// CHECK:     %[[MMAD_A:.*]] = hivm.hir.mmadL1 {{.*}} outs(%[[ITER_A]]
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:     scf.yield %[[MMAD_A]]
// CHECK:   } else {
// CHECK:     scf.yield %[[ITER_A]]
// CHECK:   }
// CHECK:   scf.yield %[[IF_A]]
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true}
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: hivm.hir.store ins(%[[FOR_A]]
//
// CHECK: %[[FOR_B:.*]] = scf.for {{.*}} iter_args(%[[ITER_B:.*]] = %[[FOR1]]#1)
// CHECK:   %[[IF_B:.*]] = scf.if
// CHECK:     %[[MMAD_B:.*]] = hivm.hir.mmadL1 {{.*}} outs(%[[ITER_B]]
// CHECK-NOT: hivm.hir.fixpipe
// CHECK:     scf.yield %[[MMAD_B]]
// CHECK:   } else {
// CHECK:     scf.yield %[[ITER_B]]
// CHECK:   }
// CHECK:   scf.yield %[[IF_B]]
// CHECK: } {fixpipe_for_mmad_result_already_inserted = true}
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: hivm.hir.store ins(%[[FOR_B]]
func.func @cascade_mmad_multiple_results_to_different_downstream_loops(
    %cond1: i1,
    %cond2: i1,
    %arg2: memref<128x128xf16>,
    %arg3: memref<128x128xf16>,
    %arg4: memref<128x128xf16>,
    %out_buf0: memref<128x128xf32>,
    %out_buf1: memref<128x128xf32>) {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index

  %0 = bufferization.to_tensor %arg2 restrict writable : memref<128x128xf16>
  %1 = bufferization.to_tensor %arg3 restrict writable : memref<128x128xf16>
  %2 = bufferization.to_tensor %arg4 restrict writable : memref<128x128xf16>
  %3 = tensor.empty() : tensor<128x128xf32>
  %4 = tensor.empty() : tensor<128x128xf32>

  // First loop produces two distinct mmad results in L0C
  %for1_res:2 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%acc0 = %3, %acc1 = %4) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %res0 = scf.if %cond1 -> (tensor<128x128xf32>) {
      %mmad0 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
          ins(%0, %1, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
          outs(%acc0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %mmad0 : tensor<128x128xf32>
    } else {
      scf.yield %acc0 : tensor<128x128xf32>
    }

    %res1 = scf.if %cond2 -> (tensor<128x128xf32>) {
      %mmad1 = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
          ins(%0, %2, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
          outs(%acc1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %mmad1 : tensor<128x128xf32>
    } else {
      scf.yield %acc1 : tensor<128x128xf32>
    }

    scf.yield %res0, %res1 : tensor<128x128xf32>, tensor<128x128xf32>
  } {hivm.remain_in_l0c, normalized_in_L0C = [0 : i32, 1 : i32]}

  // Downstream loop A: consumes %for1_res#0
  %forA_res = scf.for %j = %c0 to %c10 step %c1
      iter_args(%accA = %for1_res#0) -> (tensor<128x128xf32>) {
    %resA = scf.if %cond1 -> (tensor<128x128xf32>) {
      %mmadA = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
          ins(%0, %1, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
          outs(%accA : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %mmadA : tensor<128x128xf32>
    } else {
      scf.yield %accA : tensor<128x128xf32>
    }
    scf.yield %resA : tensor<128x128xf32>
  }
  hivm.hir.store ins(%forA_res : tensor<128x128xf32>) outs(%out_buf0 : memref<128x128xf32>) atomic = <add>

  // Downstream loop B: consumes %for1_res#1
  %forB_res = scf.for %k = %c0 to %c10 step %c1
      iter_args(%accB = %for1_res#1) -> (tensor<128x128xf32>) {
    %resB = scf.if %cond2 -> (tensor<128x128xf32>) {
      %mmadB = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
          ins(%0, %2, %false, %c128, %c128, %c128 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index)
          outs(%accB : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %mmadB : tensor<128x128xf32>
    } else {
      scf.yield %accB : tensor<128x128xf32>
    }
    scf.yield %resB : tensor<128x128xf32>
  }
  hivm.hir.store ins(%forB_res : tensor<128x128xf32>) outs(%out_buf1 : memref<128x128xf32>) atomic = <add>

  return
}

}
