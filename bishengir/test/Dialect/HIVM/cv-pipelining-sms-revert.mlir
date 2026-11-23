// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew scheduler=sms" -allow-unregistered-dialect -verify-diagnostics -split-input-file %s | FileCheck %s

// ============================================================================
// T7: SMS + non-skew mode → revert (tested here via skew + IR that SMS rejects)
// ============================================================================

// ============================================================================
// T8: SMS revert on empty/trivial loop body (no compute ops → DAG build fails)
// ============================================================================
// A loop with only scalar ops and no cube/vector compute → MsDagBuilder
// produces no compute nodes → revert. The original loop must be preserved.

// CHECK-LABEL: func.func @sms_revert_no_compute
// CHECK-NOT: scope.scope
// CHECK: scf.for
// CHECK:   arith.addi
// CHECK-NOT: scope.scope
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sms_revert_no_compute() attributes {
    hacc.entry,
    hacc.function_kind = #hacc.function_kind<DEVICE>,
    hivm.func_core_type = #hivm.func_core_type<MIX>,
    mix_mode = "mix"
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    scf.for %i = %c0 to %c8 step %c1 : i32 {
      %sum = arith.addi %i, %c1 : i32
      scf.yield
    }
    return
  }
}

// -----

// ============================================================================
// T10: SMS revert on unsupported output (scf.if tensor result)
// ============================================================================
// Same IR pattern as cv-pipelining-preload-rollback.mlir — markOutputs fails
// in the common post-partition pipeline → revert. SMS partition installs
// successfully but the common markOutputs/dependency check rejects.

// CHECK-LABEL: func.func @sms_revert_unsupported_output
// CHECK-NOT: scope.scope
// CHECK: scf.for
// CHECK-NOT: scope.scope
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sms_revert_unsupported_output(
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %vector_in: tensor<16x16xf32>,
      %cond: i1) attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        hivm.func_core_type = #hivm.func_core_type<MIX>,
        mix_mode = "mix"
      } {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    // The SMS DAG build queries the MS cost model before markOutputs rejects
    // the partition: the scf.if group has no shaped input, so its synthetic
    // GM_to_UB transfer carries no data-node metadata.
    // Matched on the stable tail: the synthetic node id varies with DAG
    // numbering, and expected-* is a plain substring match (no regex).
    // expected-warning@+1 {{(mte2, GM_to_UB) missing data node metadata}}
    scf.for %i = %c0 to %bound step %step : i32 {
      // No vector cost-model warning here: op kind no longer selects a
      // multiplier, so scf.if bills like any other op in the group.
      %vector_init = tensor.empty() : tensor<16x16xf32>
      // expected-warning@+1 {{expected to_tensor for non-tensor-empty output}}
      %selected = scf.if %cond -> tensor<16x16xf32> {
        %exp = hivm.hir.vexp ins(%vector_in : tensor<16x16xf32>) outs(%vector_init : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %exp : tensor<16x16xf32>
      } else {
        scf.yield %vector_in : tensor<16x16xf32>
      }

      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cube_out = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %cube_out_cast = memref.memory_space_cast %cube_out : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube_out_tensor = bufferization.to_tensor %cube_out_cast restrict writable : memref<16x16xf32>
      %fix = hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_out_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %sum_init = tensor.empty() : tensor<16x16xf32>
      %sum = hivm.hir.vadd ins(%selected, %fix : tensor<16x16xf32>, tensor<16x16xf32>) outs(%sum_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield
    }
    return
  }
}
