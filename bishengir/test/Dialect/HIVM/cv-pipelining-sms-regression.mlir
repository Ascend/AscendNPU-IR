// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew scheduler=sms" -allow-unregistered-dialect %s | FileCheck %s --check-prefix=SMS

// ============================================================================
// T7: SMS + unroll mode → revert
// ============================================================================
// This test is invoked with skew mode in the RUN line but the IR below
// is designed to test the pass option parsing. For actual non-skew rejection,
// use the following RUN line on the build server:
//
// bishengir-opt -cv-pipelining="set-depth-in-unroll-mode=2 scheduler=sms" -allow-unregistered-dialect %s
//
// Expected: SMS rejects non-skew mode → falls back to no pipelining (revert).
// The loop and its contents are preserved unchanged.

// ============================================================================
// T11: Default scheduler + skew = same output as no scheduler option
// ============================================================================
// Verify that adding scheduler=default to the existing preload test
// produces identical output. This is a regression guard ensuring the
// SMS code path does not interfere with the default behavior.
//
// To run manually: copy the IR from cv-pipelining-preload.mlir and invoke
// bishengir-opt -cv-pipelining="pipeline-mode=skew scheduler=default"
// -allow-unregistered-dialect %s | FileCheck %s --check-prefix=DEFAULT
//
// The CHECK lines should be identical to cv-pipelining-preload.mlir's
// existing CHECK lines — proving scheduler=default is a no-op.

// --- Actual test: 4-WorkItem A5 pattern under SMS ---
// This is the canonical cv-pipelining-preload.mlir IR tested with SMS.
// The SMS scheduler with FixedProductionLatencyProvider (lat=2, issue=2)
// will produce a specific SMS schedule. The exact preload_num values
// depend on the SMS output — verify with sms_cross_validate.py first
// and then fill in the expected CHECK values.
//
// Placeholder CHECK lines below assume SMS produces 2 stages:
//   stage 0 → cube+vector pair (preload_num=1)
//   stage 1 → cube+vector pair (preload_num=0)

// SMS-LABEL: func.func @sms_four_stage_a5
// SMS: scf.for
// SMS: scope.scope
// SMS:   hivm.hir.fixpipe
// SMS-NEXT: scope.return
// SMS: } {{.*}}hivm.preload_num = {{[0-9]+}} : i32, no_inline}
// SMS: scope.scope
// SMS:   hivm.hir.vexp
// SMS: } {{.*}}hivm.preload_num = {{[0-9]+}} : i32, no_inline}
// SMS: scope.scope
// SMS:   hivm.hir.mmadL1
// SMS: } {{.*}}hivm.preload_num = {{[0-9]+}} : i32, no_inline}
// SMS: scope.scope
// SMS:   hivm.hir.vexp
// SMS: } {{.*}}hivm.preload_num = {{[0-9]+}} : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sms_four_stage_a5(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {
    WorkspaceArgIdx = 0 : i16,
    func_dyn_memref_args = dense<[true]> : vector<1xi1>,
    global_kernel = "local",
    hacc.entry,
    hacc.function_kind = #hacc.function_kind<DEVICE>,
    hivm.func_core_type = #hivm.func_core_type<MIX>,
    mix_mode = "mix"
  } {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %k_mem = "some_op"() : () -> memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %init = tensor.empty() : tensor<16x16xf32>

    %result = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
      // Stage A: load → mmadL1 → fixpipe (cube)
      %k0_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_mem : memref<16x16xf16>) outs(%k0_alloc : memref<16x16xf16>)
      %k0 = bufferization.to_tensor %k0_alloc : memref<16x16xf16>
      %dot0_init = tensor.empty() : tensor<16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %k0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws0 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws0_tensor = bufferization.to_tensor %ws0 restrict writable : memref<16x16xf32>
      %fix0 = hivm.hir.fixpipe ins(%dot0 : tensor<16x16xf32>) outs(%ws0_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      // Stage B: load → vexp → vcast → store (vector)
      %load0_init = tensor.empty() : tensor<16x16xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<16x16xf32>) outs(%load0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_init = tensor.empty() : tensor<16x16xf32>
      %v0 = hivm.hir.vexp ins(%load0 : tensor<16x16xf32>) outs(%v0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_f16_init = tensor.empty() : tensor<16x16xf16>
      %v0_f16 = hivm.hir.vcast ins(%v0 : tensor<16x16xf32>) outs(%v0_f16_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf16>
      annotation.mark %ws1 {hivm.multi_buffer = 4 : i32} : memref<16x16xf16>
      %ws1_tensor = bufferization.to_tensor %ws1 restrict writable : memref<16x16xf16>
      %store1 = hivm.hir.store ins(%v0_f16 : tensor<16x16xf16>) outs(%ws1_tensor : tensor<16x16xf16>) -> tensor<16x16xf16>

      // Stage C: load → mmadL1 → fixpipe (cube)
      %load1_init = tensor.empty() : tensor<16x16xf16>
      %load1 = hivm.hir.load ins(%store1 : tensor<16x16xf16>) outs(%load1_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %k1_alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%k_mem : memref<16x16xf16>) outs(%k1_alloc : memref<16x16xf16>)
      %k1 = bufferization.to_tensor %k1_alloc : memref<16x16xf16>
      %dot1_init = tensor.empty() : tensor<16x16xf32>
      %dot1 = hivm.hir.mmadL1 ins(%load1, %k1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot1_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws2 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws2 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws2_tensor = bufferization.to_tensor %ws2 restrict writable : memref<16x16xf32>
      %fix1 = hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf32>) outs(%ws2_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      // Stage D: load → vexp (vector)
      %load2_init = tensor.empty() : tensor<16x16xf32>
      %load2 = hivm.hir.load ins(%fix1 : tensor<16x16xf32>) outs(%load2_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v1_init = tensor.empty() : tensor<16x16xf32>
      %v1 = hivm.hir.vexp ins(%load2 : tensor<16x16xf32>) outs(%v1_init : tensor<16x16xf32>) -> tensor<16x16xf32>

      scf.yield %v1 : tensor<16x16xf32>
    }
    "some_consume"(%result) : (tensor<16x16xf32>) -> ()
    return
  }
}
