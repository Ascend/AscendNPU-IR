// Test [MS_STATUS] stderr logging and --sms-fail-on-fallback behavior.
//
// The loop below builds a valid Default worklist (vexp/mmadL1/fixpipe/vadd
// produce multiple work items), so CVPipelineImpl::run() reaches the SMS
// block. Running scheduler=sms with pipeline-mode=unroll then fails the
// first SMS check deterministically (SMS requires skew mode), independent
// of cost-model tuning: the expected markers are attempt followed by
// fallback with reason=unsupported-pipeline-mode.

// --- Test 1: Default (sms-fail-on-fallback=false) ---
// SMS fails, logs attempt+fallback, falls back to Default; the pass itself
// succeeds and the function is emitted.
// RUN: bishengir-opt -mlir-disable-threading \
// RUN:   -cv-pipelining="pipeline-mode=unroll scheduler=sms" \
// RUN:   -allow-unregistered-dialect %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-DEFAULT,CHECK-STATUS

// --- Test 2: sms-fail-on-fallback=true ---
// SMS fails, logs attempt+fallback, then the pass signals failure and
// bishengir-opt exits non-zero.
// RUN: not bishengir-opt -mlir-disable-threading \
// RUN:   -cv-pipelining="pipeline-mode=unroll scheduler=sms sms-fail-on-fallback=true" \
// RUN:   -allow-unregistered-dialect %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-FAIL

// CHECK-STATUS: [MS_STATUS] status=attempt function=sms_fallback_unsupported_mode
// CHECK-STATUS: [MS_STATUS] status=fallback function=sms_fallback_unsupported_mode
// CHECK-STATUS-SAME: reason=unsupported-pipeline-mode
// CHECK-DEFAULT-NOT: [MS_STATUS] status=success
// CHECK-DEFAULT: func.func @sms_fallback_unsupported_mode

// CHECK-FAIL: [MS_STATUS] status=attempt function=sms_fallback_unsupported_mode
// CHECK-FAIL: [MS_STATUS] status=fallback function=sms_fallback_unsupported_mode
// CHECK-FAIL-SAME: reason=unsupported-pipeline-mode
// CHECK-FAIL-NOT: [MS_STATUS] status=success

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sms_fallback_unsupported_mode(
      %a: tensor<16x16xf16>,
      %b: tensor<16x16xf16>,
      %vector_in: tensor<16x16xf32>) attributes {
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
    scf.for %i = %c0 to %bound step %step : i32 {
      %vector_init = tensor.empty() : tensor<16x16xf32>
      %exp = hivm.hir.vexp ins(%vector_in : tensor<16x16xf32>) outs(%vector_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %dot_init = tensor.empty() : tensor<16x16xf32>
      %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %cube_out = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %cube_out_cast = memref.memory_space_cast %cube_out : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %cube_out_tensor = bufferization.to_tensor %cube_out_cast restrict writable : memref<16x16xf32>
      %fix = hivm.hir.fixpipe ins(%dot : tensor<16x16xf32>) outs(%cube_out_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>
      %sum_init = tensor.empty() : tensor<16x16xf32>
      %sum = hivm.hir.vadd ins(%exp, %fix : tensor<16x16xf32>, tensor<16x16xf32>) outs(%sum_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield
    }
    return
  }
}
