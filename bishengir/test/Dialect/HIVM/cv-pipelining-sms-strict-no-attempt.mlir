// Test that sms-fail-on-fallback only fails the pass when SMS was actually
// attempted. A trivial loop with no cube/vector compute fails the Default
// WorklistBuilder (worklist < 2 items) before the SMS block ever runs, so:
//   - no [MS_STATUS] markers are emitted, and
//   - sms-fail-on-fallback=true must NOT signal a pass failure.
// This pins the distinction between pre-SMS failures and SMS fallbacks.

// RUN: bishengir-opt -mlir-disable-threading \
// RUN:   -cv-pipelining="pipeline-mode=skew scheduler=sms sms-fail-on-fallback=true" \
// RUN:   -allow-unregistered-dialect %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK-NOT: [MS_STATUS]
// CHECK: func.func @sms_never_attempted_trivial_loop
// CHECK: scf.for
// CHECK: arith.addi

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sms_never_attempted_trivial_loop() attributes {
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
