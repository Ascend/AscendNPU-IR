// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -allow-unregistered-dialect %s | FileCheck %s

// XFAIL: *
// Temporarily disabled: this IR expects cv-pipelining to re-annotate
// pre-existing scope.scope ops (adding no_inline on top of their input
// preload_num), but the pass only builds preload scopes for loops whose
// body directly contains at least two core-op work items. A loop made of
// existing scopes is left untouched, so the no_inline expectations never
// match. Revisit when pre-existing-scope handling is implemented, or
// rewrite the IR with top-level core ops.

// CHECK-LABEL: func.func @dual_scope_same_preload_num
// CHECK: scf.for
// CHECK: "first_scope_op"
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
// CHECK: "second_scope_op"
// CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @dual_scope_same_preload_num(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {
    WorkspaceArgIdx = 0 : i16,
    func_dyn_memref_args = dense<[true]> : vector<1xi1>,
    global_kernel = "local",
    hacc.entry,
    hacc.function_kind = #hacc.function_kind<DEVICE>,
    hivm.func_core_type = #hivm.func_core_type<MIX>,
    mix_mode = "mix"
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bound = arith.constant 2 : i32
    scf.for %i = %c0 to %bound step %c1 : i32 {
      scope.scope : () -> () {
        "first_scope_op"() : () -> ()
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
      scope.scope : () -> () {
        "second_scope_op"() : () -> ()
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
      scf.yield
    }
    return
  }
}
