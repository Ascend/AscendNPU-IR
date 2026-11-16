// RUN: bishengir-opt -simt-auto-blockify --mlir-print-debuginfo %s | FileCheck %s

// The blockify prologue has no corresponding source construct, so its bound
// computation deliberately carries no source location; otherwise the debugger
// would stop on a user line while executing unrelated arithmetic. The logical
// program-id computation that replaces `tt.get_program_id` keeps the location
// of that op.

// CHECK-LABEL: tt.func public @blockify_loc(
// CHECK:       %[[YZ:.*]] = arith.muli {{.*}} loc(#[[SCAFFOLD:loc[0-9]+]])
// CHECK:       %[[LOGICAL:.*]] = arith.muli {{.*}} loc(#[[SCAFFOLD]])
// CHECK:       gpu.linear_block_id{{.*}} loc(#[[SCAFFOLD]])
// CHECK:       arith.ceildivui {{.*}} loc(#[[SCAFFOLD]])
// CHECK:       arith.minui {{.*}} loc(#[[SCAFFOLD]])
// CHECK:       arith.divui {{.*}} loc(#[[PID:loc[0-9]+]])
// CHECK:       arith.remui {{.*}} loc(#[[PID]])
// CHECK-DAG:   #[[SCAFFOLD]] = loc(unknown)
// CHECK-DAG:   #[[PID]] = loc("blockify_source.py":10:30)
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @blockify_loc(
      %arg1: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32},
      %arg2: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32},
      %arg3: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) {
    %pid_x = tt.get_program_id x : i32 loc("blockify_source.py":10:30)
    %sum = arith.addi %pid_x, %arg1 : i32
    tt.return
  }
}
