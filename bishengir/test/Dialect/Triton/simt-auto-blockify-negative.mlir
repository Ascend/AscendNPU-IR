// RUN: not bishengir-opt -simt-auto-blockify %s 2>&1 | FileCheck %s --check-prefix=GRID

// GRID: error: failed to get grid args for SIMT auto blockify
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @missing_grid_args(%arg0: i32) {
    %pid_x = tt.get_program_id x : i32
    %sum = arith.addi %pid_x, %arg0 : i32
    %cmp = arith.cmpi sgt, %sum, %arg0 : i32
    scf.if %cmp {
    }
    tt.return
  }
}
