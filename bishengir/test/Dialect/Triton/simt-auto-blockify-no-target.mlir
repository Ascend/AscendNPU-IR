// RUN: not bishengir-opt -simt-auto-blockify %s 2>&1 | FileCheck %s --check-prefix=NOSPEC

// SIMT auto blockify needs VECTOR_CORE_COUNT from the NPU target spec to
// clamp the physical grid. When the module has no `hacc.target` (and thus
// no NPUTargetSpec), the pass emits an error and aborts compilation.

// NOSPEC: error: failed to infer physical vector core count for SIMT auto blockify
module {
  tt.func public @no_target_spec(
      %arg0: i32 {tt.divisibility = 16 : i32},
      %arg1: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32},
      %arg2: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32},
      %arg3: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) {
    %pid_x = tt.get_program_id x : i32
    %sum = arith.addi %pid_x, %arg0 : i32
    %cmp = arith.cmpi sgt, %sum, %arg0 : i32
    scf.if %cmp {
    }
    tt.return
  }
}
