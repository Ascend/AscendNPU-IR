// RUN: bishengir-opt -simt-auto-blockify %s | FileCheck %s

// SIMT auto blockify silently skips (no error, no transform) when:
//   1. the function is not public (private helpers don't carry grid args)
//   2. the function body has no tt.get_program_id (no logical grid to remap)
//
// "Silent" means: no scf.for is introduced, no arith.muli for logical block
// count is emitted, no error is raised. The IR is returned unchanged.

// CHECK-NOT: scf.for
// CHECK-NOT: error:

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>>>, hacc.target = #hacc.target<"Ascend910_9589">} {

  // CHECK-LABEL: tt.func private @private_helper(
  // CHECK-NEXT: tt.get_program_id x : i32
  // CHECK-NEXT: tt.return
  tt.func private @private_helper(
      %arg0: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32},
      %arg1: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32},
      %arg2: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) {
    %pid = tt.get_program_id x : i32
    tt.return
  }

  // CHECK-LABEL: tt.func public @no_program_id(
  // CHECK-NOT: tt.get_program_id
  // CHECK: tt.return
  tt.func public @no_program_id(
      %arg0: i32 {tt.divisibility = 16 : i32},
      %arg1: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32},
      %arg2: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32},
      %arg3: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) {
    %c0 = arith.constant 0 : i32
    %cmp = arith.cmpi sgt, %arg0, %c0 : i32
    scf.if %cmp {
    }
    tt.return
  }
}
