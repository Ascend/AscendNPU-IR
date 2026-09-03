

// RUN: bishengir-opt %s -split-input-file -triton-globalkernelargs-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @triton_directly_use_gm_args
  // CHECK-SAME: %arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>
  // CHECK-SAME: %arg10: i32
  llvm.func @triton_directly_use_gm_args(
              %arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64,
              %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: i64, %arg8: i64, %arg9: i64,
              %arg10: !llvm.ptr<1>, %arg11: !llvm.ptr<1>, %arg12: i64, %arg13: i64, %arg14: i64,
              %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>, %arg17: i64, %arg18: i64, %arg19: i64,
              %arg20: !llvm.ptr<1>, %arg21: !llvm.ptr<1>, %arg22: i64, %arg23: i64, %arg24: i64,
              %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32)
              attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
              func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>,
              hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv"} {
    llvm.return
  }
  // llvm.func @triton_directly_use_gm_args(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv"}
}

// -----

// CHECK-LABEL: llvm.func @triton_directly_use_gm_args
// CHECK-SAME: %arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>
// CHECK-SAME: %arg10: i32
llvm.func @triton_directly_use_gm_args(
            %arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64,
            %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: i64, %arg8: i64, %arg9: i64,
            %arg10: !llvm.ptr<1> {hacc.arg_type = #hacc.arg_type<gm_addr>}, %arg11: !llvm.ptr<1>, %arg12: i64, %arg13: i64, %arg14: i64,
            %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>, %arg17: i64, %arg18: i64, %arg19: i64,
            %arg20: !llvm.ptr<1>, %arg21: !llvm.ptr<1>, %arg22: i64, %arg23: i64, %arg24: i64,
            %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32, %arg30: i32)
            attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
            func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>,
            hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv"} {
  llvm.return
}
