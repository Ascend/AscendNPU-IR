// RUN: bishengir-opt %s -convert-hacc-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL:  llvm.func @Fused_Add_fusion_17734579009022343263
// CHECK: llvm.switch 
// CHECK: 2: ^bb1,
// CHECK: 1: ^bb2,
// CHECK: 0: ^bb3
// CHECK: llvm.br
// CHECK: llvm.return

module {
  llvm.func @Fused_Add_fusion_17734579009022343263(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<1>, %arg12: !llvm.ptr<1>, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr<1>, %arg21: !llvm.ptr<1>, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !llvm.ptr<1>, %arg32: !llvm.ptr<1>, %arg33: i64, %arg34: i64, %arg35: i64) attributes {OperatorType = "Broadcast", compute_capability = "", frontend_symbol = {input_0 = ["s710", "s711", "s712", "1152"], input_1 = ["1", "s713", "1152"], output_0 = ["s710", "s711", "s714", "1152"]}, hacc.function_kind = #hacc.function_kind<HOST>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, mindspore_kernel, process = "aicore"} {
    %0 = llvm.mlir.constant(1152 : index) : i64
    %1 = llvm.load %arg32 : !llvm.ptr<1> -> i64
    %2 = llvm.mul %arg4, %arg5  : i64
    %3 = llvm.mul %arg15, %0  : i64
    %4 = llvm.udiv %3, %0  : i64
    %5 = llvm.trunc %1 : i64 to i32
    llvm.switch %5 : i32, ^bb4 [
      2: ^bb1,
      1: ^bb2,
      0: ^bb3
    ]
  ^bb1:  // pred: ^bb0
    %6 = llvm.mul %arg2, %arg3 : i64
    llvm.br ^bb5
  ^bb2:  // pred: ^bb0
	%7 = llvm.mul %arg6, %arg7 : i64
    llvm.br ^bb5
  ^bb3:  // pred: ^bb0
	%8 = llvm.mul %arg8, %arg9 : i64
    llvm.br ^bb5
  ^bb4:  // pred: ^bb0
    llvm.br ^bb5
  ^bb5:  // 4 preds: ^bb1, ^bb2, ^bb3, ^bb4
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @condBr
// CHECK: llvm.cond_br
// CHECK: llvm.return

module {
  llvm.func @condBr(%arg0: i1, %arg1: i32) {
    %0 = llvm.mlir.constant(10 : i32) : i32
    %1 = llvm.mlir.constant(7 : i32) : i32
    llvm.cond_br %arg0, ^bb1, ^bb2
    ^bb1: // pred: ^bb0
      %2 = llvm.mul %arg1, %0 : i32
      llvm.return
    ^bb2: // pred: ^bb0
      %3 = llvm.mul %arg1, %1 : i32
      llvm.return
  }
}