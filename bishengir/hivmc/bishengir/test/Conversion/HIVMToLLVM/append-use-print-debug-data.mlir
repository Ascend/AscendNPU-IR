// RUN: bishengir-opt %s -split-input-file -append-use-print-debug-data | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @_mlir_ciface_init_debug(!llvm.ptr<1>)
  // CHECK: llvm.func @_mlir_ciface_finish_debug(!llvm.ptr<1>)
  llvm.func @_mlir_ciface_init_debug() attributes {hacc.always_inline, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @_mlir_ciface_finish_debug() attributes {hacc.always_inline, llvm.emit_c_interface, sym_visibility = "private"}
  // CHECK: llvm.func @print_kernel(%arg0: !llvm.ptr<1>)
  // CHECK-SAME: print_arg_idx = 0 : i64
  llvm.func @print_kernel() attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    // CHECK: llvm.call @_mlir_ciface_init_debug(%arg0) : (!llvm.ptr<1>) -> ()
    // CHECK: llvm.call @_mlir_ciface_finish_debug(%arg0) : (!llvm.ptr<1>) -> ()
    llvm.call @_mlir_ciface_init_debug() : () -> ()
    llvm.call @_mlir_ciface_finish_debug() : () -> ()
    llvm.return
  }

}
