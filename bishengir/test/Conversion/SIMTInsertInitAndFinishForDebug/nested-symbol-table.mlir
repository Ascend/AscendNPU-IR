// RUN: bishengir-opt %s --simt-insert-init-and-finish-for-debug | FileCheck %s

// Each launch and helper call resolves from its nearest symbol table.
// CHECK-LABEL: module @nested_scope {
// CHECK: llvm.func @nested_wrapper() attributes {hivm_regbaseintrins.kernel} {
// CHECK: llvm.call @_mlir_ciface_init_debug() : () -> ()
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @nested_kernel threads in
// CHECK-NEXT: llvm.call @_mlir_ciface_finish_debug() : () -> ()
// CHECK-NEXT: llvm.return
module {
  module @nested_scope {
    llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

    llvm.func @nested_wrapper() attributes {hivm_regbaseintrins.kernel} {
      %c1 = llvm.mlir.constant(1 : i64) : i64
      hivm_regbaseintrins.intrins.launch_func @nested_kernel threads in (%c1, %c1, %c1)
      llvm.return
    }

    llvm.func @nested_kernel() {
      llvm.call @nested_helper() : () -> ()
      llvm.return
    }

    llvm.func @nested_helper() {
      llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
      llvm.return
    }
  }
}
