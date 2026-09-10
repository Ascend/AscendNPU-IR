// RUN: bishengir-opt %s --simt-insert-init-and-finish-for-debug | FileCheck %s

// A launched kernel inherits debug behavior through helper calls. Recursive
// helper cycles must terminate while preserving reachability to the print.
// CHECK-LABEL: llvm.func @transitive_wrapper() attributes {hivm_regbaseintrins.kernel} {
// CHECK: llvm.call @_mlir_ciface_init_debug() : () -> ()
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @transitive_kernel threads in
// CHECK-NEXT: llvm.call @_mlir_ciface_finish_debug() : () -> ()
// CHECK-NEXT: llvm.return
module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

  llvm.func @transitive_wrapper() attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @transitive_kernel threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @transitive_kernel() {
    llvm.call @recursive_helper_a() : () -> ()
    llvm.return
  }

  llvm.func @recursive_helper_a() {
    llvm.call @recursive_helper_b() : () -> ()
    llvm.return
  }

  llvm.func @recursive_helper_b() {
    llvm.call @recursive_helper_a() : () -> ()
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }
}
