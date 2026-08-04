// RUN: bishengir-opt %s --split-input-file --simt-insert-init-and-finish-for-debug --verify-diagnostics

module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

  // expected-error @+1 {{cannot insert one debug lifecycle around launches in different blocks}}
  llvm.func @non_linear_wrapper(%condition: i1) attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    llvm.cond_br %condition, ^bb1, ^bb2
  ^bb1:
    hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in (%c1, %c1, %c1)
    llvm.br ^bb3
  ^bb2:
    hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in (%c1, %c1, %c1)
    llvm.br ^bb3
  ^bb3:
    llvm.return
  }

  llvm.func @simt_kernel() {
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }
}

// -----

// A partial LLVM lifecycle must not be completed with another generated pair.
module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()
  llvm.func @_mlir_ciface_init_debug() attributes {llvm.emit_c_interface}

  // expected-error @+1 {{malformed debug lifecycle: expected exactly one same-kind init/finish pair bracketing all relevant launches}}
  llvm.func @partial_lifecycle_wrapper() attributes {hivm_regbaseintrins.kernel} {
    llvm.call @_mlir_ciface_init_debug() : () -> ()
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @partial_lifecycle_kernel threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @partial_lifecycle_kernel() {
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }
}

// -----

// A finish between two relevant launches does not bracket their full range.
module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

  // expected-error @+1 {{malformed debug lifecycle: expected exactly one same-kind init/finish pair bracketing all relevant launches}}
  llvm.func @early_finish_wrapper() attributes {hivm_regbaseintrins.kernel} {
    hivm.hir.init_debug
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @early_finish_kernel threads in (%c1, %c1, %c1)
    hivm.hir.finish_debug
    hivm_regbaseintrins.intrins.launch_func @early_finish_kernel threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @early_finish_kernel() {
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }
}

// -----

// Reversed markers are malformed rather than permission to add side effects.
module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

  // expected-error @+1 {{malformed debug lifecycle: expected exactly one same-kind init/finish pair bracketing all relevant launches}}
  llvm.func @misordered_lifecycle_wrapper() attributes {hivm_regbaseintrins.kernel} {
    hivm.hir.finish_debug
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @misordered_lifecycle_kernel threads in (%c1, %c1, %c1)
    hivm.hir.init_debug
    llvm.return
  }

  llvm.func @misordered_lifecycle_kernel() {
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }
}
