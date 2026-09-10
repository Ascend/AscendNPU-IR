// RUN: bishengir-opt %s --simt-insert-init-and-finish-for-debug --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --simt-insert-init-and-finish-for-debug --simt-insert-init-and-finish-for-debug --split-input-file | FileCheck %s --check-prefix=IDEMPOTENT

// No lowered debug call leaves the module untouched.
// CHECK-LABEL: llvm.func @no_debug_wrapper() attributes {hivm_regbaseintrins.kernel} {
// CHECK-NOT: _mlir_ciface_init_debug
// CHECK-NOT: _mlir_ciface_finish_debug
// CHECK-NEXT: llvm.return
// CHECK-NOT: _mlir_ciface_init_debug
// CHECK-NOT: _mlir_ciface_finish_debug
module {
  llvm.func @no_debug_wrapper() attributes {hivm_regbaseintrins.kernel} {
    llvm.return
  }
}

// -----

// A lowered print creates the missing declarations only for wrappers that
// launch its kernel. The lifecycle begins at the first relevant launch and
// ends after the last relevant launch.
// CHECK-LABEL: llvm.func @not_a_wrapper() {
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @wrapper_without_launch() attributes {hivm_regbaseintrins.kernel} {
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK-NEXT: llvm.return
// CHECK-LABEL: llvm.func @wrapper_with_one_launch() attributes {hivm_regbaseintrins.kernel} {
// CHECK: llvm.call @_mlir_ciface_init_debug() : () -> ()
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in
// CHECK-NEXT: llvm.call @_mlir_ciface_finish_debug() : () -> ()
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @wrapper_with_unrelated_launch() attributes {hivm_regbaseintrins.kernel} {
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK-NEXT: llvm.return
// CHECK-LABEL: llvm.func @wrapper_with_mixed_launches() attributes {hivm_regbaseintrins.kernel} {
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in
// CHECK-NEXT: llvm.call @_mlir_ciface_init_debug() : () -> ()
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in
// CHECK-NEXT: llvm.call @_mlir_ciface_finish_debug() : () -> ()
// CHECK-NEXT: hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_finish_debug
// CHECK-NEXT: llvm.return
// CHECK: llvm.func @_mlir_ciface_init_debug() attributes {llvm.emit_c_interface}
// CHECK: llvm.func @_mlir_ciface_finish_debug() attributes {llvm.emit_c_interface}
module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

  llvm.func @not_a_wrapper() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @wrapper_without_launch() attributes {hivm_regbaseintrins.kernel} {
    llvm.return
  }

  llvm.func @wrapper_with_one_launch() attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @wrapper_with_unrelated_launch() attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @wrapper_with_mixed_launches() attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in (%c1, %c1, %c1)
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in (%c1, %c1, %c1)
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in (%c1, %c1, %c1)
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_with_debug threads in (%c1, %c1, %c1)
    hivm_regbaseintrins.intrins.launch_func @simt_kernel_without_debug threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @simt_kernel_with_debug() {
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }

  llvm.func @simt_kernel_without_debug() {
    llvm.return
  }
}

// -----

// An assertion is a debug use too, and existing declarations are reused.
// CHECK: llvm.func @_mlir_ciface_init_debug() attributes {llvm.emit_c_interface}
// CHECK: llvm.func @_mlir_ciface_finish_debug() attributes {llvm.emit_c_interface}
// CHECK-LABEL: llvm.func @assert_wrapper() attributes {hivm_regbaseintrins.kernel} {
// CHECK: llvm.call @_mlir_ciface_init_debug() : () -> ()
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in
// CHECK-NEXT: llvm.call @_mlir_ciface_finish_debug() : () -> ()
// CHECK-NEXT: llvm.return
module {
  llvm.func @_mlir_ciface_assert_scalar_bool_gm_simt()
  llvm.func @_mlir_ciface_init_debug() attributes {llvm.emit_c_interface}
  llvm.func @_mlir_ciface_finish_debug() attributes {llvm.emit_c_interface}

  llvm.func @assert_wrapper() attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @simt_kernel() {
    llvm.call @_mlir_ciface_assert_scalar_bool_gm_simt() : () -> ()
    llvm.return
  }
}

// -----

// An existing complete high-level lifecycle is left unchanged.
// CHECK-LABEL: llvm.func @already_guarded_wrapper() attributes {hivm_regbaseintrins.kernel} {
// CHECK-NEXT: hivm.hir.init_debug
// CHECK-NOT: llvm.call @_mlir_ciface_init_debug
// CHECK: hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in
// CHECK-NEXT: hivm.hir.finish_debug
// CHECK-NEXT: llvm.return
module {
  llvm.func @_mlir_ciface_assert_scalar_bool_gm_simt()

  llvm.func @already_guarded_wrapper() attributes {hivm_regbaseintrins.kernel} {
    hivm.hir.init_debug
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in (%c1, %c1, %c1)
    hivm.hir.finish_debug
    llvm.return
  }

  llvm.func @simt_kernel() {
    llvm.call @_mlir_ciface_assert_scalar_bool_gm_simt() : () -> ()
    llvm.return
  }
}

// -----

// Rerunning the pass must not duplicate calls it inserted on the first run.
// IDEMPOTENT-LABEL: llvm.func @rerun_wrapper() attributes {hivm_regbaseintrins.kernel} {
// IDEMPOTENT: llvm.call @_mlir_ciface_init_debug() : () -> ()
// IDEMPOTENT-NEXT: hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in
// IDEMPOTENT-NEXT: llvm.call @_mlir_ciface_finish_debug() : () -> ()
// IDEMPOTENT-NEXT: llvm.return
module {
  llvm.func @_mlir_ciface_print_scalar_int32_t_gm_simt()

  llvm.func @rerun_wrapper() attributes {hivm_regbaseintrins.kernel} {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    hivm_regbaseintrins.intrins.launch_func @simt_kernel threads in (%c1, %c1, %c1)
    llvm.return
  }

  llvm.func @simt_kernel() {
    llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt() : () -> ()
    llvm.return
  }
}
