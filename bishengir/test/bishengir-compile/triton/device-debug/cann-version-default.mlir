// REQUIRES: enable_triton_ir_compiless
// RUN: split-file %s %t
// RUN: cp -r %t/cann91/x86_64-linux %t/cann91/aarch64-linux
// RUN: cp -r %t/beta1/x86_64-linux %t/beta1/aarch64-linux
// RUN: cp -r %t/beta2/x86_64-linux %t/beta2/aarch64-linux

// DEFINE: %{compile} = bishengir-compile %t/input.mlir --target=Ascend950PR_9579 --pure-simt --enable-triton-ir-compile --enable-triton-kernel-compile --only-run-hivm-pipeline=true --shared-mem-dynamic-size=122880 -o %t/output.mlir
// DEFINE: %{erased} = FileCheck %s --check-prefix=ERASED --implicit-check-not=_mlir_ciface_print --implicit-check-not=_mlir_ciface_assert < %t/output.mlir
// DEFINE: %{kept} = FileCheck %s --check-prefix=KEPT < %t/output.mlir

// RUN: env ASCEND_HOME_PATH=%t/cann91 %{compile}
// RUN: %{erased}
// RUN: env ASCEND_HOME_PATH=%t/beta1 %{compile}
// RUN: %{erased}
// RUN: env ASCEND_HOME_PATH=%t/beta2 %{compile}
// RUN: %{kept}
// RUN: env -u ASCEND_HOME_PATH %{compile}
// RUN: %{kept}
// RUN: env ASCEND_HOME_PATH=%t/cann91 %{compile} --enable-simt-device-debug=true
// RUN: %{kept}
// RUN: env ASCEND_HOME_PATH=%t/beta2 %{compile} --enable-simt-device-debug=false
// RUN: %{erased}

// ERASED: llvm.func{{.*}}@debug_ops
// KEPT: llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt
// KEPT: llvm.call @_mlir_ciface_assert_scalar_bool_gm_simt

//--- input.mlir
module {
  tt.func public @debug_ops(%condition: i1, %value: i32)
      attributes {noinline = false} {
    tt.print "value: " {hex = false, isSigned = array<i32: 1>}
      : %value : i32
    tt.assert %condition, "condition failed" : i1
    tt.return
  }
}

//--- cann91/x86_64-linux/ascend_toolkit_install.info
version=9.1.0

//--- beta1/x86_64-linux/ascend_all_cann_install.info
Version=9.2.0-beta.1

//--- beta2/x86_64-linux/ascend_toolkit_install.info
compatible_version=9.1.0
version=9.2.0-beta.2
