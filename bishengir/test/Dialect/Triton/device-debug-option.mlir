// RUN: bishengir-opt %s --erase-triton-debug-ops | FileCheck %s --check-prefix=ERASED
// RUN: bishengir-opt %s -lower-triton-pipeline | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt %s "-lower-triton-pipeline=enable-simt-device-debug=true" | FileCheck %s --check-prefix=ENABLED

// ERASED-LABEL: tt.func public @debug_ops
// ERASED-NOT: tt.print
// ERASED-NOT: tt.assert
// ERASED: tt.return

// DISABLED-NOT: _debug_prefix_
// DISABLED-NOT: _debug_msg_
// DISABLED-LABEL: llvm.func @debug_ops
// DISABLED-NOT: _mlir_ciface_print_
// DISABLED-NOT: _mlir_ciface_assert_
// DISABLED: llvm.return

// ENABLED: llvm.mlir.global private constant @_debug_msg_
// ENABLED: llvm.mlir.global private constant @_debug_prefix_
// ENABLED-LABEL: llvm.func @debug_ops
// ENABLED: llvm.call @_mlir_ciface_print_scalar_int32_t_gm_simt
// ENABLED: llvm.call @_mlir_ciface_assert_scalar_bool_gm_simt

module {
  tt.func public @debug_ops(%condition: i1, %value: i32)
      attributes {noinline = false} {
    tt.print "value: " {hex = false, isSigned = array<i32: 1>}
      : %value : i32
    tt.assert %condition, "condition failed" : i1
    tt.return
  }
}
