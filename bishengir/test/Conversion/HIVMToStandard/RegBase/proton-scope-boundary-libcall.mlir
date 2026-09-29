// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910_9589 -convert-hivm-to-std="mark-libcall-noinline=true" | FileCheck %s --check-prefix=A5
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -convert-hivm-to-std | FileCheck %s --check-prefix=A3

module {
  func.func @lower_scope_boundary() {
    hivm.hir.proton_scope_boundary {region_id = 0 : i32}
    hivm.hir.proton_scope_boundary {region_id = 1 : i32}
    return
  }
}

// One private declaration serves every boundary. AIC_OR_AIV stays always_inline
// when mark-libcall-noinline is set; that flag applies only to AIV libcalls.
// A5: func.func private @_mlir_ciface_proton_scope_boundary(i32)
// A5-SAME: hacc.always_inline
// A5-SAME: hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>
// A5-SAME: llvm.emit_c_interface
// A5-NOT: hacc.noinline
// A5-LABEL: func.func @lower_scope_boundary
// A5-DAG: %[[START:.*]] = arith.constant 0 : i32
// A5-DAG: %[[END:.*]] = arith.constant 1 : i32
// A5: call @_mlir_ciface_proton_scope_boundary(%[[START]]) : (i32) -> ()
// A5: call @_mlir_ciface_proton_scope_boundary(%[[END]]) : (i32) -> ()
// A5-NOT: hivm.hir.proton_scope_boundary
// A5-NOT: func.func private

// A3: func.func @lower_scope_boundary
// A3: hivm.hir.proton_scope_boundary {region_id = 0 : i32}
// A3: hivm.hir.proton_scope_boundary {region_id = 1 : i32}
// A3-NOT: _mlir_ciface_proton_scope_boundary
