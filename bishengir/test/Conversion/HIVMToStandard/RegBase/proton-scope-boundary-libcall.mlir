// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910_9589 -convert-hivm-to-std="mark-libcall-noinline=true" | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -convert-hivm-to-std="mark-libcall-noinline=true" | FileCheck %s

module {
  func.func @lower_scope_boundary() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %start = arith.constant 0 : i32
    %end = arith.constant 1 : i32
    hivm.hir.custom
        {hivm.inline_mode = #hivm.inline_mode<always_inline>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>}
        "__builtin_proton_scope_boundary" ins(%start : i32)
    hivm.hir.custom
        {hivm.inline_mode = #hivm.inline_mode<always_inline>,
         hivm.pipe = #hivm.pipe<PIPE_S>,
         hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>}
        "__builtin_proton_scope_boundary" ins(%end : i32)
    return
  }
}

// One private declaration serves every boundary. The builtin is legal on both
// AIC and AIV, so the declaration stays AIC_OR_AIV and always_inline.
// mark-libcall-noinline applies to AIV libcalls and to custom ops that do not
// set hivm.inline_mode.
// CHECK: func.func private @proton_scope_boundary(i32)
// CHECK-SAME: hacc.always_inline
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>
// CHECK-SAME: llvm.emit_c_interface
// CHECK-NOT: hacc.noinline
// CHECK-LABEL: func.func @lower_scope_boundary
// CHECK-DAG: %[[START:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[END:.*]] = arith.constant 1 : i32
// CHECK: call @proton_scope_boundary(%[[START]]) : (i32) -> ()
// CHECK: call @proton_scope_boundary(%[[END]]) : (i32) -> ()
// CHECK-NOT: hivm.hir.custom
// CHECK-NOT: func.func private
