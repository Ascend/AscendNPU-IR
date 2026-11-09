// RUN: bishengir-opt %s -triton-globalkernelargs-to-llvm -triton-globalkernelargs-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @instrumented
// CHECK-SAME: proton.instrumentation_epilogue
// CHECK: %[[DRAIN:.*]] = llvm.mlir.constant(65536 : i32) : i32
// CHECK: llvm.call @proton_scope_boundary
// CHECK-NEXT: llvm.call @proton_scope_boundary(%[[DRAIN]]) : (i32) -> ()
// CHECK-NEXT: llvm.return
// CHECK-LABEL: llvm.func @plain
// CHECK-NOT: proton.instrumentation_epilogue
// CHECK-NOT: llvm.call @proton_scope_boundary
// CHECK: llvm.return
module {
  llvm.func @proton_scope_boundary(i32)
  llvm.func @instrumented() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %start = llvm.mlir.constant(1024 : i32) : i32
    llvm.call @proton_scope_boundary(%start) : (i32) -> ()
    llvm.return
  }
  llvm.func @plain() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    llvm.return
  }
}
