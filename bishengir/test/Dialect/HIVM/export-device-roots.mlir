// RUN: bishengir-opt --hacc-append-device-spec=target=Ascend950PR_9579 --hivm-infer-vf-mode --hivm-init-entry-kernel %s | FileCheck %s

// Export roots receive compilation analysis. Launch setup and completion belong
// to their caller.
func.func private @vf() attributes {hivm.vector_function}

// CHECK-LABEL: func.func @exported()
// CHECK-SAME: export_device
// CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
// CHECK-SAME: hivm.vf_mode = #hivm.vf_mode<SIMD>
// CHECK-NOT: hacc.entry
// CHECK-NOT: hivm.hir.set_ctrl
// CHECK: call @vf()
// CHECK-NEXT: return
func.func @exported() attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  call @vf() : () -> ()
  return
}

// A helper is not promoted to an independent compilation root.
// CHECK-LABEL: func.func @helper()
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<DEVICE>}
// CHECK-NEXT: call @vf()
// CHECK-NEXT: return
func.func @helper() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  call @vf() : () -> ()
  return
}

// Existing launch entries still get mode inference and CTRL setup.
// CHECK-LABEL: func.func @launched()
// CHECK-SAME: hacc.entry
// CHECK-SAME: hivm.vf_mode = #hivm.vf_mode<SIMD>
// CHECK: hivm.hir.set_ctrl
func.func @launched() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  call @vf() : () -> ()
  return
}
