// RUN: bishengir-opt %s --hivm-init-entry-kernel -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @entryKernel
func.func @entryKernel() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: hivm.hir.set_mask_norm
    return
}