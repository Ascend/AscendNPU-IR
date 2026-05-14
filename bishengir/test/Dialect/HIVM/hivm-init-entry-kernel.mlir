// RUN: bishengir-opt %s --hivm-init-entry-kernel -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend310B4 --hivm-init-entry-kernel -split-input-file | FileCheck --check-prefix=Ascend310B4 %s

// -----
// CHECK-LABEL: func.func @entryKernelMembase
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
    func.func @entryKernelMembase() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
        // CHECK: hivm.hir.set_mask_norm
        return
    }
}

// -----
// Ascend310B4-LABEL: func.func @entryKernelRegbase
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
    func.func @entryKernelRegbase() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
        // Ascend310B4: hivm.hir.set_ctrl false at ctrl[60]
        return
    }
}

// -----
// Ascend310B4-LABEL: func.func @entryKernelRegbaseMultiReturn
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
    func.func @entryKernelRegbaseMultiReturn(%flag: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
        // Ascend310B4: hivm.hir.set_ctrl false at ctrl[60]
        // Ascend310B4-NEXT: hivm.hir.set_ctrl true at ctrl[48]
        cf.cond_br %flag, ^bb1, ^bb2
      ^bb1:
        // Ascend310B4: hivm.hir.set_ctrl true at ctrl[60]
        // Ascend310B4-NEXT: return
        return
      ^bb2:
        // Ascend310B4: hivm.hir.set_ctrl true at ctrl[60]
        // Ascend310B4-NEXT: return
        return
    }
}
