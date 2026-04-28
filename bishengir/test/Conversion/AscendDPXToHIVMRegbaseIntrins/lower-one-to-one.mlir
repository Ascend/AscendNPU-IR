// RUN: bishengir-opt %s -convert-ascend-dpx-to-hivmregbaseintrins --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @thread_idx_lowering
func.func @thread_idx_lowering() -> (i32, i32, i32) {
    // CHECK-NEXT: hivm_regbaseintrins.thread_id_x
    %tidx = "ascend_dpx.thread_id_x"() : () -> i32
    // CHECK-NEXT: hivm_regbaseintrins.thread_id_y
    %tidy = "ascend_dpx.thread_id_y"() : () -> i32
    // CHECK-NEXT: hivm_regbaseintrins.thread_id_z
    %tidz = "ascend_dpx.thread_id_z"() : () -> i32
    return %tidx, %tidy, %tidz : i32, i32, i32
}

// CHECK-LABEL: @block_dim_lowering
func.func @block_dim_lowering() -> (i32, i32, i32) {
    // CHECK-NEXT: hivm_regbaseintrins.block_dim_x
    %bdimx = ascend_dpx.block_dim_x
    // CHECK-NEXT: hivm_regbaseintrins.block_dim_y
    %bdimy = ascend_dpx.block_dim_y
    // CHECK-NEXT: hivm_regbaseintrins.block_dim_z
    %bdimz = ascend_dpx.block_dim_z
    return %bdimx, %bdimy, %bdimz : i32, i32, i32
}


// CHECK-LABEL: @sync_op_lowering
func.func @sync_op_lowering() {
    // CHECK-NEXT: hivm_regbaseintrins.sync_threads
    ascend_dpx.sync_threads
    return
}
