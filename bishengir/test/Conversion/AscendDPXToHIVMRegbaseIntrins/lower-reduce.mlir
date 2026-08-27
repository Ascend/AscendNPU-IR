// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck --enable-var-scope %s

// CHECK-LABEL: @ascend_dpx_redux_s32_lowering
func.func @ascend_dpx_redux_s32_lowering(%arg1 : i32) {
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.add.s32
    %0 = ascend_dpx.reduce.add %arg1 : (i32) -> i32
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.max.s32
    %1 = ascend_dpx.reduce.max %arg1 : (i32) -> i32
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.min.s32
    %2 = ascend_dpx.reduce.min %arg1 : (i32) -> i32
    return
}

// CHECK-LABEL: @ascend_dpx_redux_u32_lowering
func.func @ascend_dpx_redux_u32_lowering(%arg1 : i32) {
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.max.u32
    %0 = ascend_dpx.reduce.umax %arg1 : (i32) -> i32
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.min.u32
    %1 = ascend_dpx.reduce.umin %arg1 : (i32) -> i32
    return
}

// CHECK-LABEL: @ascend_dpx_redux_f32_lowering
func.func @ascend_dpx_redux_f32_lowering(%arg1 : f32) {
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.add.f32
    %0 = ascend_dpx.reduce.add %arg1 : (f32) -> f32
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.max.f32
    %1 = ascend_dpx.reduce.max %arg1 : (f32) -> f32
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.min.f32
    %2 = ascend_dpx.reduce.min %arg1 : (f32) -> f32
    return
}

// CHECK-LABEL: @ascend_dpx_redux_f16_lowering
func.func @ascend_dpx_redux_f16_lowering(%arg1 : f16) {
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.add.f16
    %0 = ascend_dpx.reduce.add %arg1 : (f16) -> f16
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.max.f16
    %1 = ascend_dpx.reduce.max %arg1 : (f16) -> f16
    // CHECK: hivm_regbaseintrins.intr.hivm.redux.min.f16
    %2 = ascend_dpx.reduce.min %arg1 : (f16) -> f16
    return
}
