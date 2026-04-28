// RUN: bishengir-opt %s -convert-ascend-dpx-to-hivmregbaseintrins -canonicalize --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @ascend_dpx_shfl_i32_lowering
// CHECK-SAME: %[[ARG0:.*]]: i32
func.func @ascend_dpx_shfl_i32_lowering(%arg0 : i32) -> (i32, i32, i32, i32) {
    %mask = llvm.mlir.constant(0 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %offset = llvm.mlir.constant(16 : i32) : i32

    // CHECK-NEXT: %[[OPS_CONST:.*]] =
    // CHECK-SAME: 7952

    // CHECK: %[[RES_UP:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.up.i32"(%[[ARG0]], %[[OPS_CONST]]) : (i32, i32) -> i32
    %res_up = ascend_dpx.shfl.up %arg0, %mask, %clamp, %offset : (i32, i32, i32, i32) -> i32

    // CHECK: %[[RES_DOWN:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.down.i32"(%[[ARG0]], %[[OPS_CONST]]) : (i32, i32) -> i32
    %res_down = ascend_dpx.shfl.down %arg0, %mask, %clamp, %offset : (i32, i32, i32, i32) -> i32

    // CHECK: %[[RES_BFLY:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.bfly.i32"(%[[ARG0]], %[[OPS_CONST]]) : (i32, i32) -> i32
    %res_bfly = ascend_dpx.shfl.bfly %arg0, %mask, %clamp, %offset : (i32, i32, i32, i32) -> i32

    // CHECK: %[[RES_IDX:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.idx.i32"(%[[ARG0]], %[[OPS_CONST]]) : (i32, i32) -> i32
    %res_idx = ascend_dpx.shfl.idx %arg0, %mask, %clamp, %offset : (i32, i32, i32, i32) -> i32

    return %res_up, %res_down, %res_bfly, %res_idx : i32, i32, i32, i32
}

// CHECK-LABEL: @ascend_dpx_shfl_i64_lowering
// CHECK-SAME: %[[ARG0:.*]]: i64
func.func @ascend_dpx_shfl_i64_lowering(%arg0 : i64) -> (i64, i64, i64, i64) {
    %mask = llvm.mlir.constant(0 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %offset = llvm.mlir.constant(16 : i32) : i32

    // CHECK-NEXT: %[[OPS_CONST:.*]] =
    // CHECK-SAME: 7952

    // CHECK: %[[RES_UP:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.up.i64"(%[[ARG0]], %[[OPS_CONST]]) : (i64, i32) -> i64
    %res_up = ascend_dpx.shfl.up %arg0, %mask, %clamp, %offset : (i64, i32, i32, i32) -> i64

    // CHECK: %[[RES_DOWN:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.down.i64"(%[[ARG0]], %[[OPS_CONST]]) : (i64, i32) -> i64
    %res_down = ascend_dpx.shfl.down %arg0, %mask, %clamp, %offset : (i64, i32, i32, i32) -> i64

    // CHECK: %[[RES_BFLY:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.bfly.i64"(%[[ARG0]], %[[OPS_CONST]]) : (i64, i32) -> i64
    %res_bfly = ascend_dpx.shfl.bfly %arg0, %mask, %clamp, %offset : (i64, i32, i32, i32) -> i64

    // CHECK: %[[RES_IDX:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.idx.i64"(%[[ARG0]], %[[OPS_CONST]]) : (i64, i32) -> i64
    %res_idx = ascend_dpx.shfl.idx %arg0, %mask, %clamp, %offset : (i64, i32, i32, i32) -> i64

    return %res_up, %res_down, %res_bfly, %res_idx : i64, i64, i64, i64
}

// CHECK-LABEL: @ascend_dpx_shfl_f32_lowering
// CHECK-SAME: %[[ARG0:.*]]: f32
func.func @ascend_dpx_shfl_f32_lowering(%arg0 : f32) -> (f32, f32, f32, f32) {
    %mask = llvm.mlir.constant(0 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %offset = llvm.mlir.constant(16 : i32) : i32

    // CHECK-NEXT: %[[OPS_CONST:.*]] =
    // CHECK-SAME: 7952

    // CHECK: %[[RES_UP:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.up.f32"(%[[ARG0]], %[[OPS_CONST]]) : (f32, i32) -> f32
    %res_up = ascend_dpx.shfl.up %arg0, %mask, %clamp, %offset : (f32, i32, i32, i32) -> f32

    // CHECK: %[[RES_DOWN:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.down.f32"(%[[ARG0]], %[[OPS_CONST]]) : (f32, i32) -> f32
    %res_down = ascend_dpx.shfl.down %arg0, %mask, %clamp, %offset : (f32, i32, i32, i32) -> f32

    // CHECK: %[[RES_BFLY:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.bfly.f32"(%[[ARG0]], %[[OPS_CONST]]) : (f32, i32) -> f32
    %res_bfly = ascend_dpx.shfl.bfly %arg0, %mask, %clamp, %offset : (f32, i32, i32, i32) -> f32

    // CHECK: %[[RES_IDX:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.idx.f32"(%[[ARG0]], %[[OPS_CONST]]) : (f32, i32) -> f32
    %res_idx = ascend_dpx.shfl.idx %arg0, %mask, %clamp, %offset : (f32, i32, i32, i32) -> f32

    return %res_up, %res_down, %res_bfly, %res_idx : f32, f32, f32, f32
}

// CHECK-LABEL: @ascend_dpx_shfl_f16_lowering
// CHECK-SAME: %[[ARG0:.*]]: f16
func.func @ascend_dpx_shfl_f16_lowering(%arg0 : f16) -> (f16, f16, f16, f16) {
    %mask = llvm.mlir.constant(0 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %offset = llvm.mlir.constant(16 : i32) : i32

    // CHECK-NEXT: %[[OPS_CONST:.*]] =
    // CHECK-SAME: 7952

    // CHECK: %[[RES_UP:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.up.f16"(%[[ARG0]], %[[OPS_CONST]]) : (f16, i32) -> f16
    %res_up = ascend_dpx.shfl.up %arg0, %mask, %clamp, %offset : (f16, i32, i32, i32) -> f16

    // CHECK: %[[RES_DOWN:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.down.f16"(%[[ARG0]], %[[OPS_CONST]]) : (f16, i32) -> f16
    %res_down = ascend_dpx.shfl.down %arg0, %mask, %clamp, %offset : (f16, i32, i32, i32) -> f16

    // CHECK: %[[RES_BFLY:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.bfly.f16"(%[[ARG0]], %[[OPS_CONST]]) : (f16, i32) -> f16
    %res_bfly = ascend_dpx.shfl.bfly %arg0, %mask, %clamp, %offset : (f16, i32, i32, i32) -> f16

    // CHECK: %[[RES_IDX:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.idx.f16"(%[[ARG0]], %[[OPS_CONST]]) : (f16, i32) -> f16
    %res_idx = ascend_dpx.shfl.idx %arg0, %mask, %clamp, %offset : (f16, i32, i32, i32) -> f16

    return %res_up, %res_down, %res_bfly, %res_idx : f16, f16, f16, f16
}

// CHECK-LABEL: @ascend_dpx_shfl_v2f16_lowering
// CHECK-SAME: %[[ARG0:.*]]: vector<2xf16>
func.func @ascend_dpx_shfl_v2f16_lowering(%arg0 : vector<2xf16>) -> (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>) {
    %mask = llvm.mlir.constant(0 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %offset = llvm.mlir.constant(16 : i32) : i32

    // CHECK-NEXT: %[[OPS_CONST:.*]] =
    // CHECK-SAME: 7952

    // CHECK: %[[RES_UP:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.up.v2f16"(%[[ARG0]], %[[OPS_CONST]]) : (vector<2xf16>, i32) -> vector<2xf16>
    %res_up = ascend_dpx.shfl.up %arg0, %mask, %clamp, %offset : (vector<2xf16>, i32, i32, i32) -> vector<2xf16>

    // CHECK: %[[RES_DOWN:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.down.v2f16"(%[[ARG0]], %[[OPS_CONST]]) : (vector<2xf16>, i32) -> vector<2xf16>
    %res_down = ascend_dpx.shfl.down %arg0, %mask, %clamp, %offset : (vector<2xf16>, i32, i32, i32) -> vector<2xf16>

    // CHECK: %[[RES_BFLY:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.bfly.v2f16"(%[[ARG0]], %[[OPS_CONST]]) : (vector<2xf16>, i32) -> vector<2xf16>
    %res_bfly = ascend_dpx.shfl.bfly %arg0, %mask, %clamp, %offset : (vector<2xf16>, i32, i32, i32) -> vector<2xf16>

    // CHECK: %[[RES_IDX:.*]] = "hivm_regbaseintrins.intr.hivm.shfl.idx.v2f16"(%[[ARG0]], %[[OPS_CONST]]) : (vector<2xf16>, i32) -> vector<2xf16>
    %res_idx = ascend_dpx.shfl.idx %arg0, %mask, %clamp, %offset : (vector<2xf16>, i32, i32, i32) -> vector<2xf16>

    return %res_up, %res_down, %res_bfly, %res_idx : vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>
}
