// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck --enable-var-scope %s

// CHECK-LABEL: @ascend_dpx_atomic_lowering_i32
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: i32
func.func @ascend_dpx_atomic_lowering_i32(%arg0 : !llvm.ptr<1>, %arg1 : i32) -> (i32, i32, i32) {
    // CHECK: %[[CONST0:.*]] =
    // CHECK-SAME: 0
    // CHECK-NEXT: hivm.atom.AND.G.s32
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST0]])
    %res1 = ascend_dpx.atomic_and %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NORMAL_FV > : (!llvm.ptr<1>, i32) -> i32
    // CHECK: %[[CONST1:.*]] =
    // CHECK-SAME: 1
    // CHECK-NEXT: hivm.atom.OR.G.s32
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST1]])
    %res2 = ascend_dpx.atomic_or %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NORMAL_LV > : (!llvm.ptr<1>, i32) -> i32
    // CHECK: %[[CONST2:.*]] =
    // CHECK-SAME: 2
    // CHECK-NEXT: hivm.atom.INC.G.s32
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST2]])
    %res3 = ascend_dpx.atomic_inc %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NORMAL_PERS > : (!llvm.ptr<1>, i32) -> i32
    return %res1, %res2, %res3 : i32, i32, i32
}

// CHECK-LABEL: @ascend_dpx_atomic_lowering_i64
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: i64
func.func @ascend_dpx_atomic_lowering_i64(%arg0 : !llvm.ptr<1>, %arg1 : i64) -> (i64, i64) {
    // CHECK: %[[CONST3:.*]] =
    // CHECK-SAME: 3
    // CHECK-NEXT: hivm.atom.XOR.G.s64
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST3]])
    %res1 = ascend_dpx.atomic_xor %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NORMAL_RED > : (!llvm.ptr<1>, i64) -> i64
    // CHECK: %[[CONST4:.*]] =
    // CHECK-SAME: 4
    // CHECK-NEXT: hivm.atom.DEC.G.s64
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST4]])
    %res2 = ascend_dpx.atomic_dec %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NOTALLOC_CI > : (!llvm.ptr<1>, i64) -> i64
    return %res1, %res2 : i64, i64
}

// CHECK-LABEL: @ascend_dpx_atomic_lowering_f16
// CHECK: llvm.hivm.atom.CAS.G.u32
func.func @ascend_dpx_atomic_lowering_f16(%arg0 : !llvm.ptr<1>, %arg1 : f16) -> f16 {
    %res1 = ascend_dpx.atomic_max %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NOTALLOC_PW > : (!llvm.ptr<1>, f16) -> f16
    return %res1 : f16
}

// CHECK-LABEL: @ascend_dpx_atomic_lowering_f32
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: f32
func.func @ascend_dpx_atomic_lowering_f32(%arg0 : !llvm.ptr<1>, %arg1 : f32) -> f32 {
    // CHECK: %[[CONST6:.*]] =
    // CHECK-SAME: 6
    // CHECK-NEXT: hivm.atom.MIN.G.fp32
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST6]])
    %res1 = ascend_dpx.atomic_min %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NOTALLOC_PI > : (!llvm.ptr<1>, f32) -> f32
    return %res1 : f32
}

// CHECK-LABEL: @ascend_dpx_atomic_lowering_bf16
// CHECK: llvm.hivm.atom.CAS.G.u32
func.func @ascend_dpx_atomic_lowering_bf16(%arg0 : !llvm.ptr<1>, %arg1 : bf16) -> bf16 {
    %res1 = ascend_dpx.atomic_min %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_NOTALLOC_RED > : (!llvm.ptr<1>, bf16) -> bf16
    return %res1 : bf16
}

// CHECK-LABEL: @ascend_dpx_atomic_lowering_f16x2
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: vector<2xf16>
func.func @ascend_dpx_atomic_lowering_f16x2(%arg0 : !llvm.ptr<1>, %arg1 : vector<2xf16>) -> vector<2xf16> {
    // CHECK: %[[CONST8:.*]] =
    // CHECK-SAME: 8
    // CHECK-NEXT: hivm.atom.ADD.G.f16x2
    // CHECK-SAME: (%[[ARG0]], %[[ARG1]], %[[CONST8]])
    %res1 = ascend_dpx.atomic_add %arg0, %arg1 cacheModifier = < L2_CACHE_HINT_WBH_FV > : (!llvm.ptr<1>, vector<2xf16>) -> vector<2xf16>
    return %res1 : vector<2xf16>
}

// CHECK-LABEL: @ascend_dpx_atomic_lowering_bf16x2
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
// CHECK-SAME: %[[ARG1:.*]]: vector<2xbf16>
func.func @ascend_dpx_atomic_lowering_bf16x2(%arg0 : !llvm.ptr<6>, %arg1 : vector<2xbf16>) -> vector<2xbf16> {
    // CHECK: hivm.atom.ADD.S.bf16x2
    // CHECK-SAME: %[[ARG0]], %[[ARG1]],
    %res1 = ascend_dpx.atomic_add %arg0, %arg1 : (!llvm.ptr<6>, vector<2xbf16>) -> vector<2xbf16>
    return %res1 : vector<2xbf16>
}

// CHECK-LABEL: @ascend_dpx_atomic_CAS_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
// CHECK-SAME: %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64
func.func @ascend_dpx_atomic_CAS_lowering(%arg0 : !llvm.ptr<6>, %arg1 : i64, %arg2 : i64) -> i64 {
    // CHECK: hivm.atom.CAS.S.s64
    // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]]
    %res1 = ascend_dpx.atomic_cas %arg0, %arg1, %arg2 : (!llvm.ptr<6>, i64, i64) -> i64
    return %res1 : i64
}

// CHECK-LABEL: @ascend_dpx_atomic_min_max_umin_max_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
// CHECK-SAME: %[[ARG1:.*]]: i32
// umin and umax have special lowering rules that must be differentiated from min and max lowering
func.func @ascend_dpx_atomic_min_max_umin_max_lowering(%arg0 : !llvm.ptr<6>, %arg1 : i32) -> (i32, i32, i32, i32) {
    // CHECK: hivm.atom.MIN.S.s32
    %A = ascend_dpx.atomic_min %arg0, %arg1 : (!llvm.ptr<6>, i32) -> i32
    // CHECK: hivm.atom.MAX.S.s32
    %B = ascend_dpx.atomic_max %arg0, %arg1 : (!llvm.ptr<6>, i32) -> i32
    // CHECK: hivm.atom.MIN.S.u32
    %C = ascend_dpx.atomic_umin %arg0, %arg1 : (!llvm.ptr<6>, i32) -> i32
    // CHECK: hivm.atom.MAX.S.u32
    %D = ascend_dpx.atomic_umax %arg0, %arg1 : (!llvm.ptr<6>, i32) -> i32
    return %A, %B, %C, %D : i32, i32, i32, i32
}
