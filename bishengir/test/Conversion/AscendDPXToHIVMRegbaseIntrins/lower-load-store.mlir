// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck --enable-var-scope %s

// CHECK-LABEL: @ascend_dpx_load_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
func.func @ascend_dpx_load_lowering(%arg0 : !llvm.ptr<6>) -> i32 {
    // CHECK-NEXT: %[[RES:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<6> -> i32
    %res = ascend_dpx.load %arg0 : (!llvm.ptr<6>) -> i32
    return %res : i32
}

// -----

// CHECK-LABEL: @ascend_dpx_predicate_load_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
// CHECK-SAME: %[[ARG1:.*]]: i1
// CHECK-SAME: %[[ARG2:.*]]: i32
func.func @ascend_dpx_predicate_load_lowering(%arg0 : !llvm.ptr<6>, %arg1 : i1, %arg2 : i32) -> i32 {
    // CHECK-NEXT: %[[RES:.*]] = scf.if %[[ARG1]] -> (i32) 
    // CHECK-NEXT: %[[RES1:.*]] = llvm.load %arg0 : !llvm.ptr<6> -> i32
    // CHECK-NEXT: scf.yield %[[RES1]] : i32
    // CHECK-NEXT: else
    // CHECK-NEXT: scf.yield %[[ARG2]] : i32
    %res = ascend_dpx.load %arg0, %arg1, %arg2 : (!llvm.ptr<6>, i1, i32) -> i32
    return %res : i32
}

// -----

// CHECK-LABEL: @ascend_dpx_cache_hint_load_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
func.func @ascend_dpx_cache_hint_load_lowering(%arg0 : !llvm.ptr<1>) -> i32 {
    // CHECK: %[[CONST0:.*]] =
    // CHECK-SAME: 8
    // CHECK: %[[RES:.*]] = 
    // CHECK-SAME: "llvm.hivm.ldg.cache.s32"
    // CHECK-SAME: %[[ARG0]]
    // CHECK-SAME: %[[CONST0]]
    %res = ascend_dpx.load %arg0 cacheModifier = < L2_CACHE_HINT_IDS_FV > : (!llvm.ptr<1>) -> i32
    return %res : i32
}

// CHECK-LABEL: @ascend_dpx_cache_hint_option_load_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
func.func @ascend_dpx_cache_hint_option_load_lowering(%arg0 : !llvm.ptr<1>) -> i32 {
    // CHECK: %[[CONST0:.*]] =
    // CHECK-SAME: 8
    // CHECK: %[[RES:.*]] = 
    // CHECK-SAME: "llvm.hivm.ldg.uncache.s32"
    // CHECK-SAME: %[[ARG0]]
    // CHECK-SAME: %[[CONST0]]
    %res = ascend_dpx.load %arg0 cacheModifier = < L2_CACHE_HINT_IDS_FV > cacheOption = < LOADCACHEOPTION_NCA > : (!llvm.ptr<1>) -> i32
    return %res : i32
}

// -----

// CHECK-LABEL: @ascend_dpx_store_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
// CHECK-SAME: %[[ARG1:.*]]: i32
func.func @ascend_dpx_store_lowering(%arg0 : !llvm.ptr<6>, %arg1 : i32) {
    // CHECK-NEXT: llvm.store %[[ARG1]], %[[ARG0]] : i32, !llvm.ptr<6>
    ascend_dpx.store %arg0, %arg1 : !llvm.ptr<6>, i32
    return
}

// -----

// CHECK-LABEL: @ascend_dpx_predicate_store_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<6>
// CHECK-SAME: %[[ARG1:.*]]: i32
// CHECK-SAME: %[[ARG2:.*]]: i1
func.func @ascend_dpx_predicate_store_lowering(%arg0 : !llvm.ptr<6>, %arg1 : i32, %arg2 : i1) {
    // CHECK-NEXT: scf.if %[[ARG2]]
    // CHECK-NEXT: llvm.store %[[ARG1]], %[[ARG0]] : i32, !llvm.ptr<6>
    ascend_dpx.store %arg0, %arg1, %arg2 : !llvm.ptr<6>, i32
    return
}

// -----

// CHECK-LABEL: @ascend_dpx_cache_hint_store_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: i32
func.func @ascend_dpx_cache_hint_store_lowering(%arg0 : !llvm.ptr<1>, %arg1 : i32) {
    // CHECK: %[[CONST0:.*]] =
    // CHECK-SAME: 15
    // CHECK: %[[ARG1CAST:.*]] = llvm.bitcast %[[ARG1]] : i32 to i32
    // CHECK: "llvm.hivm.stg.cache.b32"
    // CHECK-SAME: %[[ARG0]]
    // CHECK-SAME: %[[ARG1CAST]]
    // CHECK-SAME: %[[CONST0]]
    ascend_dpx.store %arg0, %arg1, cacheModifier = < L2_CACHE_HINT_WTS_RED > : !llvm.ptr<1>, i32
    return
}

// CHECK-LABEL: @ascend_dpx_cache_hint_option_store_lowering
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<1>
// CHECK-SAME: %[[ARG1:.*]]: i32
func.func @ascend_dpx_cache_hint_option_store_lowering(%arg0 : !llvm.ptr<1>, %arg1 : i32) {
    // CHECK: %[[CONST0:.*]] =
    // CHECK-SAME: 15
    // CHECK: %[[ARG1CAST:.*]] = llvm.bitcast %[[ARG1]] : i32 to i32
    // CHECK: "llvm.hivm.stg.uncache.b32"
    // CHECK-SAME: %[[ARG0]]
    // CHECK-SAME: %[[ARG1CAST]]
    // CHECK-SAME: %[[CONST0]]
    ascend_dpx.store %arg0, %arg1, cacheModifier = < L2_CACHE_HINT_WTS_RED > cacheOption = < LOADCACHEOPTION_NCA > : !llvm.ptr<1>, i32
    return
}
