// RUN: bishengir-opt %s --convert-ascend-dpx-to-hivmregbaseintrins -canonicalize --split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck --enable-var-scope %s

// CHECK-LABEL: @ascend_dpx_wide_predicate_load_lowering_i32
func.func @ascend_dpx_wide_predicate_load_lowering_i32(%arg0 : !llvm.ptr<1>, %arg1 : i1) -> vector<4xi32> {
    // CHECK: dense<0> : vector<4xi32>
    %res = ascend_dpx.load %arg0, %arg1 : (!llvm.ptr<1>, i1) -> vector<4xi32>
    return %res : vector<4xi32>
}

// CHECK-LABEL: @ascend_dpx_wide_predicate_load_lowering_f32
func.func @ascend_dpx_wide_predicate_load_lowering_f32(%arg0 : !llvm.ptr<1>, %arg1 : i1) -> vector<4xf32> {
    // CHECK: dense<0.000000e+00> : vector<4xf32>
    %res = ascend_dpx.load %arg0, %arg1 : (!llvm.ptr<1>, i1) -> vector<4xf32>
    return %res : vector<4xf32>
}

// CHECK-LABEL: @ascend_dpx_wide_predicate_load_lowering_i1
func.func @ascend_dpx_wide_predicate_load_lowering_i1(%arg0 : !llvm.ptr<1>, %arg1 : i1) -> vector<4xi1> {
    // CHECK: dense<false> : vector<4xi1>
    %res = ascend_dpx.load %arg0, %arg1 : (!llvm.ptr<1>, i1) -> vector<4xi1>
    return %res : vector<4xi1>
}

// CHECK-LABEL: @ascend_dpx_wide_predicate_load_lowering_i64
func.func @ascend_dpx_wide_predicate_load_lowering_i64(%arg0 : !llvm.ptr<1>, %arg1 : i1) -> vector<4xi64> {
    // CHECK: dense<0> : vector<4xi64>
    %res = ascend_dpx.load %arg0, %arg1 : (!llvm.ptr<1>, i1) -> vector<4xi64>
    return %res : vector<4xi64>
}

// CHECK-LABEL: @ascend_dpx_wide_predicate_load_lowering_bf16
func.func @ascend_dpx_wide_predicate_load_lowering_bf16(%arg0 : !llvm.ptr<1>, %arg1 : i1) -> vector<4xbf16> {
    // CHECK: dense<0.000000e+00> : vector<4xbf16>
    %res = ascend_dpx.load %arg0, %arg1 : (!llvm.ptr<1>, i1) -> vector<4xbf16>
    return %res : vector<4xbf16>
}

// CHECK-LABEL: @ascend_dpx_wide_load_lowering
func.func @ascend_dpx_wide_load_lowering(%arg0 : !llvm.ptr<1>) -> vector<4xi32> {
    // CHECK: llvm.load
    %0 = ascend_dpx.load %arg0 cacheModifier = < L2_CACHE_HINT_NORMAL_FV > : (!llvm.ptr<1>) -> vector<4xi32>
    return %0 : vector<4xi32>
}

// CHECK-LABEL: @ascend_dpx_wide_store_lowering
func.func @ascend_dpx_wide_store_lowering(%arg0 : !llvm.ptr<1>, %vectorArg : vector<4xi32>, %mask : i1) {
    // CHECK: scf.if
    // CHECK: llvm.extractelement
    // CHECK: llvm.insertelement
    // CHECK: llvm.extractelement
    // CHECK: llvm.insertelement
    // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.getelementptr %{{.*}}[2]
    // CHECK: llvm.extractelement
    // CHECK: llvm.insertelement
    // CHECK: llvm.extractelement
    // CHECK: llvm.insertelement
    // CHECK: llvm.store %{{.*}}, %{{.*}} : vector<2xi32>, !llvm.ptr<1>
    ascend_dpx.store %arg0, %vectorArg, %mask cacheModifier = < L2_CACHE_HINT_NORMAL_FV > : !llvm.ptr<1>, vector<4xi32>
    return
}
