// RUN: bishengir-opt --convert-ascend-dpx-to-hivmregbaseintrins --split-input-file %s | FileCheck %s
 
 
// CHECK-LABEL: @ascend_dpx_cast_lowering_0
func.func @ascend_dpx_cast_lowering_0(%arg0 : vector<2xbf16>) {
    // CHECK: llvm.call_intrinsic "llvm.hivm.bf16x2.to.f32x2"
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.f8e4m3x2"
    %0 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xbf16> to vector<2xf8E4M3FN>
    // CHECK: llvm.call_intrinsic "llvm.hivm.f8e4m3x2.to.f32x2"
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.bf16x2"
    %1 = ascend_dpx.cast %0 kind <fp_to_fp> : vector<2xf8E4M3FN> to vector<2xbf16>
    return
}
 
// CHECK-LABEL: @ascend_dpx_cast_lowering_1
func.func @ascend_dpx_cast_lowering_1(%arg0 : vector<2xf32>) {
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.f8e4m3x2"
    %0 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xf32> to vector<2xf8E4M3FN>
    // CHECK: llvm.call_intrinsic "llvm.hivm.f8e4m3x2.to.f32x2"
    %1 = ascend_dpx.cast %0 kind <fp_to_fp> : vector<2xf8E4M3FN> to vector<2xf32>
    return
}
 
// CHECK-LABEL: @ascend_dpx_cast_lowering_2
func.func @ascend_dpx_cast_lowering_2(%arg0 : vector<2xbf16>) {
    // CHECK: llvm.call_intrinsic "llvm.hivm.bf16x2.to.f32x2"
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.f8e5m2x2"
    %0 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xbf16> to vector<2xf8E5M2>
    // CHECK: llvm.call_intrinsic "llvm.hivm.f8e5m2x2.to.f32x2"
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.bf16x2"
    %1 = ascend_dpx.cast %0 kind <fp_to_fp> : vector<2xf8E5M2> to vector<2xbf16>
    return
}
 
// CHECK-LABEL: @ascend_dpx_cast_lowering_3
func.func @ascend_dpx_cast_lowering_3(%arg0 : vector<2xf32>) {
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.f8e5m2x2"
    %0 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xf32> to vector<2xf8E5M2>
    // CHECK: llvm.call_intrinsic "llvm.hivm.f8e5m2x2.to.f32x2"
    %1 = ascend_dpx.cast %0 kind <fp_to_fp> : vector<2xf8E5M2> to vector<2xf32>
    return
}

// CHECK-LABEL: @ascend_dpx_cast_lowering_4
func.func @ascend_dpx_cast_lowering_4(%arg0 : vector<2xf32>) {
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.bf16x2"
    %0 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xf32> to vector<2xbf16>
    // CHECK: llvm.call_intrinsic "llvm.hivm.f32x2.to.f16x2"
    %1 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xf32> to vector<2xf16>
    return
}

// CHECK-LABEL: @ascend_dpx_cast_lowering_5
func.func @ascend_dpx_cast_lowering_5(%arg0 : vector<2xbf16>, %arg1 : vector<2xf16>) {
    // CHECK: llvm.call_intrinsic "llvm.hivm.bf16x2.to.f32x2"
    %0 = ascend_dpx.cast %arg0 kind <fp_to_fp> : vector<2xbf16> to vector<2xf32>
    // CHECK: llvm.call_intrinsic "llvm.hivm.f16x2.to.f32x2"
    %1 = ascend_dpx.cast %arg1 kind <fp_to_fp> : vector<2xf16> to vector<2xf32>
    return
}