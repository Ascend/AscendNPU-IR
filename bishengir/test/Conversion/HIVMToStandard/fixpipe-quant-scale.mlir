// RUN: bishengir-opt -convert-hivm-to-std %s -split-input-file | FileCheck %s
 
// Test that FixpipeOp's quant_scale operand is unified to f32 type in the
// library function declaration regardless of the original quant_scale type.
// - f16 quant_scale: should insert arith.extf
// - f32 quant_scale: should pass through without conversion
// The library function is declared exactly once with f32 for that parameter.
 
// The library function declaration with f32 quant_scale parameter.
// CHECK:      func.func private @fixpipe_nz2nd_float_to_half_2d_to_2d_ubuf({{.*}}, {{.*}}, {{.*}}, f32, {{.*}}, {{.*}}, {{.*}})
// CHECK-NOT:  func.func private @fixpipe_nz2nd_float_to_half_2d_to_2d_ubuf
 
// CHECK-LABEL: func.func @fixpipe_quant_scale_f32_unify
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
    func.func @fixpipe_quant_scale_f32_unify(
        %src0: memref<2x32xf32>, %dst0: memref<2x32xf16, #hivm.address_space<ub>>,
        %src1: memref<2x32xf32>, %dst1: memref<2x32xf16, #hivm.address_space<ub>>,
        %qs_f16: f16, %qs_f32: f32) {
    
    // f16 quant_scale → extended to f32.
    // CHECK:      %[[F32_QS0:.*]] = arith.extf {{.*}} : f16 to f32
    // CHECK:      call @fixpipe_nz2nd_float_to_half_2d_to_2d_ubuf({{.*}}, {{.*}}, {{.*}}, %[[F32_QS0]], {{.*}}, {{.*}}, {{.*}})
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
        ins(%src0 : memref<2x32xf32>) outs(%dst0 : memref<2x32xf16, #hivm.address_space<ub>>)
        quant_scale = %qs_f16 : f16
    
    // f32 quant_scale → passed through directly, no conversion op.
    // CHECK-NOT:  arith.truncf
    // CHECK-NOT:  arith.extf
    // CHECK:      call @fixpipe_nz2nd_float_to_half_2d_to_2d_ubuf({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
        ins(%src1 : memref<2x32xf32>) outs(%dst1 : memref<2x32xf16, #hivm.address_space<ub>>)
        quant_scale = %qs_f32 : f32
    
    return
    }
}