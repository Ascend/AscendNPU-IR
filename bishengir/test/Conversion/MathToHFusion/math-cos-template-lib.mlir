// RUN: bishengir-opt -convert-math-to-hfusion %s -split-input-file | FileCheck %s

// f32 cos with rank 1..5 is lowered to the rank-specialized template-library
// SIMT entry point. The 2/pi table is staged from a single shared global.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_rank1(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = math.cos %arg0 : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func @cos_rank2(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = math.cos %arg0 : tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
  func.func @cos_rank3(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    %0 = math.cos %arg0 : tensor<2x4x8xf32>
    return %0 : tensor<2x4x8xf32>
  }
  func.func @cos_rank4(%arg0: tensor<2x2x4x8xf32>) -> tensor<2x2x4x8xf32> {
    %0 = math.cos %arg0 : tensor<2x2x4x8xf32>
    return %0 : tensor<2x2x4x8xf32>
  }
  func.func @cos_rank5(%arg0: tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32> {
    %0 = math.cos %arg0 : tensor<2x2x2x2x2xf32>
    return %0 : tensor<2x2x2x2x2xf32>
  }
}

// The table is the compiler-style 2/pi table shared with the Normalize path:
// exactly 320 i32 limbs, ending with the low limb 0x9e21c820.
// CHECK-COUNT-1: memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK-SAME: 20C8219E"> {alignment = 32 : i64}

// CHECK-LABEL: func.func @cos_rank1
// CHECK: memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [320], strides: [1]
// CHECK: memref.alloc() : memref<320xi32>
// CHECK: memref.copy {{.*}} : memref<320xi32, strided<[1]>, #hivm.address_space<gm>> to memref<320xi32>
// CHECK: bufferization.to_tensor {{.*}} : memref<320xi32>
// CHECK: hivm.hir.custom
// CHECK-SAME: arg_attrs = []
// CHECK-SAME: bitcode = ""
// CHECK-SAME: hivm.inline_mode = #hivm.inline_mode<always_inline>
// CHECK-SAME: hivm.pipe = #hivm.pipe<PIPE_V>
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK-SAME: hivm.vf_mode = #hivm.vf_mode<SIMT>
// CHECK-SAME: symbol = "cos_fp32_simt_1d"
// CHECK-SAME: ins({{.*}} : tensor<16xf32>, tensor<320xi32>) outs({{.*}} : tensor<16xf32>)
// CHECK: return

// CHECK-LABEL: func.func @cos_rank2
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK-SAME: ins({{.*}} : tensor<8x16xf32>, tensor<320xi32>) outs({{.*}} : tensor<8x16xf32>)

// CHECK-LABEL: func.func @cos_rank3
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_3d"
// CHECK-SAME: ins({{.*}} : tensor<2x4x8xf32>, tensor<320xi32>) outs({{.*}} : tensor<2x4x8xf32>)

// CHECK-LABEL: func.func @cos_rank4
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_4d"
// CHECK-SAME: ins({{.*}} : tensor<2x2x4x8xf32>, tensor<320xi32>) outs({{.*}} : tensor<2x2x4x8xf32>)

// CHECK-LABEL: func.func @cos_rank5
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_5d"
// CHECK-SAME: ins({{.*}} : tensor<2x2x2x2x2xf32>, tensor<320xi32>) outs({{.*}} : tensor<2x2x2x2x2xf32>)

// -----

// Ranks above SIMT_TRIG_MAX_RANK fall back to the HFusion Normalize expansion.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_rank6(%arg0: tensor<1x1x1x1x1x2xf32>) -> tensor<1x1x1x1x1x2xf32> {
    %0 = math.cos %arg0 : tensor<1x1x1x1x1x2xf32>
    return %0 : tensor<1x1x1x1x1x2xf32>
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @cos_rank6
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<1x1x1x1x1x2xf32>) outs({{.*}} : tensor<1x1x1x1x1x2xf32>)
// CHECK: return

// -----

// f16 cos falls back to the HFusion Normalize expansion.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_f16(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %0 = math.cos %arg0 : tensor<8x16xf16>
    return %0 : tensor<8x16xf16>
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @cos_f16
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<8x16xf16>) outs({{.*}} : tensor<8x16xf16>)
// CHECK: return

// -----

// bf16 cos falls back to the HFusion Normalize expansion.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_bf16(%arg0: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
    %0 = math.cos %arg0 : tensor<8x16xbf16>
    return %0 : tensor<8x16xbf16>
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @cos_bf16
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<8x16xbf16>) outs({{.*}} : tensor<8x16xbf16>)
// CHECK: return

// -----

// Any element type other than f32 falls back to the HFusion expansion too.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_f64(%arg0: tensor<8x16xf64>) -> tensor<8x16xf64> {
    %0 = math.cos %arg0 : tensor<8x16xf64>
    return %0 : tensor<8x16xf64>
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @cos_f64
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<8x16xf64>) outs({{.*}} : tensor<8x16xf64>)
// CHECK: return

// -----

// Scalar f32 cos is promoted to a rank-1 tensor first, then uses the SIMT
// entry point as well.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_scalar(%arg0: f32) -> f32 {
    %0 = math.cos %arg0 : f32
    return %0 : f32
  }
}

// CHECK-LABEL: func.func @cos_scalar
// CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_1d"
// CHECK: tensor.extract
// CHECK: return

// -----

// Scalar f16 cos is promoted to a rank-1 tensor but stays on the Normalize
// fallback because the element type is not f32.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_scalar_f16(%arg0: f16) -> f16 {
    %0 = math.cos %arg0 : f16
    return %0 : f16
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @cos_scalar_f16
// CHECK-NOT: hivm.hir.custom
// CHECK: tensor.from_elements {{.*}} : tensor<1xf16>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<1xf16>) outs({{.*}} : tensor<1xf16>)
// CHECK: tensor.extract
// CHECK: return

// -----

// Scalar f64 cos is promoted to a rank-1 tensor and falls back as well.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_scalar_f64(%arg0: f64) -> f64 {
    %0 = math.cos %arg0 : f64
    return %0 : f64
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @cos_scalar_f64
// CHECK-NOT: hivm.hir.custom
// CHECK: tensor.from_elements {{.*}} : tensor<1xf64>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins({{.*}} : tensor<1xf64>) outs({{.*}} : tensor<1xf64>)
// CHECK: tensor.extract
// CHECK: return

// -----

// Scalar bf16 cos is promoted through f32 and then uses the rank-1 SIMT
// entry point.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_scalar_bf16(%arg0: bf16) -> bf16 {
    %0 = math.cos %arg0 : bf16
    return %0 : bf16
  }
}

// CHECK-LABEL: func.func @cos_scalar_bf16
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: tensor.from_elements {{.*}} : tensor<1xf32>
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_1d"
// CHECK: tensor.extract
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: return
