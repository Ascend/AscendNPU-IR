// RUN: bishengir-opt -convert-math-to-hfusion %s -split-input-file | FileCheck %s

// A `math.sin`/`math.cos` pair that reads the same f32 tile is fused into one
// template-library `sincos_fp32_simt_<rank>d` call. Both results share the
// staged 2/pi table and one range reduction. f16-bracketed pairs, which stay in
// MLIR as one inline low-precision recipe, are covered by
// math-sincos-f16-bracket.mlir.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sincos_rank1(%arg0: tensor<16xf32>) -> (tensor<16xf32>, tensor<16xf32>) {
    %s = math.sin %arg0 : tensor<16xf32>
    %c = math.cos %arg0 : tensor<16xf32>
    return %s, %c : tensor<16xf32>, tensor<16xf32>
  }
  func.func @sincos_rank2(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
    %s = math.sin %arg0 : tensor<8x16xf32>
    %c = math.cos %arg0 : tensor<8x16xf32>
    return %s, %c : tensor<8x16xf32>, tensor<8x16xf32>
  }
  func.func @sincos_rank3(%arg0: tensor<2x4x8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x8xf32>) {
    %s = math.sin %arg0 : tensor<2x4x8xf32>
    %c = math.cos %arg0 : tensor<2x4x8xf32>
    return %s, %c : tensor<2x4x8xf32>, tensor<2x4x8xf32>
  }
  func.func @sincos_rank4(%arg0: tensor<2x2x4x8xf32>) -> (tensor<2x2x4x8xf32>, tensor<2x2x4x8xf32>) {
    %s = math.sin %arg0 : tensor<2x2x4x8xf32>
    %c = math.cos %arg0 : tensor<2x2x4x8xf32>
    return %s, %c : tensor<2x2x4x8xf32>, tensor<2x2x4x8xf32>
  }
  func.func @sincos_rank5(%arg0: tensor<2x2x2x2x2xf32>) -> (tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>) {
    %s = math.sin %arg0 : tensor<2x2x2x2x2xf32>
    %c = math.cos %arg0 : tensor<2x2x2x2x2xf32>
    return %s, %c : tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>
  }
}

// Dense integer attributes print as the raw bytes in memory order, so limb
// 0x45f306dc (index 161) appears as DC06F345 and the last limb 0x9e21c820 as
// 20C8219E. Pinning a mid-table limb as well as the tail is what catches an edit
// to the copy the compiler serializes into the global; the checksum in
// sincos-simt.c only covers the device-side copy, and the two are maintained by
// hand.
// CHECK-COUNT-1: memref.global "private" constant @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK-SAME: DC06F345
// CHECK-SAME: 20C8219E"> {alignment = 32 : i64}

// CHECK-LABEL: func.func @sincos_rank1
// CHECK: memref.get_global @tbl : memref<320xi32, #hivm.address_space<gm>>
// CHECK: memref.reinterpret_cast {{.*}} to offset: [0], sizes: [320], strides: [1]
// CHECK: memref.alloc() : memref<320xi32>
// CHECK: memref.copy {{.*}} : memref<320xi32, strided<[1]>, #hivm.address_space<gm>> to memref<320xi32>
// CHECK: bufferization.to_tensor {{.*}} : memref<320xi32>
// CHECK: %[[R:.*]]:2 = hivm.hir.custom
// CHECK-SAME: arg_attrs = []
// CHECK-SAME: bitcode = ""
// CHECK-SAME: hivm.inline_mode = #hivm.inline_mode<always_inline>
// CHECK-SAME: hivm.pipe = #hivm.pipe<PIPE_V>
// CHECK-SAME: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK-SAME: hivm.vf_mode = #hivm.vf_mode<SIMT>
// CHECK-SAME: symbol = "sincos_fp32_simt_1d"
// CHECK-SAME: ins({{.*}} : tensor<16xf32>, tensor<320xi32>) outs({{.*}}, {{.*}} : tensor<16xf32>, tensor<16xf32>)
// CHECK-SAME: -> (tensor<16xf32>, tensor<16xf32>)
// CHECK: return

// CHECK-LABEL: func.func @sincos_rank2
// CHECK: hivm.hir.custom {{.*}} symbol = "sincos_fp32_simt_2d"
// CHECK-SAME: ins({{.*}} : tensor<8x16xf32>, tensor<320xi32>) outs({{.*}}, {{.*}} : tensor<8x16xf32>, tensor<8x16xf32>)

// CHECK-LABEL: func.func @sincos_rank3
// CHECK: hivm.hir.custom {{.*}} symbol = "sincos_fp32_simt_3d"
// CHECK-SAME: ins({{.*}} : tensor<2x4x8xf32>, tensor<320xi32>) outs({{.*}}, {{.*}} : tensor<2x4x8xf32>, tensor<2x4x8xf32>)

// CHECK-LABEL: func.func @sincos_rank4
// CHECK: hivm.hir.custom {{.*}} symbol = "sincos_fp32_simt_4d"
// CHECK-SAME: ins({{.*}} : tensor<2x2x4x8xf32>, tensor<320xi32>) outs({{.*}}, {{.*}} : tensor<2x2x4x8xf32>, tensor<2x2x4x8xf32>)

// CHECK-LABEL: func.func @sincos_rank5
// CHECK: hivm.hir.custom {{.*}} symbol = "sincos_fp32_simt_5d"
// CHECK-SAME: ins({{.*}} : tensor<2x2x2x2x2xf32>, tensor<320xi32>) outs({{.*}}, {{.*}} : tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>)

// -----

// The pair is fused even when the cosine is defined first; the shared call is
// inserted before the earlier operation.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cos_before_sin(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
    %c = math.cos %arg0 : tensor<8x16xf32>
    %s = math.sin %arg0 : tensor<8x16xf32>
    return %s, %c : tensor<8x16xf32>, tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func @cos_before_sin
// CHECK: hivm.hir.custom {{.*}} symbol = "sincos_fp32_simt_2d"
// CHECK-NOT: symbol = "sin_fp32_simt_2d"
// CHECK-NOT: symbol = "cos_fp32_simt_2d"
// CHECK: return

// -----

// Different operands are not fused: each op keeps its own template-library
// call.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @different_inputs(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
    %s = math.sin %arg0 : tensor<8x16xf32>
    %c = math.cos %arg1 : tensor<8x16xf32>
    return %s, %c : tensor<8x16xf32>, tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func @different_inputs
// CHECK: hivm.hir.custom {{.*}} symbol = "sin_fp32_simt_2d"
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK-NOT: symbol = "sincos_fp32_simt_2d"
// CHECK: return

// -----

// A lone sin or cos still uses the single-operation path.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @only_sin(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %s = math.sin %arg0 : tensor<8x16xf32>
    return %s : tensor<8x16xf32>
  }
  func.func @only_cos(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c = math.cos %arg0 : tensor<8x16xf32>
    return %c : tensor<8x16xf32>
  }
}

// CHECK-LABEL: func.func @only_sin
// CHECK: hivm.hir.custom {{.*}} symbol = "sin_fp32_simt_2d"
// CHECK-NOT: symbol = "sincos_fp32_simt_2d"

// CHECK-LABEL: func.func @only_cos
// CHECK: hivm.hir.custom {{.*}} symbol = "cos_fp32_simt_2d"
// CHECK-NOT: symbol = "sincos_fp32_simt_2d"

// -----

// Pairs above the template-library rank limit and real f16 pairs fall back to
// the HFusion Normalize expansion (no fused call).
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sincos_rank6(%arg0: tensor<1x1x1x1x1x2xf32>) -> (tensor<1x1x1x1x1x2xf32>, tensor<1x1x1x1x1x2xf32>) {
    %s = math.sin %arg0 : tensor<1x1x1x1x1x2xf32>
    %c = math.cos %arg0 : tensor<1x1x1x1x1x2xf32>
    return %s, %c : tensor<1x1x1x1x1x2xf32>, tensor<1x1x1x1x1x2xf32>
  }
  func.func @sincos_f16(%arg0: tensor<8x16xf16>) -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %s = math.sin %arg0 : tensor<8x16xf16>
    %c = math.cos %arg0 : tensor<8x16xf16>
    return %s, %c : tensor<8x16xf16>, tensor<8x16xf16>
  }
}

// CHECK-NOT: memref.global @tbl
// CHECK-LABEL: func.func @sincos_rank6
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
// CHECK: return

// CHECK-LABEL: func.func @sincos_f16
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
// CHECK: return

// -----

// A rank-0 `tensor<f32>` has no template-library entry point: the per-rank
// symbols are emitted under `#if *_SIMT_MAX_RANK >= 1`, so dispatching rank 0
// would emit a call to an undefined `*_fp32_simt_0d`. Both the per-op path and
// the fusion path must leave rank 0 on the HFusion Normalize fallback.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sincos_rank0(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %s = math.sin %arg0 : tensor<f32>
    %c = math.cos %arg0 : tensor<f32>
    return %s, %c : tensor<f32>, tensor<f32>
  }
  func.func @only_sin_rank0(%arg0: tensor<f32>) -> tensor<f32> {
    %s = math.sin %arg0 : tensor<f32>
    return %s : tensor<f32>
  }
  func.func @only_cos_rank0(%arg0: tensor<f32>) -> tensor<f32> {
    %c = math.cos %arg0 : tensor<f32>
    return %c : tensor<f32>
  }
}

// CHECK-LABEL: func.func @sincos_rank0
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
// CHECK: return

// CHECK-LABEL: func.func @only_sin_rank0
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
// CHECK: return

// CHECK-LABEL: func.func @only_cos_rank0
// CHECK-NOT: hivm.hir.custom
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
// CHECK: return

// -----

// The 2/pi table staging is loop-invariant, so it is emitted once in front of
// the enclosing loop while the fused call stays inside the loop body. This is
// the layout that matters for MTE2 traffic: staging per iteration would copy the
// 1280-byte table for every tile.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pair_in_loop(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (tensor<8x16xf32>) {
      %s = math.sin %acc : tensor<8x16xf32>
      %c = math.cos %acc : tensor<8x16xf32>
      %sum = arith.addf %s, %c : tensor<8x16xf32>
      scf.yield %sum : tensor<8x16xf32>
    }
    return %r : tensor<8x16xf32>
  }
}

// One staging buffer, one copy, both before the loop; exactly one fused call,
// inside the loop body.
// CHECK-LABEL: func.func @pair_in_loop
// CHECK-COUNT-1: memref.alloc() : memref<320xi32>
// CHECK: memref.copy
// CHECK: scf.for
// CHECK-COUNT-1: hivm.hir.custom {{.*}} symbol = "sincos_fp32_simt_2d"
// CHECK: scf.yield
// CHECK: return
