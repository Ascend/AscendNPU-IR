// RUN: bishengir-opt %s --hfusion-pre-vectorization-fusion --hfusion-auto-vectorize-v2 -split-input-file | FileCheck %s
//
// Verify that AutoVectorizeV2 inlines VFFusionPass-generated fused_N
// sub-functions for pure SIMD functions (parallel_mode="simd"). After
// inlining, all compute ops sit in the entry function and fuse into a
// single VF, eliminating intermediate UB buffers that would overflow
// when multiplied by tt.num_stages.

// CHECK-LABEL: func.func @simd_inline_fused_calls
// Private fused sub-function is inlined and erased.
// CHECK-NOT: func.func private @simd_inline_fused_calls_fused_0
// With kSIMDMaxFusedOps=100 both the original and inlined linalg ops
// fuse into one compute loop.
// CHECK: {"outlined-loop-target-{{[0-9]+}}"}
// CHECK-NOT: {"outlined-loop-target-{{[0-9]+}}"}

module {
  func.func private @simd_inline_fused_calls_fused_0(%arg0: tensor<64xf32>) -> tensor<64xf32>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]} ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.mulf %in, %in : f32
      %3 = arith.mulf %2, %in : f32
      %4 = arith.addf %3, %in : f32
      %5 = arith.mulf %4, %in : f32
      %6 = arith.addf %5, %in : f32
      linalg.yield %6 : f32
    } -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }

  func.func @simd_inline_fused_calls(%arg0: tensor<64xf32>) -> tensor<64xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = func.call @simd_inline_fused_calls_fused_0(%arg0) : (tensor<64xf32>) -> tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]} ins(%1 : tensor<64xf32>) outs(%0 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.mulf %in, %in : f32
      %4 = arith.addf %3, %in : f32
      linalg.yield %4 : f32
    } -> tensor<64xf32>
    return %2 : tensor<64xf32>
  }
}
