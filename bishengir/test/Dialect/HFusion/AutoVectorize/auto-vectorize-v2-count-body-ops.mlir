// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="max-fused-ops=15" -split-input-file | FileCheck %s
//
// Verify that countFusedBodyOps correctly accounts for tensor inputs,
// inits, and body ops (not just the number of linalg ops). When the
// total vector-slot estimate exceeds maxFusedOps the fusion splits,
// preventing oversized VFs that would spill vector registers and
// overflow the 6144-byte VF stack.
//
// NOTE: --hfusion-pre-vectorization-fusion is intentionally omitted.
// It would fuse the two independent generics into one before
// AutoVectorizeV2, defeating the purpose of this test.

// -----
// Case 1: two linalg.generic ops with 8 body ops each.
// countFusedBodyOps per leaf = 2(inputs) + 1(init) + 8(body) = 11.
// Two leaves = 22 > 15 → split into 2 compute groups.

// CHECK-LABEL: func.func @body_ops_exceed_limit
// CHECK: {"outlined-loop-target-{{[0-9]+}}"}
// CHECK: {"outlined-loop-target-{{[0-9]+}}"}
// CHECK-NOT: {"outlined-loop-target-{{[0-9]+}}"}

module {
  func.func @body_ops_exceed_limit(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, parallel_mode = "mix_simd_simt"} {
    %cst = arith.constant 1.000000e+00 : f32
    %o0 = tensor.empty() : tensor<64xf32>
    %o1 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%o0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %10 = arith.mulf %in, %in : f32
      %11 = arith.mulf %10, %in2 : f32
      %12 = arith.mulf %11, %cst : f32
      %13 = arith.addf %12, %in : f32
      %14 = arith.mulf %13, %cst : f32
      %15 = arith.addf %14, %in2 : f32
      %16 = arith.mulf %15, %cst : f32
      %17 = arith.addf %16, %in : f32
      linalg.yield %17 : f32
    } -> tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%o1 : tensor<64xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %10 = arith.mulf %in, %in2 : f32
      %11 = arith.mulf %10, %cst : f32
      %12 = arith.addf %11, %in : f32
      %13 = arith.mulf %12, %in2 : f32
      %14 = arith.subf %13, %in : f32
      %15 = arith.addf %14, %in2 : f32
      %16 = arith.mulf %15, %cst : f32
      %17 = arith.subf %16, %in : f32
      linalg.yield %17 : f32
    } -> tensor<64xf32>
    return %1, %2 : tensor<64xf32>, tensor<64xf32>
  }
}

// -----
// Case 2: two small linalg.generic ops, 2 body ops each.
// countFusedBodyOps per leaf = 2(inputs) + 1(init) + 2(body) = 5.
// Two leaves = 10 < 15 → no split → 1 compute group.

// CHECK-LABEL: func.func @body_ops_under_limit
// CHECK: {"outlined-loop-target-{{[0-9]+}}"}
// CHECK-NOT: {"outlined-loop-target-{{[0-9]+}}"}

module {
  func.func @body_ops_under_limit(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>, parallel_mode = "mix_simd_simt"} {
    %cst = arith.constant 1.000000e+00 : f32
    %o0 = tensor.empty() : tensor<64xf32>
    %o1 = tensor.empty() : tensor<64xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%o0 : tensor<64xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %10 = arith.mulf %in, %in2 : f32
      %11 = arith.addf %10, %in : f32
      linalg.yield %11 : f32
    } -> tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%o1 : tensor<64xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %10 = arith.subf %in, %in2 : f32
      %11 = arith.mulf %10, %in : f32
      linalg.yield %11 : f32
    } -> tensor<64xf32>
    return %1, %2 : tensor<64xf32>, tensor<64xf32>
  }
}
