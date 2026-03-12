// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -outline-vector-function -split-input-file 2>&1 | FileCheck %s
//
//==============================================================================
// TEST SPECIFICATION: AutoVectorizeV2 Generalization Tests
//==============================================================================
//
// PURPOSE: Generalization tests for the hfusion-auto-vectorize-v2 pass, which
// fuses vector-related linalg/hfusion ops, tiles by register size (32 bytes),
// and outlines loops into VectorFunction (VF) callees.
//
// FUSABLE OPS (per isFusableOp in AutoVectorizeV2.cpp):
//   - linalg::LinalgOp: fill, elemwise_binary, elemwise_unary, transpose,
//     reduce, broadcast, generic
//   - hfusion::InterleaveOp, hfusion::DeinterleaveOp
//
// NON-VECTORIZABLE OPS (barriers per isNonVectorizableOp):
//   - hfusion::LoadOp, StoreOp, CastOp, GatherOp, CumsumOp, etc.
//   - tensor::ExtractSliceOp, InsertSliceOp, CollapseShapeOp, ExpandShapeOp,
//     ReshapeOp, ConcatOp, CastOp
//   - scf::ForOp, scf::IfOp, func::CallOp
//
// COVERAGE:
//   - Common shapes: 1D (64), 2D (8x32, 256x64), 3D, 4D
//   - Uncommon shapes: odd (33), small (2x7), single-row (1x16)
//   - Special shapes: prime-like (17, 13x23), tall (128x1),
//     odd-square (17x17, 3x3), odd-inner (8x31, 4x8x15), single-batch (1x64x1)
//   - Data types: f32, f16, bf16, i8, i16, i32
//   - Op combinations: elemwise chains, reduce+broadcast, transpose+elemwise,
//     interleave+deinterleave, fill+reduce; combo tests on special shapes:
//     reduce+broadcast+elemwise (7x13), transpose+reduce (3x7x11), unary chain
//     (31), elemwise+transpose (3x3), reduce odd-inner (4x8x15), softmax-like
//     (1x31), elemwise chain on prime 2D (13x17 f16)
//   - Barrier tests: tensor.extract_slice, hfusion.cast, tensor.collapse_shape
//
// EXPECTED: Each test passes and produces _outlined_vf_* functions.
//==============================================================================

// TEST_SPEC: common_1d_f32 | Shape: 64xf32 | Ops: elemwise_binary(add) | Desc: 1D vector add, 32-byte aligned
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @common_1d_f32(
// CHECK: _outlined_vf_
module {
  func.func @common_1d_f32(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
}

// -----
// TEST_SPEC: common_2d_mx32 | Shape: 8x32xf32 | Ops: elemwise_binary(mul) | Desc: Typical Mx32 block
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @common_2d_mx32(
// CHECK: _outlined_vf_
module {
  func.func @common_2d_mx32(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32xf32>) -> tensor<8x32xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<8x32xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%0 : tensor<8x32xf32>) -> tensor<8x32xf32>
    return %1 : tensor<8x32xf32>
  }
}

// -----
// TEST_SPEC: common_2d_256x64 | Shape: 256x64xf16 | Ops: elemwise_binary(add) + elemwise_unary(exp) | Desc: 2D activation pattern
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @common_2d_256x64(
// CHECK: _outlined_vf_
module {
  func.func @common_2d_256x64(%arg0: tensor<256x64xf16>, %arg1: tensor<256x64xf16>) -> tensor<256x64xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<256x64xf16>
    %1 = tensor.empty() : tensor<256x64xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<256x64xf16>, tensor<256x64xf16>) outs(%0 : tensor<256x64xf16>) -> tensor<256x64xf16>
    %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%2 : tensor<256x64xf16>) outs(%1 : tensor<256x64xf16>) -> tensor<256x64xf16>
    return %3 : tensor<256x64xf16>
  }
}

// -----
// TEST_SPEC: common_reduce_last | Shape: 16x64xf32 | Ops: elemwise_binary(mul) + reduce(dims=[1]) | Desc: Last-axis reduction
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @common_reduce_last(
// CHECK: _outlined_vf_
module {
  func.func @common_reduce_last(%arg0: tensor<16x64xf32>, %arg1: tensor<16x64xf32>) -> tensor<16xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<16x64xf32>
    %1 = tensor.empty() : tensor<16xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%0 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %reduced = linalg.reduce ins(%2 : tensor<16x64xf32>) outs(%1 : tensor<16xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %3 = arith.addf %in, %init : f32
        linalg.yield %3 : f32
      }
    return %reduced : tensor<16xf32>
  }
}

// -----
// TEST_SPEC: ops_elemwise_chain | Shape: 64xf32 | Ops: add -> mul -> sub | Desc: Long elemwise chain in 1D
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @ops_elemwise_chain(
// CHECK: _outlined_vf_
module {
  func.func @ops_elemwise_chain(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg2 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%2, %arg0 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    return %3 : tensor<64xf32>
  }
}

// -----
// TEST_SPEC: ops_transpose_elemwise | Shape: 8x16x32xf32 | Ops: elemwise_binary(add) -> transpose | Desc: Transpose as consumer of elementwise op
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @ops_transpose_elemwise(
// CHECK: _outlined_vf_
module {
  func.func @ops_transpose_elemwise(%arg0: tensor<8x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> (tensor<8x16x32xf32>, tensor<8x32x16xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<8x16x32xf32>
    %1 = tensor.empty() : tensor<8x32x16xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<8x16x32xf32>, tensor<8x16x32xf32>) outs(%0 : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %transposed = linalg.transpose ins(%2 : tensor<8x16x32xf32>) outs(%1 : tensor<8x32x16xf32>) permutation = [0, 2, 1]
    return %2, %transposed : tensor<8x16x32xf32>, tensor<8x32x16xf32>
  }
}

// -----
// TEST_SPEC: uncommon_odd_1d | Shape: 33xf32 | Ops: elemwise_binary(add) | Desc: Odd-length 1D vector with tail
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @uncommon_odd_1d(
// CHECK: _outlined_vf_
module {
  func.func @uncommon_odd_1d(%arg0: tensor<33xf32>, %arg1: tensor<33xf32>) -> tensor<33xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<33xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>) -> tensor<33xf32>
    return %1 : tensor<33xf32>
  }
}

// -----
// TEST_SPEC: uncommon_small_2d | Shape: 2x7xf32 | Ops: elemwise_binary(mul) | Desc: Small non-power-of-two 2D shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @uncommon_small_2d(
// CHECK: _outlined_vf_
module {
  func.func @uncommon_small_2d(%arg0: tensor<2x7xf32>, %arg1: tensor<2x7xf32>) -> tensor<2x7xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<2x7xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<2x7xf32>, tensor<2x7xf32>) outs(%0 : tensor<2x7xf32>) -> tensor<2x7xf32>
    return %1 : tensor<2x7xf32>
  }
}

// -----
// TEST_SPEC: uncommon_1x16 | Shape: 1x16xf16 | Ops: elemwise_binary(add) | Desc: Single-row 2D shape exactly 32 bytes
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @uncommon_1x16(
// CHECK: _outlined_vf_
module {
  func.func @uncommon_1x16(%arg0: tensor<1x16xf16>, %arg1: tensor<1x16xf16>) -> tensor<1x16xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<1x16xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1x16xf16>, tensor<1x16xf16>) outs(%0 : tensor<1x16xf16>) -> tensor<1x16xf16>
    return %1 : tensor<1x16xf16>
  }
}

// -----
// TEST_SPEC: ops_interleave_deinterleave | Shape: 4x16xf16 | Ops: interleave -> elemwise_binary(mul) -> deinterleave | Desc: HFusion interleave/deinterleave pattern
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @ops_interleave_deinterleave(
// CHECK: _outlined_vf_
module {
  func.func @ops_interleave_deinterleave(%arg0: tensor<4x16xf16>, %arg1: tensor<4x16xf16>) -> tensor<4x16xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<4x32xf16>
    %1 = tensor.empty() : tensor<4x16xf16>
    %2 = hfusion.interleave %arg0, %arg1 : tensor<4x16xf16>, tensor<4x16xf16> -> tensor<4x32xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %2 : tensor<4x32xf16>, tensor<4x32xf16>) outs(%0 : tensor<4x32xf16>) -> tensor<4x32xf16>
    %4 = hfusion.deinterleave %3 channel<0> : tensor<4x32xf16> -> tensor<4x16xf16>
    return %4 : tensor<4x16xf16>
  }
}

// -----
// TEST_SPEC: broadcast_reduce_elemwise | Shape: 8x32xf32 | Ops: reduce -> broadcast -> elemwise_binary(add) | Desc: Last-axis reduce then broadcast and add
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @broadcast_reduce_elemwise(
// CHECK: _outlined_vf_
module {
  func.func @broadcast_reduce_elemwise(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<8x32xf32>) outs(%filled : tensor<8xf32>) dimensions = [1]
      (%in: f32, %initval: f32) {
        %sum = arith.addf %in, %initval : f32
        linalg.yield %sum : f32
      }
    %bcast_dst = tensor.empty() : tensor<8x32xf32>
    %broadcast = linalg.broadcast ins(%reduced : tensor<8xf32>) outs(%bcast_dst : tensor<8x32xf32>) dimensions = [1]
    %out_init = tensor.empty() : tensor<8x32xf32>
    %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %broadcast : tensor<8x32xf32>, tensor<8x32xf32>) outs(%out_init : tensor<8x32xf32>) -> tensor<8x32xf32>
    return %out : tensor<8x32xf32>
  }
}

// -----
// TEST_SPEC: broadcast_last_axis | Shape: 16x64xf32 | Ops: reduce -> broadcast -> elemwise_binary(mul) | Desc: Reduce to 1D then broadcast back to 2D
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @broadcast_last_axis(
// CHECK: _outlined_vf_
module {
  func.func @broadcast_last_axis(%arg0: tensor<16x64xf32>) -> tensor<16x64xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<16xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<16xf32>) -> tensor<16xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<16x64xf32>) outs(%filled : tensor<16xf32>) dimensions = [1]
      (%in: f32, %initval: f32) {
        %sum = arith.addf %in, %initval : f32
        linalg.yield %sum : f32
      }
    %bcast_dst = tensor.empty() : tensor<16x64xf32>
    %broadcast = linalg.broadcast ins(%reduced : tensor<16xf32>) outs(%bcast_dst : tensor<16x64xf32>) dimensions = [1]
    %out_init = tensor.empty() : tensor<16x64xf32>
    %out = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %broadcast : tensor<16x64xf32>, tensor<16x64xf32>) outs(%out_init : tensor<16x64xf32>) -> tensor<16x64xf32>
    return %out : tensor<16x64xf32>
  }
}

// -----
// TEST_SPEC: reduce_broadcast_reduce_simple | Shape: 4x4x8x16xf32 | Ops: elemwise -> reduce -> broadcast -> reduce | Desc: 4D reduce-broadcast-reduce pipeline
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @reduce_broadcast_reduce_simple(
// CHECK: _outlined_vf_
module {
  func.func @reduce_broadcast_reduce_simple(%arg0: tensor<4x4x8x16xf32>, %arg1: tensor<4x4x8x16xf32>) -> tensor<4x4xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0.0 : f32
    %dst = tensor.empty() : tensor<4x4x8x16xf32>
    %red_init = tensor.empty() : tensor<4x4xf32>
    %red_filled = linalg.fill ins(%c0 : f32) outs(%red_init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<4x4x8x16xf32>, tensor<4x4x8x16xf32>) outs(%dst : tensor<4x4x8x16xf32>) -> tensor<4x4x8x16xf32>
    %reduced0 = linalg.reduce ins(%sum : tensor<4x4x8x16xf32>) outs(%red_filled : tensor<4x4xf32>) dimensions = [2, 3]
      (%in: f32, %initval: f32) {
        %add = arith.addf %in, %initval : f32
        linalg.yield %add : f32
      }
    %bcast_dst = tensor.empty() : tensor<4x4x8x16xf32>
    %broadcast = linalg.broadcast ins(%reduced0 : tensor<4x4xf32>) outs(%bcast_dst : tensor<4x4x8x16xf32>) dimensions = [2, 3]
    %red_init2 = tensor.empty() : tensor<4x4xf32>
    %red_filled2 = linalg.fill ins(%c0 : f32) outs(%red_init2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %reduced1 = linalg.reduce ins(%broadcast : tensor<4x4x8x16xf32>) outs(%red_filled2 : tensor<4x4xf32>) dimensions = [2, 3]
      (%in: f32, %initval: f32) {
        %add2 = arith.addf %in, %initval : f32
        linalg.yield %add2 : f32
      }
    return %reduced1 : tensor<4x4xf32>
  }
}

// -----
// TEST_SPEC: transpose_4d_perm | Shape: 2x4x8x32xf32 | Ops: elemwise_binary(add) -> transpose | Desc: 4D transpose with permutation [0,2,1,3]
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @transpose_4d_perm(
// CHECK: _outlined_vf_
module {
  func.func @transpose_4d_perm(%arg0: tensor<2x4x8x32xf32>, %arg1: tensor<2x4x8x32xf32>) -> tensor<2x8x4x32xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<2x4x8x32xf32>
    %out = tensor.empty() : tensor<2x8x4x32xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) outs(%dst : tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
    %t = linalg.transpose ins(%sum : tensor<2x4x8x32xf32>) outs(%out : tensor<2x8x4x32xf32>) permutation = [0, 2, 1, 3]
    return %t : tensor<2x8x4x32xf32>
  }
}

// -----
// TEST_SPEC: transpose_3d_then_elemwise | Shape: 3x5x11xf32 | Ops: transpose -> elemwise_binary(add) | Desc: Transpose feeding into elementwise
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @transpose_3d_then_elemwise(
// CHECK: _outlined_vf_
module {
  func.func @transpose_3d_then_elemwise(%arg0: tensor<3x5x11xf32>, %arg1: tensor<3x5x11xf32>) -> tensor<3x11x5xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<3x11x5xf32>
    %out = tensor.empty() : tensor<3x11x5xf32>
    %t0 = linalg.transpose ins(%arg0 : tensor<3x5x11xf32>) outs(%tmp : tensor<3x11x5xf32>) permutation = [0, 2, 1]
    %t1 = linalg.transpose ins(%arg1 : tensor<3x5x11xf32>) outs(%out : tensor<3x11x5xf32>) permutation = [0, 2, 1]
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%t0, %t1 : tensor<3x11x5xf32>, tensor<3x11x5xf32>) outs(%out : tensor<3x11x5xf32>) -> tensor<3x11x5xf32>
    return %sum : tensor<3x11x5xf32>
  }
}

// -----
// TEST_SPEC: transpose_vsstb_shape | Shape: 4x8x8xf32 | Ops: transpose | Desc: 3D transpose with last axis 8*f32=32 bytes (vsstb pattern)
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @transpose_vsstb_shape(
// CHECK: _outlined_vf_
module {
  func.func @transpose_vsstb_shape(%arg0: tensor<4x8x8xf32>) -> tensor<8x4x8xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %out = tensor.empty() : tensor<8x4x8xf32>
    %t = linalg.transpose ins(%arg0 : tensor<4x8x8xf32>) outs(%out : tensor<8x4x8xf32>) permutation = [1, 0, 2]
    return %t : tensor<8x4x8xf32>
  }
}

// -----
// TEST_SPEC: dtype_bf16 | Shape: 16x64xbf16 | Ops: elemwise_binary(add) | Desc: BF16 elementwise add
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @dtype_bf16(
// CHECK: _outlined_vf_
module {
  func.func @dtype_bf16(%arg0: tensor<16x64xbf16>, %arg1: tensor<16x64xbf16>) -> tensor<16x64xbf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<16x64xbf16>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<16x64xbf16>, tensor<16x64xbf16>) outs(%dst : tensor<16x64xbf16>) -> tensor<16x64xbf16>
    return %res : tensor<16x64xbf16>
  }
}

// -----
// TEST_SPEC: dtype_i8 | Shape: 8x32xi8 | Ops: elemwise_binary(mul) | Desc: I8 elementwise multiply
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @dtype_i8(
// CHECK: _outlined_vf_
module {
  func.func @dtype_i8(%arg0: tensor<8x32xi8>, %arg1: tensor<8x32xi8>) -> tensor<8x32xi8>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<8x32xi8>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8x32xi8>, tensor<8x32xi8>) outs(%dst : tensor<8x32xi8>) -> tensor<8x32xi8>
    return %res : tensor<8x32xi8>
  }
}

// -----
// TEST_SPEC: dtype_i32 | Shape: 4x64xi32 | Ops: elemwise_binary(add) | Desc: I32 elementwise add
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @dtype_i32(
// CHECK: _outlined_vf_
module {
  func.func @dtype_i32(%arg0: tensor<4x64xi32>, %arg1: tensor<4x64xi32>) -> tensor<4x64xi32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<4x64xi32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<4x64xi32>, tensor<4x64xi32>) outs(%dst : tensor<4x64xi32>) -> tensor<4x64xi32>
    return %res : tensor<4x64xi32>
  }
}

// -----
// TEST_SPEC: dtype_i16 | Shape: 16x32xi16 | Ops: elemwise_binary(sub) | Desc: I16 elementwise subtract
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @dtype_i16(
// CHECK: _outlined_vf_
module {
  func.func @dtype_i16(%arg0: tensor<16x32xi16>, %arg1: tensor<16x32xi16>) -> tensor<16x32xi16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<16x32xi16>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<16x32xi16>, tensor<16x32xi16>) outs(%dst : tensor<16x32xi16>) -> tensor<16x32xi16>
    return %res : tensor<16x32xi16>
  }
}

// -----
// TEST_SPEC: barrier_extract_slice | Shape: 16x64xf32 | Ops: elemwise -> tensor.extract_slice (barrier) -> elemwise | Desc: ExtractSlice as non-vectorizable barrier per isNonVectorizableOp
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @barrier_extract_slice(
// CHECK: _outlined_vf_
module {
  func.func @barrier_extract_slice(%arg0: tensor<16x64xf32>, %arg1: tensor<16x64xf32>) -> (tensor<16x64xf32>, tensor<1x32xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %full_init = tensor.empty() : tensor<16x64xf32>
    %full = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%full_init : tensor<16x64xf32>) -> tensor<16x64xf32>
    %slice0 = tensor.extract_slice %full[0, 0] [1, 32] [1, 1] : tensor<16x64xf32> to tensor<1x32xf32>
    %slice1 = tensor.extract_slice %full[0, 32] [1, 32] [1, 1] : tensor<16x64xf32> to tensor<1x32xf32>
    %small_init = tensor.empty() : tensor<1x32xf32>
    %small = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%slice0, %slice1 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%small_init : tensor<1x32xf32>) -> tensor<1x32xf32>
    return %full, %small : tensor<16x64xf32>, tensor<1x32xf32>
  }
}

// -----
// TEST_SPEC: barrier_hfusion_cast | Shape: 32xf32 | Ops: elemwise -> hfusion.cast (barrier) -> elemwise | Desc: HFusion CastOp as non-vectorizable barrier per isNonVectorizableOp
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @barrier_hfusion_cast(
// CHECK: _outlined_vf_
module {
  func.func @barrier_hfusion_cast(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %add_init = tensor.empty() : tensor<32xf32>
    %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%add_init : tensor<32xf32>) -> tensor<32xf32>
    %cast_dst = tensor.empty() : tensor<32xf16>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%add : tensor<32xf32>) outs(%cast_dst : tensor<32xf16>) -> tensor<32xf16>
    %mul_init = tensor.empty() : tensor<32xf16>
    %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cast, %cast : tensor<32xf16>, tensor<32xf16>) outs(%mul_init : tensor<32xf16>) -> tensor<32xf16>
    return %mul : tensor<32xf16>
  }
}

// -----
// TEST_SPEC: barrier_tensor_collapse | Shape: 4x4x8xf32 | Ops: elemwise -> tensor.collapse_shape (barrier) -> elemwise | Desc: CollapseShape as non-vectorizable barrier
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @barrier_tensor_collapse(
// CHECK: _outlined_vf_
module {
  func.func @barrier_tensor_collapse(%arg0: tensor<4x4x8xf32>, %arg1: tensor<4x4x8xf32>) -> tensor<16x8xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %add_init = tensor.empty() : tensor<4x4x8xf32>
    %add = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<4x4x8xf32>, tensor<4x4x8xf32>) outs(%add_init : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    %collapsed = tensor.collapse_shape %add [[0, 1], [2]] : tensor<4x4x8xf32> into tensor<16x8xf32>
    %mul_init = tensor.empty() : tensor<16x8xf32>
    %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %collapsed : tensor<16x8xf32>, tensor<16x8xf32>) outs(%mul_init : tensor<16x8xf32>) -> tensor<16x8xf32>
    return %mul : tensor<16x8xf32>
  }
}

// -----
// TEST_SPEC: softmax_like_chain | Shape: 1x64xf32 | Ops: sub -> exp -> reduce -> broadcast -> mul | Desc: Softmax-like chain with reduction and broadcast
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @softmax_like_chain(
// CHECK: _outlined_vf_
module {
  func.func @softmax_like_chain(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<1x64xf32>
    %tmp2 = tensor.empty() : tensor<1x64xf32>
    %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%tmp : tensor<1x64xf32>) -> tensor<1x64xf32>
    %exp = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%sub : tensor<1x64xf32>) outs(%tmp2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %c0 = arith.constant 0.0 : f32
    %sum_init = tensor.empty() : tensor<1xf32>
    %sum_filled = linalg.fill ins(%c0 : f32) outs(%sum_init : tensor<1xf32>) -> tensor<1xf32>
    %sum = linalg.reduce ins(%exp : tensor<1x64xf32>) outs(%sum_filled : tensor<1xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %add = arith.addf %in, %init : f32
        linalg.yield %add : f32
      }
    %bcast_dst = tensor.empty() : tensor<1x64xf32>
    %bcast = linalg.broadcast ins(%sum : tensor<1xf32>) outs(%bcast_dst : tensor<1x64xf32>) dimensions = [1]
    %out_init = tensor.empty() : tensor<1x64xf32>
    %out = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%exp, %bcast : tensor<1x64xf32>, tensor<1x64xf32>) outs(%out_init : tensor<1x64xf32>) -> tensor<1x64xf32>
    return %out : tensor<1x64xf32>
  }
}

// -----
// TEST_SPEC: elemwise_long_chain_2d | Shape: 8x32xf32 | Ops: add -> sub -> mul -> add | Desc: Long elementwise chain on 2D tensor
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @elemwise_long_chain_2d(
// CHECK: _outlined_vf_
module {
  func.func @elemwise_long_chain_2d(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<8x32xf32>
    %a = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%tmp : tensor<8x32xf32>) -> tensor<8x32xf32>
    %b = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%a, %arg2 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%tmp : tensor<8x32xf32>) -> tensor<8x32xf32>
    %c = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%b, %arg0 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%tmp : tensor<8x32xf32>) -> tensor<8x32xf32>
    %d = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%c, %arg1 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%tmp : tensor<8x32xf32>) -> tensor<8x32xf32>
    return %d : tensor<8x32xf32>
  }
}

// -----
// TEST_SPEC: fill_elemwise_reduce | Shape: 8x64xf32 | Ops: fill -> elemwise_binary(mul) -> reduce | Desc: Fill feeding elementwise and reduction
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @fill_elemwise_reduce(
// CHECK: _outlined_vf_
module {
  func.func @fill_elemwise_reduce(%arg0: tensor<8x64xf32>) -> tensor<8xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c1 = arith.constant 1.0 : f32
    %filled_init = tensor.empty() : tensor<8x64xf32>
    %filled = linalg.fill ins(%c1 : f32) outs(%filled_init : tensor<8x64xf32>) -> tensor<8x64xf32>
    %mul_init = tensor.empty() : tensor<8x64xf32>
    %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %filled : tensor<8x64xf32>, tensor<8x64xf32>) outs(%mul_init : tensor<8x64xf32>) -> tensor<8x64xf32>
    %c0 = arith.constant 0.0 : f32
    %red_init = tensor.empty() : tensor<8xf32>
    %red_filled = linalg.fill ins(%c0 : f32) outs(%red_init : tensor<8xf32>) -> tensor<8xf32>
    %sum = linalg.reduce ins(%mul : tensor<8x64xf32>) outs(%red_filled : tensor<8xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %add = arith.addf %in, %init : f32
        linalg.yield %add : f32
      }
    return %sum : tensor<8xf32>
  }
}

// -----
// TEST_SPEC: unary_sqrt_log | Shape: 64xf32 | Ops: elemwise_unary(sqrt) -> elemwise_unary(log) | Desc: Chained unary operators
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @unary_sqrt_log(
// CHECK: _outlined_vf_
module {
  func.func @unary_sqrt_log(%arg0: tensor<64xf32>) -> tensor<64xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<64xf32>
    %out = tensor.empty() : tensor<64xf32>
    %s = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%arg0 : tensor<64xf32>) outs(%tmp : tensor<64xf32>) -> tensor<64xf32>
    %l = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%s : tensor<64xf32>) outs(%out : tensor<64xf32>) -> tensor<64xf32>
    return %l : tensor<64xf32>
  }
}

// -----
// TEST_SPEC: unary_abs | Shape: 32xf32 | Ops: elemwise_unary(abs) | Desc: Absolute value unary op
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @unary_abs(
// CHECK: _outlined_vf_
module {
  func.func @unary_abs(%arg0: tensor<32xf32>) -> tensor<32xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %out = tensor.empty() : tensor<32xf32>
    %abs = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<32xf32>) outs(%out : tensor<32xf32>) -> tensor<32xf32>
    return %abs : tensor<32xf32>
  }
}

// -----
// TEST_SPEC: generic_simple_add | Shape: 16xf32 | Ops: linalg.generic add | Desc: Simple linalg.generic with one parallel loop
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @generic_simple_add(
// CHECK: _outlined_vf_
module {
  func.func @generic_simple_add(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<16xf32>
    %res = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %sum = arith.addf %in0, %in1 : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
    return %res : tensor<16xf32>
  }
}

// -----
// TEST_SPEC: shape_3d_irregular | Shape: 5x7x11xf32 | Ops: elemwise_binary(add) | Desc: Irregular 3D shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_3d_irregular(
// CHECK: _outlined_vf_
module {
  func.func @shape_3d_irregular(%arg0: tensor<5x7x11xf32>, %arg1: tensor<5x7x11xf32>) -> tensor<5x7x11xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<5x7x11xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<5x7x11xf32>, tensor<5x7x11xf32>) outs(%dst : tensor<5x7x11xf32>) -> tensor<5x7x11xf32>
    return %res : tensor<5x7x11xf32>
  }
}

// -----
// TEST_SPEC: shape_4d_nchw | Shape: 2x4x8x32xf32 | Ops: elemwise_binary(mul) | Desc: 4D NCHW-style shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_4d_nchw(
// CHECK: _outlined_vf_
module {
  func.func @shape_4d_nchw(%arg0: tensor<2x4x8x32xf32>, %arg1: tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<2x4x8x32xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<2x4x8x32xf32>, tensor<2x4x8x32xf32>) outs(%dst : tensor<2x4x8x32xf32>) -> tensor<2x4x8x32xf32>
    return %res : tensor<2x4x8x32xf32>
  }
}

// -----
// TEST_SPEC: shape_64x1 | Shape: 64x1xf32 | Ops: elemwise_binary(add) | Desc: Single-column 2D shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_64x1(
// CHECK: _outlined_vf_
module {
  func.func @shape_64x1(%arg0: tensor<64x1xf32>, %arg1: tensor<64x1xf32>) -> tensor<64x1xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<64x1xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%dst : tensor<64x1xf32>) -> tensor<64x1xf32>
    return %res : tensor<64x1xf32>
  }
}

// -----
// TEST_SPEC: shape_100x63 | Shape: 100x63xf32 | Ops: elemwise_binary(add) | Desc: Non-power-of-two 2D shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_100x63(
// CHECK: _outlined_vf_
module {
  func.func @shape_100x63(%arg0: tensor<100x63xf32>, %arg1: tensor<100x63xf32>) -> tensor<100x63xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<100x63xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<100x63xf32>, tensor<100x63xf32>) outs(%dst : tensor<100x63xf32>) -> tensor<100x63xf32>
    return %res : tensor<100x63xf32>
  }
}

// -----
// TEST_SPEC: shape_127_odd | Shape: 127xf32 | Ops: elemwise_binary(add) | Desc: Large odd-length 1D with tail
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_127_odd(
// CHECK: _outlined_vf_
module {
  func.func @shape_127_odd(%arg0: tensor<127xf32>, %arg1: tensor<127xf32>) -> tensor<127xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<127xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<127xf32>, tensor<127xf32>) outs(%dst : tensor<127xf32>) -> tensor<127xf32>
    return %res : tensor<127xf32>
  }
}

// -----
// TEST_SPEC: shape_prime_1d | Shape: 17xf32 | Ops: elemwise_binary(sub) | Desc: Prime-like 1D dimension
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_prime_1d(
// CHECK: _outlined_vf_
module {
  func.func @shape_prime_1d(%arg0: tensor<17xf32>, %arg1: tensor<17xf32>) -> tensor<17xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<17xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<17xf32>, tensor<17xf32>) outs(%dst : tensor<17xf32>) -> tensor<17xf32>
    return %res : tensor<17xf32>
  }
}

// -----
// TEST_SPEC: shape_prime_2d | Shape: 13x23xf32 | Ops: elemwise_binary(add) | Desc: Prime-like 2D shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_prime_2d(
// CHECK: _outlined_vf_
module {
  func.func @shape_prime_2d(%arg0: tensor<13x23xf32>, %arg1: tensor<13x23xf32>) -> tensor<13x23xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<13x23xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<13x23xf32>, tensor<13x23xf32>) outs(%dst : tensor<13x23xf32>) -> tensor<13x23xf32>
    return %res : tensor<13x23xf32>
  }
}

// -----
// TEST_SPEC: shape_128x1_tall | Shape: 128x1xf32 | Ops: elemwise_binary(add) | Desc: Tall single-column 2D
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_128x1_tall(
// CHECK: _outlined_vf_
module {
  func.func @shape_128x1_tall(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>) -> tensor<128x1xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<128x1xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<128x1xf32>, tensor<128x1xf32>) outs(%dst : tensor<128x1xf32>) -> tensor<128x1xf32>
    return %res : tensor<128x1xf32>
  }
}

// -----
// TEST_SPEC: shape_3x3_conv | Shape: 3x3xf32 | Ops: elemwise_binary(mul) | Desc: Conv kernel-like 3x3 shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_3x3_conv(
// CHECK: _outlined_vf_
module {
  func.func @shape_3x3_conv(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<3x3xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%dst : tensor<3x3xf32>) -> tensor<3x3xf32>
    return %res : tensor<3x3xf32>
  }
}

// -----
// TEST_SPEC: shape_1x64x1 | Shape: 1x64x1xf32 | Ops: elemwise_binary(add) | Desc: Batch=1, feature=64, channel=1
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_1x64x1(
// CHECK: _outlined_vf_
module {
  func.func @shape_1x64x1(%arg0: tensor<1x64x1xf32>, %arg1: tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<1x64x1xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<1x64x1xf32>, tensor<1x64x1xf32>) outs(%dst : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    return %res : tensor<1x64x1xf32>
  }
}

// -----
// TEST_SPEC: shape_8x31_odd_last | Shape: 8x31xf32 | Ops: elemwise_binary(mul) | Desc: Odd last dimension (31 < 32)
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_8x31_odd_last(
// CHECK: _outlined_vf_
module {
  func.func @shape_8x31_odd_last(%arg0: tensor<8x31xf32>, %arg1: tensor<8x31xf32>) -> tensor<8x31xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<8x31xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8x31xf32>, tensor<8x31xf32>) outs(%dst : tensor<8x31xf32>) -> tensor<8x31xf32>
    return %res : tensor<8x31xf32>
  }
}

// -----
// TEST_SPEC: shape_17x17_odd_square | Shape: 17x17xf32 | Ops: elemwise_binary(add) | Desc: Odd square 2D
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @shape_17x17_odd_square(
// CHECK: _outlined_vf_
module {
  func.func @shape_17x17_odd_square(%arg0: tensor<17x17xf32>, %arg1: tensor<17x17xf32>) -> tensor<17x17xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<17x17xf32>
    %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<17x17xf32>, tensor<17x17xf32>) outs(%dst : tensor<17x17xf32>) -> tensor<17x17xf32>
    return %res : tensor<17x17xf32>
  }
}

// -----
// TEST_SPEC: combo_reduce_broadcast_elemwise_odd | Shape: 7x13xf32 | Ops: reduce -> broadcast -> elemwise_binary | Desc: Reduce+broadcast+elemwise on odd 2D shape
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_reduce_broadcast_elemwise_odd(
// CHECK: _outlined_vf_
module {
  func.func @combo_reduce_broadcast_elemwise_odd(%arg0: tensor<7x13xf32>) -> tensor<7x13xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<7xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<7xf32>) -> tensor<7xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<7x13xf32>) outs(%filled : tensor<7xf32>) dimensions = [1]
      (%in: f32, %initval: f32) {
        %sum = arith.addf %in, %initval : f32
        linalg.yield %sum : f32
      }
    %bcast_dst = tensor.empty() : tensor<7x13xf32>
    %broadcast = linalg.broadcast ins(%reduced : tensor<7xf32>) outs(%bcast_dst : tensor<7x13xf32>) dimensions = [1]
    %out_init = tensor.empty() : tensor<7x13xf32>
    %out = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %broadcast : tensor<7x13xf32>, tensor<7x13xf32>) outs(%out_init : tensor<7x13xf32>) -> tensor<7x13xf32>
    return %out : tensor<7x13xf32>
  }
}

// -----
// TEST_SPEC: combo_transpose_reduce_irregular | Shape: 3x7x11xf32 | Ops: transpose -> elemwise -> reduce | Desc: Transpose and reduce on irregular 3D
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_transpose_reduce_irregular(
// CHECK: _outlined_vf_
module {
  func.func @combo_transpose_reduce_irregular(%arg0: tensor<3x7x11xf32>, %arg1: tensor<3x7x11xf32>) -> tensor<3x11xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<3x11x7xf32>
    %add_init = tensor.empty() : tensor<3x11x7xf32>
    %t0 = linalg.transpose ins(%arg0 : tensor<3x7x11xf32>) outs(%tmp : tensor<3x11x7xf32>) permutation = [0, 2, 1]
    %t1 = linalg.transpose ins(%arg1 : tensor<3x7x11xf32>) outs(%add_init : tensor<3x11x7xf32>) permutation = [0, 2, 1]
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%t0, %t1 : tensor<3x11x7xf32>, tensor<3x11x7xf32>) outs(%add_init : tensor<3x11x7xf32>) -> tensor<3x11x7xf32>
    %c0 = arith.constant 0.0 : f32
    %red_init = tensor.empty() : tensor<3x11xf32>
    %red_filled = linalg.fill ins(%c0 : f32) outs(%red_init : tensor<3x11xf32>) -> tensor<3x11xf32>
    %reduced = linalg.reduce ins(%sum : tensor<3x11x7xf32>) outs(%red_filled : tensor<3x11xf32>) dimensions = [2]
      (%in: f32, %initval: f32) {
        %add = arith.addf %in, %initval : f32
        linalg.yield %add : f32
      }
    return %reduced : tensor<3x11xf32>
  }
}

// -----
// TEST_SPEC: combo_unary_chain_odd | Shape: 31xf32 | Ops: sqrt -> log -> exp | Desc: Unary chain on odd 1D
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_unary_chain_odd(
// CHECK: _outlined_vf_
module {
  func.func @combo_unary_chain_odd(%arg0: tensor<31xf32>) -> tensor<31xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<31xf32>
    %out = tensor.empty() : tensor<31xf32>
    %s = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%arg0 : tensor<31xf32>) outs(%tmp : tensor<31xf32>) -> tensor<31xf32>
    %l = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%s : tensor<31xf32>) outs(%out : tensor<31xf32>) -> tensor<31xf32>
    %e = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%l : tensor<31xf32>) outs(%tmp : tensor<31xf32>) -> tensor<31xf32>
    return %e : tensor<31xf32>
  }
}

// -----
// TEST_SPEC: combo_elemwise_transpose_3x3 | Shape: 3x3xf32 | Ops: elemwise_binary(add) -> transpose | Desc: Elemwise then transpose on 3x3
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_elemwise_transpose_3x3(
// CHECK: _outlined_vf_
module {
  func.func @combo_elemwise_transpose_3x3(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<3x3xf32>
    %out = tensor.empty() : tensor<3x3xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%dst : tensor<3x3xf32>) -> tensor<3x3xf32>
    %t = linalg.transpose ins(%sum : tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) permutation = [1, 0]
    return %t : tensor<3x3xf32>
  }
}

// -----
// TEST_SPEC: combo_reduce_odd_inner | Shape: 4x8x15xf32 | Ops: elemwise -> reduce | Desc: Reduce on odd last dim (15)
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_reduce_odd_inner(
// CHECK: _outlined_vf_
module {
  func.func @combo_reduce_odd_inner(%arg0: tensor<4x8x15xf32>, %arg1: tensor<4x8x15xf32>) -> tensor<4x8xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %dst = tensor.empty() : tensor<4x8x15xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<4x8x15xf32>, tensor<4x8x15xf32>) outs(%dst : tensor<4x8x15xf32>) -> tensor<4x8x15xf32>
    %c0 = arith.constant 0.0 : f32
    %red_init = tensor.empty() : tensor<4x8xf32>
    %red_filled = linalg.fill ins(%c0 : f32) outs(%red_init : tensor<4x8xf32>) -> tensor<4x8xf32>
    %reduced = linalg.reduce ins(%sum : tensor<4x8x15xf32>) outs(%red_filled : tensor<4x8xf32>) dimensions = [2]
      (%in: f32, %initval: f32) {
        %add = arith.addf %in, %initval : f32
        linalg.yield %add : f32
      }
    return %reduced : tensor<4x8xf32>
  }
}

// -----
// TEST_SPEC: combo_elemwise_chain_prime | Shape: 13x17xf16 | Ops: add -> mul -> sub | Desc: Elemwise chain on prime 2D with f16
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_elemwise_chain_prime(
// CHECK: _outlined_vf_
module {
  func.func @combo_elemwise_chain_prime(%arg0: tensor<13x17xf16>, %arg1: tensor<13x17xf16>, %arg2: tensor<13x17xf16>) -> tensor<13x17xf16>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<13x17xf16>
    %a = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<13x17xf16>, tensor<13x17xf16>) outs(%tmp : tensor<13x17xf16>) -> tensor<13x17xf16>
    %b = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%a, %arg2 : tensor<13x17xf16>, tensor<13x17xf16>) outs(%tmp : tensor<13x17xf16>) -> tensor<13x17xf16>
    %c = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%b, %arg0 : tensor<13x17xf16>, tensor<13x17xf16>) outs(%tmp : tensor<13x17xf16>) -> tensor<13x17xf16>
    return %c : tensor<13x17xf16>
  }
}

// -----
// TEST_SPEC: combo_softmax_odd | Shape: 1x31xf32 | Ops: sub -> exp -> reduce -> broadcast -> div | Desc: Softmax-like on odd inner dim
// EXPECTED: Pass succeeds, generates _outlined_vf_* functions
// CHECK-LABEL: func @combo_softmax_odd(
// CHECK: _outlined_vf_
module {
  func.func @combo_softmax_odd(%arg0: tensor<1x31xf32>, %arg1: tensor<1x31xf32>) -> tensor<1x31xf32>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %tmp = tensor.empty() : tensor<1x31xf32>
    %tmp2 = tensor.empty() : tensor<1x31xf32>
    %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<1x31xf32>, tensor<1x31xf32>) outs(%tmp : tensor<1x31xf32>) -> tensor<1x31xf32>
    %exp = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%sub : tensor<1x31xf32>) outs(%tmp2 : tensor<1x31xf32>) -> tensor<1x31xf32>
    %c0 = arith.constant 0.0 : f32
    %sum_init = tensor.empty() : tensor<1xf32>
    %sum_filled = linalg.fill ins(%c0 : f32) outs(%sum_init : tensor<1xf32>) -> tensor<1xf32>
    %sum = linalg.reduce ins(%exp : tensor<1x31xf32>) outs(%sum_filled : tensor<1xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %add = arith.addf %in, %init : f32
        linalg.yield %add : f32
      }
    %bcast_dst = tensor.empty() : tensor<1x31xf32>
    %bcast = linalg.broadcast ins(%sum : tensor<1xf32>) outs(%bcast_dst : tensor<1x31xf32>) dimensions = [1]
    %out_init = tensor.empty() : tensor<1x31xf32>
    %out = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%exp, %bcast : tensor<1x31xf32>, tensor<1x31xf32>) outs(%out_init : tensor<1x31xf32>) -> tensor<1x31xf32>
    return %out : tensor<1x31xf32>
  }
}
