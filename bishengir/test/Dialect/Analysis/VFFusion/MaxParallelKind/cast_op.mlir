// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel enable-cast-opt=true" --split-input-file %s | FileCheck %s

// Narrowing f32 -> bf16 followed by transpose.
// CHECK-LABEL: func.func private @narrowing_cast_f322bf16_with_transpose_fused_0(
// CHECK: %[[NBT_SUB:.*]] = linalg.sub
// CHECK: %[[NBT_EXP:.*]] = linalg.exp ins(%[[NBT_SUB]] : tensor<32x64xf32>)
// CHECK: %[[NBT_RED:.*]] = linalg.reduce ins(%[[NBT_EXP]] : tensor<32x64xf32>)
// CHECK: return %[[NBT_EXP]], %[[NBT_RED]] : tensor<32x64xf32>, tensor<32xf32>

// CHECK-LABEL: func.func private @narrowing_cast_f322bf16_with_transpose_fused_1(
// CHECK: %[[NBT_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32x64xf32>) {{.*}} -> tensor<32x64xbf16>
// CHECK: %[[NBT_EXPAND0:.*]] = tensor.expand_shape %[[NBT_CAST]] {{.*}} into tensor<32x4x16xbf16>
// CHECK: %[[NBT_TRANSPOSE:.*]] = linalg.transpose ins(%[[NBT_EXPAND0]] : tensor<32x4x16xbf16>) {{.*}} permutation = [1, 0, 2]
// CHECK: return %[[NBT_TRANSPOSE]] : tensor<4x32x16xbf16>

// CHECK-LABEL: func.func @narrowing_cast_f322bf16_with_transpose(
// CHECK: %[[BF_SIDE_SUB:.*]] = linalg.sub ins(%arg2, %arg3 : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[BF_FUSED0:.*]]:2 = call @narrowing_cast_f322bf16_with_transpose_fused_0(
// CHECK: hivm.hir.copy ins(%[[BF_FUSED0]]#1 : tensor<32xf32>)
// CHECK: %[[BF_FUSED1:.*]] = call @narrowing_cast_f322bf16_with_transpose_fused_1(%[[BF_FUSED0]]#0, {{.*}}) {{.*}} -> tensor<4x32x16xbf16>
// CHECK: return %[[BF_FUSED0]]#1, %[[BF_FUSED1]] : tensor<32xf32>, tensor<4x32x16xbf16>

module {
  func.func @narrowing_cast_f322bf16_with_transpose(%arg0 : tensor<32x64xf32>, %arg1 : tensor<32x64xf32>, %arg2 : tensor<32xf32>, %arg3 : tensor<32xf32>) -> (tensor<32xf32>, tensor<4x32x16xbf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %alloc_53 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = tensor.empty() : tensor<32x64xf32>
    %2 = tensor.empty() : tensor<32x64xbf16>
    %3 = tensor.empty() : tensor<32xf32>
    %4 = tensor.empty() : tensor<4x32x16xbf16>
    %5 = tensor.empty() : tensor<32xf32>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %exp = linalg.exp ins(%sub : tensor<32x64xf32>) outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%exp : tensor<32x64xf32>) outs(%2 : tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xf32>, tensor<32xf32>) outs(%5 : tensor<32xf32>) -> tensor<32xf32>

    hivm.hir.copy ins(%263 : tensor<32xf32>) outs(%memspacecast_50 : memref<32xf32>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%exp : tensor<32x64xf32>) outs(%3 : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %274 = arith.addf %in, %init : f32
        linalg.yield %274 : f32
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xf32>) outs(%memspacecast_54 : memref<32xf32>) {tiled_op}

    %expanded_85 = tensor.expand_shape %cast [[0], [1, 2]] output_shape [32, 4, 16] : tensor<32x64xbf16> into tensor<32x4x16xbf16>
    %transposed_86 = linalg.transpose ins(%expanded_85 : tensor<32x4x16xbf16>) outs(%4 : tensor<4x32x16xbf16>) permutation = [1, 0, 2]

    return %reduced_84, %transposed_86 : tensor<32xf32>, tensor<4x32x16xbf16>

  }
}

// Narrowing f32 -> f16 followed by transpose.
// CHECK-LABEL: func.func private @narrowing_cast_f322f16_with_transpose_fused_0(
// CHECK: %[[NFT_SUB:.*]] = linalg.sub
// CHECK: %[[NFT_EXP:.*]] = linalg.exp ins(%[[NFT_SUB]] : tensor<32x64xf32>)
// CHECK: %[[NFT_RED:.*]] = linalg.reduce ins(%[[NFT_EXP]] : tensor<32x64xf32>)
// CHECK: return %[[NFT_EXP]], %[[NFT_RED]] : tensor<32x64xf32>, tensor<32xf32>

// CHECK-LABEL: func.func private @narrowing_cast_f322f16_with_transpose_fused_1(
// CHECK: %[[NFT_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32x64xf32>) {{.*}} -> tensor<32x64xf16>
// CHECK: %[[NFT_EXPAND0:.*]] = tensor.expand_shape %[[NFT_CAST]] {{.*}} into tensor<32x4x16xf16>
// CHECK: %[[NFT_TRANSPOSE:.*]] = linalg.transpose ins(%[[NFT_EXPAND0]] : tensor<32x4x16xf16>) {{.*}} permutation = [1, 0, 2]
// CHECK: return %[[NFT_TRANSPOSE]] : tensor<4x32x16xf16>

// CHECK-LABEL: func.func @narrowing_cast_f322f16_with_transpose(
// CHECK: %[[F16_SIDE_SUB:.*]] = linalg.sub ins(%arg2, %arg3 : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[F16_FUSED0:.*]]:2 = call @narrowing_cast_f322f16_with_transpose_fused_0(
// CHECK: hivm.hir.copy ins(%[[F16_FUSED0]]#1 : tensor<32xf32>)
// CHECK: %[[F16_FUSED1:.*]] = call @narrowing_cast_f322f16_with_transpose_fused_1(%[[F16_FUSED0]]#0, {{.*}}) {{.*}} -> tensor<4x32x16xf16>
// CHECK: return %[[F16_FUSED0]]#1, %[[F16_FUSED1]] : tensor<32xf32>, tensor<4x32x16xf16>

module {
  func.func @narrowing_cast_f322f16_with_transpose(%arg0 : tensor<32x64xf32>, %arg1 : tensor<32x64xf32>, %arg2 : tensor<32xf32>, %arg3 : tensor<32xf32>) -> (tensor<32xf32>, tensor<4x32x16xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %alloc_53 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = tensor.empty() : tensor<32x64xf32>
    %2 = tensor.empty() : tensor<32x64xf16>
    %3 = tensor.empty() : tensor<32xf32>
    %4 = tensor.empty() : tensor<4x32x16xf16>
    %5 = tensor.empty() : tensor<32xf32>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %exp = linalg.exp ins(%sub : tensor<32x64xf32>) outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%exp : tensor<32x64xf32>) outs(%2 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xf32>, tensor<32xf32>) outs(%5 : tensor<32xf32>) -> tensor<32xf32>

    hivm.hir.copy ins(%263 : tensor<32xf32>) outs(%memspacecast_50 : memref<32xf32>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%exp : tensor<32x64xf32>) outs(%3 : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %274 = arith.addf %in, %init : f32
        linalg.yield %274 : f32
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xf32>) outs(%memspacecast_54 : memref<32xf32>) {tiled_op}

    %expanded_85 = tensor.expand_shape %cast [[0], [1, 2]] output_shape [32, 4, 16] : tensor<32x64xf16> into tensor<32x4x16xf16>
    %transposed_86 = linalg.transpose ins(%expanded_85 : tensor<32x4x16xf16>) outs(%4 : tensor<4x32x16xf16>) permutation = [1, 0, 2]

    return %reduced_84, %transposed_86 : tensor<32xf32>, tensor<4x32x16xf16>

  }
}

// Narrowing f32 -> f16 with consumers in f16.
// CHECK-LABEL: func.func private @narrowing_cast_f322f16_with_compute_op_fused_0(
// CHECK: %[[NFC_SUB:.*]] = linalg.sub
// CHECK: %[[NFC_EXP:.*]] = linalg.exp ins(%[[NFC_SUB]] : tensor<32x64xf32>)
// CHECK: %[[NFC_RED:.*]] = linalg.reduce ins(%[[NFC_EXP]] : tensor<32x64xf32>)
// CHECK: return %[[NFC_EXP]], %[[NFC_RED]] : tensor<32x64xf32>, tensor<32xf32>
// CHECK-LABEL: func.func private @narrowing_cast_f322f16_with_compute_op_fused_1(
// CHECK: %[[NFC_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32x64xf32>) {{.*}} -> tensor<32x64xf16>
// CHECK: %[[NFC_ADD:.*]] = linalg.add ins(%[[NFC_CAST]], %arg2 : tensor<32x64xf16>, tensor<32x64xf16>)
// CHECK: %[[NFC_MUL:.*]] = linalg.mul ins(%[[NFC_ADD]], %arg4 : tensor<32x64xf16>, tensor<32x64xf16>)
// CHECK: return %[[NFC_MUL]] : tensor<32x64xf16>
// CHECK-LABEL: func.func @narrowing_cast_f322f16_with_compute_op(
// CHECK: %[[NFC_FUSED0:.*]]:2 = call @narrowing_cast_f322f16_with_compute_op_fused_0(
// CHECK: %[[NFC_FUSED1:.*]] = call @narrowing_cast_f322f16_with_compute_op_fused_1(%[[NFC_FUSED0]]#0,
// CHECK: return %[[NFC_FUSED0]]#1, %[[NFC_FUSED1]] : tensor<32xf32>, tensor<32x64xf16>

module {
  func.func @narrowing_cast_f322f16_with_compute_op(%arg0 : tensor<32x64xf32>, %arg1 : tensor<32x64xf32>, %arg2 : tensor<32xf32>, %arg3 : tensor<32xf32>, %arg4 : tensor<32x64xf16>, %arg5 : tensor<32x64xf16>) -> (tensor<32xf32>, tensor<32x64xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %alloc_53 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = tensor.empty() : tensor<32x64xf32>
    %2 = tensor.empty() : tensor<32x64xf16>
    %3 = tensor.empty() : tensor<32xf32>
    %5 = tensor.empty() : tensor<32xf32>
    %6 = tensor.empty() : tensor<32x64xf16>
    %7 = tensor.empty() : tensor<32x64xf16>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %exp = linalg.exp ins(%sub : tensor<32x64xf32>) outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%exp : tensor<32x64xf32>) outs(%2 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xf32>, tensor<32xf32>) outs(%5 : tensor<32xf32>) -> tensor<32xf32>

    hivm.hir.copy ins(%263 : tensor<32xf32>) outs(%memspacecast_50 : memref<32xf32>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%exp : tensor<32x64xf32>) outs(%3 : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %274 = arith.addf %in, %init : f32
        linalg.yield %274 : f32
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xf32>) outs(%memspacecast_54 : memref<32xf32>) {tiled_op}

    %add = linalg.add ins(%cast, %arg4 : tensor<32x64xf16>, tensor<32x64xf16>) outs(%6 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %mul = linalg.mul ins(%add, %arg5 : tensor<32x64xf16>, tensor<32x64xf16>) outs(%7 : tensor<32x64xf16>) -> tensor<32x64xf16>

    return %reduced_84, %mul : tensor<32xf32>, tensor<32x64xf16>

  }
}

// Narrowing f32 -> bf16 with consumers in bf16.
// CHECK-LABEL: func.func private @narrowing_cast_f322bf16_with_compute_op_fused_0(
// CHECK: %[[NBC_SUB:.*]] = linalg.sub
// CHECK: %[[NBC_EXP:.*]] = linalg.exp ins(%[[NBC_SUB]] : tensor<32x64xf32>)
// CHECK: %[[NBC_RED:.*]] = linalg.reduce ins(%[[NBC_EXP]] : tensor<32x64xf32>)
// CHECK: return %[[NBC_EXP]], %[[NBC_RED]] : tensor<32x64xf32>, tensor<32xf32>
// CHECK-LABEL: func.func private @narrowing_cast_f322bf16_with_compute_op_fused_1(
// CHECK: %[[NBC_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32x64xf32>) {{.*}} -> tensor<32x64xbf16>
// CHECK: %[[NBC_ADD:.*]] = linalg.add ins(%[[NBC_CAST]], %arg2 : tensor<32x64xbf16>, tensor<32x64xbf16>)
// CHECK: %[[NBC_MUL:.*]] = linalg.mul ins(%[[NBC_ADD]], %arg4 : tensor<32x64xbf16>, tensor<32x64xbf16>)
// CHECK: return %[[NBC_MUL]] : tensor<32x64xbf16>
// CHECK-LABEL: func.func @narrowing_cast_f322bf16_with_compute_op(
// CHECK: %[[NBC_FUSED0:.*]]:2 = call @narrowing_cast_f322bf16_with_compute_op_fused_0(
// CHECK: %[[NBC_FUSED1:.*]] = call @narrowing_cast_f322bf16_with_compute_op_fused_1(%[[NBC_FUSED0]]#0,
// CHECK: return %[[NBC_FUSED0]]#1, %[[NBC_FUSED1]] : tensor<32xf32>, tensor<32x64xbf16>

module {
  func.func @narrowing_cast_f322bf16_with_compute_op(%arg0 : tensor<32x64xf32>, %arg1 : tensor<32x64xf32>, %arg2 : tensor<32xf32>, %arg3 : tensor<32xf32>, %arg4 : tensor<32x64xbf16>, %arg5 : tensor<32x64xbf16>) -> (tensor<32xf32>, tensor<32x64xbf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %alloc_53 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf32, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xf32, #hivm.address_space<ub>> to memref<32xf32>

    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = tensor.empty() : tensor<32x64xf32>
    %2 = tensor.empty() : tensor<32x64xbf16>
    %3 = tensor.empty() : tensor<32xf32>
    %5 = tensor.empty() : tensor<32xf32>
    %6 = tensor.empty() : tensor<32x64xbf16>
    %7 = tensor.empty() : tensor<32x64xbf16>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %exp = linalg.exp ins(%sub : tensor<32x64xf32>) outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%exp : tensor<32x64xf32>) outs(%2 : tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xf32>, tensor<32xf32>) outs(%5 : tensor<32xf32>) -> tensor<32xf32>

    hivm.hir.copy ins(%263 : tensor<32xf32>) outs(%memspacecast_50 : memref<32xf32>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%exp : tensor<32x64xf32>) outs(%3 : tensor<32xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %274 = arith.addf %in, %init : f32
        linalg.yield %274 : f32
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xf32>) outs(%memspacecast_54 : memref<32xf32>) {tiled_op}

    %add = linalg.add ins(%cast, %arg4 : tensor<32x64xbf16>, tensor<32x64xbf16>) outs(%6 : tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %mul = linalg.mul ins(%add, %arg5 : tensor<32x64xbf16>, tensor<32x64xbf16>) outs(%7 : tensor<32x64xbf16>) -> tensor<32x64xbf16>

    return %reduced_84, %mul : tensor<32xf32>, tensor<32x64xbf16>

  }
}

// Widening bf16 -> f32 with consumers in f32.
// CHECK-LABEL: func.func private @widening_cast_bf162f32_with_compute_op_fused_0(
// CHECK: %[[WBC_SUB:.*]] = linalg.sub
// CHECK: %[[WBC_EXP:.*]] = linalg.exp ins(%[[WBC_SUB]] : tensor<32x64xbf16>)
// CHECK: %[[WBC_RED:.*]] = linalg.reduce ins(%[[WBC_EXP]] : tensor<32x64xbf16>)
// CHECK: return %[[WBC_EXP]], %[[WBC_RED]] : tensor<32x64xbf16>, tensor<32xbf16>
// CHECK-LABEL: func.func private @widening_cast_bf162f32_with_compute_op_fused_1(
// CHECK: %[[WBC_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32x64xbf16>) {{.*}} -> tensor<32x64xf32>
// CHECK: %[[WBC_ADD:.*]] = linalg.add ins(%[[WBC_CAST]], %arg2 : tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK: %[[WBC_MUL:.*]] = linalg.mul ins(%[[WBC_ADD]], %arg4 : tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK: return %[[WBC_MUL]] : tensor<32x64xf32>
// CHECK-LABEL: func.func @widening_cast_bf162f32_with_compute_op(
// CHECK: %[[WBC_FUSED0:.*]]:2 = call @widening_cast_bf162f32_with_compute_op_fused_0(
// CHECK: %[[WBC_FUSED1:.*]] = call @widening_cast_bf162f32_with_compute_op_fused_1(%[[WBC_FUSED0]]#0,
// CHECK: return %[[WBC_FUSED0]]#1, %[[WBC_FUSED1]] : tensor<32xbf16>, tensor<32x64xf32>

module {
  func.func @widening_cast_bf162f32_with_compute_op(%arg0 : tensor<32x64xbf16>, %arg1 : tensor<32x64xbf16>, %arg2 : tensor<32xbf16>, %arg3 : tensor<32xbf16>, %arg4 : tensor<32x64xf32>, %arg5 : tensor<32x64xf32>) -> (tensor<32xbf16>, tensor<32x64xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xbf16, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xbf16, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xbf16, #hivm.address_space<ub>> to memref<32xbf16>

    %alloc_53 = memref.alloc() : memref<32xbf16, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xbf16, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xbf16, #hivm.address_space<ub>> to memref<32xbf16>

    %0 = tensor.empty() : tensor<32x64xbf16>
    %1 = tensor.empty() : tensor<32x64xbf16>
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = tensor.empty() : tensor<32xbf16>
    %5 = tensor.empty() : tensor<32xbf16>
    %6 = tensor.empty() : tensor<32x64xf32>
    %7 = tensor.empty() : tensor<32x64xf32>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xbf16>, tensor<32x64xbf16>) outs(%0 : tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %exp = linalg.exp ins(%sub : tensor<32x64xbf16>) outs(%1 : tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%exp : tensor<32x64xbf16>) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xbf16>, tensor<32xbf16>) outs(%5 : tensor<32xbf16>) -> tensor<32xbf16>

    hivm.hir.copy ins(%263 : tensor<32xbf16>) outs(%memspacecast_50 : memref<32xbf16>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%exp : tensor<32x64xbf16>) outs(%3 : tensor<32xbf16>) dimensions = [1]
      (%in: bf16, %init: bf16) {
        %274 = arith.addf %in, %init : bf16
        linalg.yield %274 : bf16
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xbf16>) outs(%memspacecast_54 : memref<32xbf16>) {tiled_op}

    %add = linalg.add ins(%cast, %arg4 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%6 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %mul = linalg.mul ins(%add, %arg5 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%7 : tensor<32x64xf32>) -> tensor<32x64xf32>

    return %reduced_84, %mul : tensor<32xbf16>, tensor<32x64xf32>

  }
}

// Widening f16 -> f32 with consumers in f32.
// CHECK-LABEL: func.func private @widening_cast_f162f32_with_compute_op_fused_0(
// CHECK: %[[WFC_SUB:.*]] = linalg.sub
// CHECK: %[[WFC_EXP:.*]] = linalg.exp ins(%[[WFC_SUB]] : tensor<32x64xf16>)
// CHECK: %[[WFC_RED:.*]] = linalg.reduce ins(%[[WFC_EXP]] : tensor<32x64xf16>)
// CHECK: return %[[WFC_EXP]], %[[WFC_RED]] : tensor<32x64xf16>, tensor<32xf16>
// CHECK-LABEL: func.func private @widening_cast_f162f32_with_compute_op_fused_1(
// CHECK: %[[WFC_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<32x64xf16>) {{.*}} -> tensor<32x64xf32>
// CHECK: %[[WFC_ADD:.*]] = linalg.add ins(%[[WFC_CAST]], %arg2 : tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK: %[[WFC_MUL:.*]] = linalg.mul ins(%[[WFC_ADD]], %arg4 : tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK: return %[[WFC_MUL]] : tensor<32x64xf32>
// CHECK-LABEL: func.func @widening_cast_f162f32_with_compute_op(
// CHECK: %[[WFC_FUSED0:.*]]:2 = call @widening_cast_f162f32_with_compute_op_fused_0(
// CHECK: %[[WFC_FUSED1:.*]] = call @widening_cast_f162f32_with_compute_op_fused_1(%[[WFC_FUSED0]]#0,
// CHECK: return %[[WFC_FUSED0]]#1, %[[WFC_FUSED1]] : tensor<32xf16>, tensor<32x64xf32>

module {
  func.func @widening_cast_f162f32_with_compute_op(%arg0 : tensor<32x64xf16>, %arg1 : tensor<32x64xf16>, %arg2 : tensor<32xf16>, %arg3 : tensor<32xf16>, %arg4 : tensor<32x64xf32>, %arg5 : tensor<32x64xf32>) -> (tensor<32xf16>, tensor<32x64xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xf16, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf16, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xf16, #hivm.address_space<ub>> to memref<32xf16>

    %alloc_53 = memref.alloc() : memref<32xf16, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xf16, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xf16, #hivm.address_space<ub>> to memref<32xf16>

    %0 = tensor.empty() : tensor<32x64xf16>
    %1 = tensor.empty() : tensor<32x64xf16>
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = tensor.empty() : tensor<32xf16>
    %5 = tensor.empty() : tensor<32xf16>
    %6 = tensor.empty() : tensor<32x64xf32>
    %7 = tensor.empty() : tensor<32x64xf32>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xf16>, tensor<32x64xf16>) outs(%0 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %exp = linalg.exp ins(%sub : tensor<32x64xf16>) outs(%1 : tensor<32x64xf16>) -> tensor<32x64xf16>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%exp : tensor<32x64xf16>) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xf16>, tensor<32xf16>) outs(%5 : tensor<32xf16>) -> tensor<32xf16>

    hivm.hir.copy ins(%263 : tensor<32xf16>) outs(%memspacecast_50 : memref<32xf16>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%exp : tensor<32x64xf16>) outs(%3 : tensor<32xf16>) dimensions = [1]
      (%in: f16, %init: f16) {
        %274 = arith.addf %in, %init : f16
        linalg.yield %274 : f16
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xf16>) outs(%memspacecast_54 : memref<32xf16>) {tiled_op}

    %add = linalg.add ins(%cast, %arg4 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%6 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %mul = linalg.mul ins(%add, %arg5 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%7 : tensor<32x64xf32>) -> tensor<32x64xf32>

    return %reduced_84, %mul : tensor<32xf16>, tensor<32x64xf32>

  }
}

// Same-width i32 -> f32 cast.
// CHECK-LABEL: func.func private @same_width_cast_i322f32_with_compute_op_fused_0(
// CHECK: %[[SWC_SUB:.*]] = linalg.sub
// CHECK: %[[SWC_ABS:.*]] = linalg.abs ins(%[[SWC_SUB]] : tensor<32x64xi32>)
// CHECK: %[[SWC_RED:.*]] = linalg.reduce ins(%[[SWC_ABS]] : tensor<32x64xi32>)
// CHECK: return %[[SWC_ABS]], %[[SWC_RED]] : tensor<32x64xi32>, tensor<32xi32>
// CHECK-LABEL: func.func private @same_width_cast_i322f32_with_compute_op_fused_1(
// CHECK: %[[SWC_CAST:.*]] = hfusion.cast {{.*}}enable_overflow = false{{.*}} ins(%arg0 : tensor<32x64xi32>) {{.*}} -> tensor<32x64xf32>
// CHECK: %[[SWC_ADD:.*]] = linalg.add ins(%[[SWC_CAST]], %arg2 : tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK: %[[SWC_MUL:.*]] = linalg.mul ins(%[[SWC_ADD]], %arg4 : tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK: return %[[SWC_MUL]] : tensor<32x64xf32>
// CHECK-LABEL: func.func @same_width_cast_i322f32_with_compute_op(
// CHECK: %[[SWC_FUSED0:.*]]:2 = call @same_width_cast_i322f32_with_compute_op_fused_0(
// CHECK: %[[SWC_FUSED1:.*]] = call @same_width_cast_i322f32_with_compute_op_fused_1(%[[SWC_FUSED0]]#0,
// CHECK: return %[[SWC_FUSED0]]#1, %[[SWC_FUSED1]] : tensor<32xi32>, tensor<32x64xf32>

module {
  func.func @same_width_cast_i322f32_with_compute_op(%arg0 : tensor<32x64xi32>, %arg1 : tensor<32x64xi32>, %arg2 : tensor<32xi32>, %arg3 : tensor<32xi32>, %arg4 : tensor<32x64xf32>, %arg5 : tensor<32x64xf32>) -> (tensor<32xi32>, tensor<32x64xf32>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc_49 = memref.alloc() : memref<32xi32, #hivm.address_space<ub>>
    annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xi32, #hivm.address_space<ub>>
    %memspacecast_50 = memref.memory_space_cast %alloc_49 : memref<32xi32, #hivm.address_space<ub>> to memref<32xi32>

    %alloc_53 = memref.alloc() : memref<32xi32, #hivm.address_space<ub>>
    annotation.mark %alloc_53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<32xi32, #hivm.address_space<ub>>
    %memspacecast_54 = memref.memory_space_cast %alloc_53 : memref<32xi32, #hivm.address_space<ub>> to memref<32xi32>

    %0 = tensor.empty() : tensor<32x64xi32>
    %1 = tensor.empty() : tensor<32x64xi32>
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = tensor.empty() : tensor<32xi32>
    %5 = tensor.empty() : tensor<32xi32>
    %6 = tensor.empty() : tensor<32x64xf32>
    %7 = tensor.empty() : tensor<32x64xf32>

    %sub = linalg.sub ins(%arg0, %arg1 : tensor<32x64xi32>, tensor<32x64xi32>) outs(%0 : tensor<32x64xi32>) -> tensor<32x64xi32>
    %abs = linalg.abs ins(%sub : tensor<32x64xi32>) outs(%1 : tensor<32x64xi32>) -> tensor<32x64xi32>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%abs : tensor<32x64xi32>) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %263 = linalg.sub ins(%arg2, %arg3 : tensor<32xi32>, tensor<32xi32>) outs(%5 : tensor<32xi32>) -> tensor<32xi32>

    hivm.hir.copy ins(%263 : tensor<32xi32>) outs(%memspacecast_50 : memref<32xi32>) {tiled_op}

    %reduced_84 = linalg.reduce ins(%abs : tensor<32x64xi32>) outs(%3 : tensor<32xi32>) dimensions = [1]
      (%in: i32, %init: i32) {
        %274 = arith.addi %in, %init : i32
        linalg.yield %274 : i32
      }

    hivm.hir.copy ins(%reduced_84 : tensor<32xi32>) outs(%memspacecast_54 : memref<32xi32>) {tiled_op}

    %add = linalg.add ins(%cast, %arg4 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%6 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %mul = linalg.mul ins(%add, %arg5 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%7 : tensor<32x64xf32>) -> tensor<32x64xf32>

    return %reduced_84, %mul : tensor<32xi32>, tensor<32x64xf32>

  }
}

// Narrowing f32 -> f16 without a fusible producer before cast.
// CHECK-LABEL: func.func private @narrowing_cast_f322f16_no_valid_cast_producer_fused_0(
// CHECK: %[[NFP16_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<64x128xf32>) {{.*}} -> tensor<64x128xf16>
// CHECK: %[[NFP16_CMP:.*]] = hfusion.compare {{.*}} ins(%[[NFP16_CAST]], %cst : tensor<64x128xf16>, f16)
// CHECK: %[[NFP16_SELECT:.*]] = linalg.select ins(%[[NFP16_CMP]], %cst, %arg4 : tensor<64x128xi1>, f16, tensor<64x128xf16>)
// CHECK: %[[NFP16_RED:.*]] = linalg.reduce ins(%[[NFP16_SELECT]] : tensor<64x128xf16>)
// CHECK: return %[[NFP16_RED]] : tensor<64xf16>
// CHECK-LABEL: func.func @narrowing_cast_f322f16_no_valid_cast_producer(
// CHECK: %[[NFP16_FUSED:.*]] = call @narrowing_cast_f322f16_no_valid_cast_producer_fused_0(
// CHECK: return %[[NFP16_FUSED]] : tensor<64xf16>

module {
  func.func @narrowing_cast_f322f16_no_valid_cast_producer(%arg0 : tensor<64x128xf32>, %arg1 : tensor<64x128xf16>) -> tensor<64xf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst_0 = arith.constant 0.0 : f16
    %cst_1 = arith.constant 0.0 : f16
    %0 = tensor.empty() : tensor<64x128xf16>
    %1 = tensor.empty() : tensor<64x128xf16>
    %2 = tensor.empty() : tensor<64x128xi1>
    %3 = tensor.empty() : tensor<64xf16>
    %4 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<64x128xf32>) outs(%0 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %5 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%4, %cst_0 : tensor<64x128xf16>, f16) outs(%2 : tensor<64x128xi1>) -> tensor<64x128xi1>
    %6 = linalg.select ins(%5, %cst_1, %arg1 : tensor<64x128xi1>, f16, tensor<64x128xf16>) outs(%1 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %reduced = linalg.reduce ins(%6 : tensor<64x128xf16>) outs(%3 : tensor<64xf16>) dimensions = [1]
      (%in: f16, %init: f16) {
        %131 = arith.maximumf %in, %init : f16
        linalg.yield %131 : f16
      }
    return %reduced : tensor<64xf16>
  }
}

// Narrowing f32 -> bf16 without a fusible producer before cast.
// CHECK-LABEL: func.func private @narrowing_cast_f322bf16_no_valid_cast_producer_fused_0(
// CHECK: %[[NFPBF_CAST:.*]] = hfusion.cast {{.*}} ins(%arg0 : tensor<64x128xf32>) {{.*}} -> tensor<64x128xbf16>
// CHECK: %[[NFPBF_CMP:.*]] = hfusion.compare {{.*}} ins(%[[NFPBF_CAST]], %cst : tensor<64x128xbf16>, bf16)
// CHECK: %[[NFPBF_SELECT:.*]] = linalg.select ins(%[[NFPBF_CMP]], %cst, %arg4 : tensor<64x128xi1>, bf16, tensor<64x128xbf16>)
// CHECK: %[[NFPBF_RED:.*]] = linalg.reduce ins(%[[NFPBF_SELECT]] : tensor<64x128xbf16>)
// CHECK: return %[[NFPBF_RED]] : tensor<64xbf16>
// CHECK-LABEL: func.func @narrowing_cast_f322bf16_no_valid_cast_producer(
// CHECK: %[[NFPBF_FUSED:.*]] = call @narrowing_cast_f322bf16_no_valid_cast_producer_fused_0(
// CHECK: return %[[NFPBF_FUSED]] : tensor<64xbf16>

module {
  func.func @narrowing_cast_f322bf16_no_valid_cast_producer(%arg0 : tensor<64x128xf32>, %arg1 : tensor<64x128xbf16>) -> tensor<64xbf16> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst_0 = arith.constant 0.0 : bf16
    %cst_1 = arith.constant 0.0 : bf16
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = tensor.empty() : tensor<64x128xbf16>
    %2 = tensor.empty() : tensor<64x128xi1>
    %3 = tensor.empty() : tensor<64xbf16>
    %4 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<64x128xf32>) outs(%0 : tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %5 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%4, %cst_0 : tensor<64x128xbf16>, bf16) outs(%2 : tensor<64x128xi1>) -> tensor<64x128xi1>
    %6 = linalg.select ins(%5, %cst_1, %arg1 : tensor<64x128xi1>, bf16, tensor<64x128xbf16>) outs(%1 : tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %reduced = linalg.reduce ins(%6 : tensor<64x128xbf16>) outs(%3 : tensor<64xbf16>) dimensions = [1]
      (%in: bf16, %init: bf16) {
        %131 = arith.maximumf %in, %init : bf16
        linalg.yield %131 : bf16
      }
    return %reduced : tensor<64xbf16>
  }
}
