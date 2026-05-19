// RUN: bishengir-opt %s -hivm-bind-sub-block -split-input-file -verify-diagnostics | FileCheck %s

// Test: odd-dimension tiling (dimSize=99, ceil(99/2)=50, tail=49)
// Verify offset = iv * 50, size = 50 - iv
// When iv=0: offset=0, size=50; when iv=1: offset=50, size=49

// CHECK-DAG: affine_map<()[s0] -> (s0 * 50)>
// CHECK-DAG: affine_map<()[s0] -> (-s0 + 50)>
// CHECK-LABEL: func.func @odd_tiling_aiv(
// CHECK: scf.for %[[IV:.+]] =
// CHECK: %[[OFFSET:.+]] = affine.apply #[[OFFSET_MAP:map[a-z0-9]*]]()[%[[IV]]]
// CHECK: %[[SIZE:.+]] = affine.apply #[[SIZE_MAP:map[a-z0-9]*]]()[%[[IV]]]
// CHECK: memref.subview {{.+}}[%[[OFFSET]]] [%[[SIZE]]] [1]
// CHECK: hivm.hir.store {{.*}} {tiled_op}

// CHECK-NOT: limit_sub_block_id0
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @odd_tiling_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [99], strides: [1] : memref<?xf16> to memref<99xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<99xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<99xf16, strided<[1]>>) outs(%alloc : memref<99xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<99xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [99], strides: [1] : memref<?xf16> to memref<99xf16, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<99xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<99xf16, strided<[1]>>) outs(%alloc_1 : memref<99xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<99xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [99], strides: [1] : memref<?xf32> to memref<99xf32, strided<[1]>>
    %4 = tensor.empty() : tensor<99xf16>
    %5 = hivm.hir.vadd ins(%2, %3 : tensor<99xf16>, tensor<99xf16>) outs(%4 : tensor<99xf16>) -> tensor<99xf16>
    %6 = tensor.empty() : tensor<99xf32>
    %7 = hivm.hir.vcast ins(%5 : tensor<99xf16>) outs(%6 : tensor<99xf32>) -> tensor<99xf32>
    hivm.hir.store ins(%7 : tensor<99xf32>) outs(%reinterpret_cast_2 : memref<99xf32, strided<[1]>>)
    return
  }
}

// -----

// Test: odd-dimension tiling (dimSize=15, ceil(15/2)=8, tail=7)
// Verify offset = iv * 8, size = 8 - iv
// When iv=0: offset=0, size=8; when iv=1: offset=8, size=7

// CHECK-DAG: affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: affine_map<()[s0] -> (-s0 + 8)>
// CHECK-LABEL: func.func @odd_tiling_15_aiv(
// CHECK: scf.for %[[IV:.+]] =
// CHECK: %[[OFFSET:.+]] = affine.apply #[[OFFSET_MAP:map[a-z0-9]*]]()[%[[IV]]]
// CHECK: %[[SIZE:.+]] = affine.apply #[[SIZE_MAP:map[a-z0-9]*]]()[%[[IV]]]
// CHECK: memref.subview {{.+}}[%[[OFFSET]]] [%[[SIZE]]] [1]
// CHECK: hivm.hir.store {{.*}} {tiled_op}
// CHECK-NOT: limit_sub_block_id0
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @odd_tiling_15_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [15], strides: [1] : memref<?xf16> to memref<15xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<15xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<15xf16, strided<[1]>>) outs(%alloc : memref<15xf16>) eviction_policy = <EvictFirst>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<15xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [15], strides: [1] : memref<?xf16> to memref<15xf16, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<15xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<15xf16, strided<[1]>>) outs(%alloc_1 : memref<15xf16>) eviction_policy = <EvictFirst>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<15xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [15], strides: [1] : memref<?xf32> to memref<15xf32, strided<[1]>>
    %4 = tensor.empty() : tensor<15xf16>
    %5 = hivm.hir.vadd ins(%2, %3 : tensor<15xf16>, tensor<15xf16>) outs(%4 : tensor<15xf16>) -> tensor<15xf16>
    %6 = tensor.empty() : tensor<15xf32>
    %7 = hivm.hir.vcast ins(%5 : tensor<15xf16>) outs(%6 : tensor<15xf32>) -> tensor<15xf32>
    hivm.hir.store ins(%7 : tensor<15xf32>) outs(%reinterpret_cast_2 : memref<15xf32, strided<[1]>>)
    return
  }
}

// -----

// Test: odd tiling with scf.if inside subblock loop.
// tensor.empty is outside scf.if but inside subblock loop.
// The extract_slice on tensor.empty is rejected by BubbleUpPattern's
// areOperandsUpperLevel because operands (affine.apply) are in deeper region
// (inner loop inside scf.if). MarkEmptySliceBufferSize should
// catch it and add the missing buffer_size_in_byte + buffer_static_shape marks.
//
// dimSize=7: ceil(7/2)=4 (tileSize), tail=3. buffer_size=1*4*4(f32)=16.
// CHECK-DAG: affine_map<()[s0] -> (s0 * 4)>
// CHECK-DAG: affine_map<()[s0] -> (-s0 + 4)>
// CHECK-LABEL: func.func @odd_tiling_if_region_aiv(
// CHECK:      scf.for %[[IV:.+]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:        %[[OFF:.+]] = affine.apply #[[OFFSET_MAP:map[a-z0-9]*]]()[%[[IV]]]
// CHECK:        %[[SIZE:.+]] = affine.apply #[[SIZE_MAP:map[a-z0-9]*]]()[%[[IV]]]
// CHECK:        %[[EMPTY:.+]] = tensor.empty(%[[SIZE]])
// CHECK:        annotation.mark %[[EMPTY]] {buffer_size_in_byte = 16 : i64} : tensor<1x?xf32>
// CHECK:        scf.if
// CHECK:          scf.for
// CHECK:            hivm.hir.vmul
// CHECK:            hivm.hir.store {{.*}} {tiled_op}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @odd_tiling_if_region_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i8 {tt.divisibility = 16 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false]> : vector<6xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 2.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.trunci %arg5 : i8 to i1
    %1 = tensor.empty() : tensor<1x7xf32>
    scf.if %0 {
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1, 7], strides: [7, 1] : memref<?xf32> to memref<1x7xf32, strided<[7, 1]>>
      scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %alloc = memref.alloc() : memref<1x7xf32>
        hivm.hir.load ins(%reinterpret_cast : memref<1x7xf32, strided<[7, 1]>>) outs(%alloc : memref<1x7xf32>) eviction_policy = <EvictFirst>
        %2 = bufferization.to_tensor %alloc restrict writable : memref<1x7xf32>
        %3 = hivm.hir.vmul ins(%2, %cst : tensor<1x7xf32>, f32) outs(%1 : tensor<1x7xf32>) -> tensor<1x7xf32>
        hivm.hir.store ins(%3 : tensor<1x7xf32>) outs(%reinterpret_cast : memref<1x7xf32, strided<[7, 1]>>)
      }
    }
    return
  }
}
