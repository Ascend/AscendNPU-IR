// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 | FileCheck %s

// CHECK-LABEL: func @test_hfusion_indirect_load(
func.func @test_hfusion_indirect_load(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32>) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c1_i8 = arith.constant 1 : i8
  %c2_i8 = arith.constant 2 : i8
  %empty_0 = tensor.empty() : tensor<2x32xi8>
  %empty_1 = tensor.empty() : tensor<2x32xi8>
  %0 = linalg.fill ins(%c1_i8 : i8) outs(%empty_0 : tensor<2x32xi8>) -> tensor<2x32xi8>
  %1 = linalg.fill ins(%c2_i8 : i8) outs(%empty_1 : tensor<2x32xi8>) -> tensor<2x32xi8>
  %empty_2 = tensor.empty() : tensor<2x32xi8>
  %2 = linalg.fill ins(%c1_i8 : i8) outs(%empty_2 : tensor<2x32xi8>) -> tensor<2x32xi8>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %1 : tensor<2x32xi8>, tensor<2x32xi8>) outs(%2 : tensor<2x32xi8>) -> tensor<2x32xi8>
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %empty_4 = tensor.empty() : tensor<2x32xi64>
  %empty_5 = tensor.empty() : tensor<2x32xi64>
  %4 = linalg.fill ins(%c0_i64 : i64) outs(%empty_4 : tensor<2x32xi64>) -> tensor<2x32xi64>  
  %5 = linalg.fill ins(%c1_i64 : i64) outs(%empty_5 : tensor<2x32xi64>) -> tensor<2x32xi64>  
  %empty_6 = tensor.empty() : tensor<2x32xi64>
  %6 = linalg.fill ins(%c0_i64 : i64) outs(%empty_6 : tensor<2x32xi64>) -> tensor<2x32xi64>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<2x32xi64>, tensor<2x32xi64>) outs(%6 : tensor<2x32xi64>) -> tensor<2x32xi64>
  %c0_f32 = arith.constant 0.0 : f32
  %c1_f32 = arith.constant 1.0 : f32
  %empty_8 = tensor.empty() : tensor<2x32xf32>
  %empty_9 = tensor.empty() : tensor<2x32xf32>
  %8 = linalg.fill ins(%c0_f32 : f32) outs(%empty_8 : tensor<2x32xf32>) -> tensor<2x32xf32>  
  %9 = linalg.fill ins(%c0_f32 : f32) outs(%empty_9 : tensor<2x32xf32>) -> tensor<2x32xf32> 
  %10 = hfusion.indirect_load ins(%arg3 : memref<?xf32>, %7 : tensor<2x32xi64>, %3 : tensor<2x32xi8>, %8 : tensor<2x32xf32>) outs(%9 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %empty_11 = tensor.empty() : tensor<2x32xf32>
  %empty_12 = tensor.empty() : tensor<2x32xf32>
  %11 = linalg.fill ins(%c1_f32 : f32) outs(%empty_11 : tensor<2x32xf32>) -> tensor<2x32xf32>  
  %12 = linalg.fill ins(%c0_f32 : f32) outs(%empty_12 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %11 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%12 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %14 = memref.alloc() : memref<2x32xf32>
  bufferization.materialize_in_destination %13 in writable %14 : (tensor<2x32xf32>, memref<2x32xf32>) -> ()
  return
}

// CHECK-LABEL: @triton_add_outlined_vf_0
// CHECK: %[[res:.*]] = scf.for
// CHECK: vector<128xi8> to vector<128xf16>
// CHECK: %[[vector:.*]] = arith.addf %{{[0-9]+}}, %{{[0-9]+}} : vector<128xf16>
// CHECK: vector<128xf16> to vector<128xi8>
// return: %[[res]]
#map = affine_map<(d0) -> (d0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @triton_add(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = arith.addi %arg10, %arg11 : i32
    %1 = arith.addi %0, %arg12 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%2], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1xi8>
    memref.copy %reinterpret_cast, %alloc : memref<1xi8, strided<[1], offset: ?>> to memref<1xi8>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%2], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1xi8, strided<[1], offset: ?>> to memref<1xi8>
    %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<1xi8>
    %5 = tensor.empty() : tensor<1xi8>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : tensor<1xi8>, tensor<1xi8>) outs(%5 : tensor<1xi8>) {
    ^bb0(%in: i8, %in_3: i8, %out: i8):
      %7 = arith.sitofp %in_3 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : i8 to f16
      %8 = arith.sitofp %in {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : i8 to f16
      %9 = arith.addf %8, %7 : f16
      %10 = arith.fptosi %9 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} : f16 to i8
      linalg.yield %10 : i8
    } -> tensor<1xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
    return
  }
}

// CHECK-LABEL: func @test_hfusion_indirect_store(
func.func @test_hfusion_indirect_store(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<4x32xf32>) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c1_f32 = arith.constant 1.0 : f32
  %c2_f32 = arith.constant 2.0 : f32
  %empty_0 = tensor.empty() : tensor<4x32xf32>
  %empty_1 = tensor.empty() : tensor<4x32xf32>
  %0 = linalg.fill ins(%c1_f32 : f32) outs(%empty_0 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %1 = linalg.fill ins(%c2_f32 : f32) outs(%empty_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %empty_2 = tensor.empty() : tensor<4x32xf32>
  %2 = linalg.fill ins(%c1_f32 : f32) outs(%empty_2 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %1 : tensor<4x32xf32>, tensor<4x32xf32>) outs(%2 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty_4 = tensor.empty() : tensor<4x32xi32>
  %empty_5 = tensor.empty() : tensor<4x32xi32>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%empty_4 : tensor<4x32xi32>) -> tensor<4x32xi32>  
  %5 = linalg.fill ins(%c1_i32 : i32) outs(%empty_5 : tensor<4x32xi32>) -> tensor<4x32xi32>  
  %empty_6 = tensor.empty() : tensor<4x32xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%empty_6 : tensor<4x32xi32>) -> tensor<4x32xi32>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<4x32xi32>, tensor<4x32xi32>) outs(%6 : tensor<4x32xi32>) -> tensor<4x32xi32>
  hfusion.indirect_store ins(%3 : tensor<4x32xf32>, %7 : tensor<4x32xi32>) outs(%arg3 : memref<4x32xf32>)     
  return
}

// CHECK-LABEL: func @test_hfusion_index_put(
func.func @test_hfusion_index_put(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<4x32xf32>) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c0_i64 = arith.constant 0 : i64
  %c2_i64 = arith.constant 2 : i64
  %c4_i64 = arith.constant 4 : i64
  %c1_f32 = arith.constant 1.0 : f32
  %c2_f32 = arith.constant 2.0 : f32
  %empty_0 = tensor.empty() : tensor<2x2xf32>
  %empty_1 = tensor.empty() : tensor<2x2xf32>
  %0 = linalg.fill ins(%c1_f32 : f32) outs(%empty_0 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = linalg.fill ins(%c2_f32 : f32) outs(%empty_1 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %empty_2 = tensor.empty() : tensor<2x2xf32>
  %2 = linalg.fill ins(%c1_f32 : f32) outs(%empty_2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<2x2xf32>) -> tensor<2x2xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty_4 = tensor.empty() : tensor<2xi64>
  %empty_5 = tensor.empty() : tensor<2xi64>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%empty_4 : tensor<2xi64>) -> tensor<2xi64>  
  %5 = linalg.fill ins(%c1_i32 : i32) outs(%empty_5 : tensor<2xi64>) -> tensor<2xi64>  
  %empty_6 = tensor.empty() : tensor<2xi64>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%empty_6 : tensor<2xi64>) -> tensor<2xi64>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<2xi64>, tensor<2xi64>) outs(%6 : tensor<2xi64>) -> tensor<2xi64>
  hfusion.index_put ins(%arg3 : memref<4x32xf32>, %7 : tensor<2xi64>, %3 : tensor<2x2xf32>, %c0_i32 : i32, %c2_i64 : i64, [%c4_i64, %c2_i64 : i64, i64], [%c0_i64, %c0_i64 : i64, i64], [%c0_i64, %c0_i64 : i64, i64])
  return
}

// CHECK-LABEL: func @test_hfusion_scatterT(
func.func @test_hfusion_scatterT(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<4x32xf32>) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c1_f32 = arith.constant 1.0 : f32
  %c2_f32 = arith.constant 2.0 : f32
  %empty_0 = tensor.empty() : tensor<4x32xf32>
  %empty_1 = tensor.empty() : tensor<4x32xf32>
  %0 = linalg.fill ins(%c1_f32 : f32) outs(%empty_0 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %1 = linalg.fill ins(%c2_f32 : f32) outs(%empty_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %empty_2 = tensor.empty() : tensor<4x32xf32>
  %2 = linalg.fill ins(%c1_f32 : f32) outs(%empty_2 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%0, %1 : tensor<4x32xf32>, tensor<4x32xf32>) outs(%2 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty_4 = tensor.empty() : tensor<4x32xi32>
  %empty_5 = tensor.empty() : tensor<4x32xi32>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%empty_4 : tensor<4x32xi32>) -> tensor<4x32xi32>  
  %5 = linalg.fill ins(%c1_i32 : i32) outs(%empty_5 : tensor<4x32xi32>) -> tensor<4x32xi32>  
  %empty_6 = tensor.empty() : tensor<4x32xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%empty_6 : tensor<4x32xi32>) -> tensor<4x32xi32>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<4x32xi32>, tensor<4x32xi32>) outs(%6 : tensor<4x32xi32>) -> tensor<4x32xi32>
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  hfusion.scatterT ins(%arg3 : memref<4x32xf32>, %3 : tensor<4x32xf32>, %7 : tensor<4x32xi32>, %c32_i32 : i32, %c1_i32 : i32, [%c32_i32, %c1_i32 : i32, i32], [%c4_i32, %c32_i32 : i32, i32], [%c0_i32, %c0_i32 : i32, i32])
  return 
}

// CHECK-LABEL: func @test_hfusion_embedding_gather(
func.func @test_hfusion_embedding_gather(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32>) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty_4 = tensor.empty() : tensor<16x400xi32>
  %empty_5 = tensor.empty() : tensor<16x400xi32>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%empty_4 : tensor<16x400xi32>) -> tensor<16x400xi32>  
  %5 = linalg.fill ins(%c1_i32 : i32) outs(%empty_5 : tensor<16x400xi32>) -> tensor<16x400xi32>  
  %empty_6 = tensor.empty() : tensor<16x400xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%empty_6 : tensor<16x400xi32>) -> tensor<16x400xi32>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<16x400xi32>, tensor<16x400xi32>) outs(%6 : tensor<16x400xi32>) -> tensor<16x400xi32>
  %c32_i32 = arith.constant 32 : i32
  %c4000_i32 = arith.constant 4000 : i32
  %c9000_i32 = arith.constant 9000 : i32
  %0 = tensor.empty() : tensor<16x400x32xf32>
  %1 = hfusion.embedding_gather ins(%arg3 : memref<?xf32>, %7 : tensor<16x400xi32>, %c9000_i32 : i32, [%c0_i32, %c0_i32, %c0_i32 : i32, i32, i32], [%c32_i32, %c4000_i32, %c32_i32 : i32, i32, i32]) outs(%0 : tensor<16x400x32xf32>) -> tensor<16x400x32xf32>
  return
}

// CHECK-LABEL: func @test_hfusion_gatherT(
func.func @test_hfusion_gatherT(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32>) attributes {DirectlyUsedGMArgIdxList = [3], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %empty_4 = tensor.empty() : tensor<2x2xi64>
  %empty_5 = tensor.empty() : tensor<2x2xi64>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%empty_4 : tensor<2x2xi64>) -> tensor<2x2xi64>  
  %5 = linalg.fill ins(%c1_i32 : i32) outs(%empty_5 : tensor<2x2xi64>) -> tensor<2x2xi64>  
  %empty_6 = tensor.empty() : tensor<2x2xi64>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%empty_6 : tensor<2x2xi64>) -> tensor<2x2xi64>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %5 : tensor<2x2xi64>, tensor<2x2xi64>) outs(%6 : tensor<2x2xi64>) -> tensor<2x2xi64>
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
  %0 = hfusion.gatherT ins(%arg3 : memref<?xf32>, %7 : tensor<2x2xi64>, %c4_i64 : i64, %c0_i32 : i32, [%c2_i64, %c1_i64 : i64, i64], [%c2_i32, %c2_i32 : i32, i32], [%c0_i32, %c0_i32 : i32, i32]) outs(%cst : tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}