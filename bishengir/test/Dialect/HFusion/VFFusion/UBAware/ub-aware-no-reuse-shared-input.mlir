// RUN: bishengir-opt %s --vf-fusion="fusion-mode=ub-aware-op enable-outline-memref=true" | FileCheck %s
//
// canReuseInputForOutput must reject reuse when an input has consumers outside
// the merged group. Without this check, inputs shared across VF boundaries
// would be double-counted in PlanMemory.
//
// Merged {op1, op2}: 3 inputs (384B) + 1 output (128B) = 512B > budget 448B.
// Reuse would wrongly reduce this to 384B ≤ 448B → incorrect merge.

#map = affine_map<(d0) -> (d0)>

// UB_SIZE = 3584 bits = 448 bytes.  tensor<32xf32> = 128 bytes.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"UB_SIZE", 3584 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {

// op1(shared1, shared2) and op2(t1, shared3) must NOT merge: each shared input
// has an external consumer, so reuse is denied and 512B > 448B budget.
// CHECK-LABEL: func.func @test_no_reuse_shared(
// CHECK:         linalg.generic
// CHECK:           arith.addf
// CHECK:         linalg.generic
// CHECK:           arith.mulf
func.func @test_no_reuse_shared(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %s1_mr: memref<32xf32>, %s2_mr: memref<32xf32>, %s3_mr: memref<32xf32>,
    %a_mr: memref<32xf32>, %b_mr: memref<32xf32>, %c_mr: memref<32xf32>,
    %out_mr: memref<32xf32>, %out3_mr: memref<32xf32>,
    %out4_mr: memref<32xf32>, %out5_mr: memref<32xf32>
) attributes {
    SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
    hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
    mix_mode = "aiv", parallel_mode = "simd"
} {
    %alloc_s1 = memref.alloc() : memref<32xf32>
    memref.copy %s1_mr, %alloc_s1 : memref<32xf32> to memref<32xf32>
    %shared1 = bufferization.to_tensor %alloc_s1 restrict writable : memref<32xf32>

    %alloc_s2 = memref.alloc() : memref<32xf32>
    memref.copy %s2_mr, %alloc_s2 : memref<32xf32> to memref<32xf32>
    %shared2 = bufferization.to_tensor %alloc_s2 restrict writable : memref<32xf32>

    %alloc_s3 = memref.alloc() : memref<32xf32>
    memref.copy %s3_mr, %alloc_s3 : memref<32xf32> to memref<32xf32>
    %shared3 = bufferization.to_tensor %alloc_s3 restrict writable : memref<32xf32>

    %alloc_a = memref.alloc() : memref<32xf32>
    memref.copy %a_mr, %alloc_a : memref<32xf32> to memref<32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<32xf32>

    %alloc_b = memref.alloc() : memref<32xf32>
    memref.copy %b_mr, %alloc_b : memref<32xf32> to memref<32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32xf32>

    %alloc_c = memref.alloc() : memref<32xf32>
    memref.copy %c_mr, %alloc_c : memref<32xf32> to memref<32xf32>
    %c = bufferization.to_tensor %alloc_c restrict writable : memref<32xf32>

    // op1: t1 = shared1 + shared2
    %empty1 = tensor.empty() : tensor<32xf32>
    %t1 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%shared1, %shared2 : tensor<32xf32>, tensor<32xf32>)
      outs(%empty1 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %v = arith.addf %in0, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    // op2: result = t1 * shared3
    %empty2 = tensor.empty() : tensor<32xf32>
    %result = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%t1, %shared3 : tensor<32xf32>, tensor<32xf32>)
      outs(%empty2 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %v = arith.mulf %in0, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    // External consumers of shared inputs (force reuse denial)
    %empty3 = tensor.empty() : tensor<32xf32>
    %out3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%shared1, %a : tensor<32xf32>, tensor<32xf32>)
      outs(%empty3 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %v = arith.addf %in0, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    %empty4 = tensor.empty() : tensor<32xf32>
    %out4 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%shared2, %b : tensor<32xf32>, tensor<32xf32>)
      outs(%empty4 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %v = arith.mulf %in0, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    %empty5 = tensor.empty() : tensor<32xf32>
    %out5 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%shared3, %c : tensor<32xf32>, tensor<32xf32>)
      outs(%empty5 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %v = arith.subf %in0, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    bufferization.materialize_in_destination %result in writable %out_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    bufferization.materialize_in_destination %out3 in writable %out3_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    bufferization.materialize_in_destination %out4 in writable %out4_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    bufferization.materialize_in_destination %out5 in writable %out5_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    return
}

} // module
