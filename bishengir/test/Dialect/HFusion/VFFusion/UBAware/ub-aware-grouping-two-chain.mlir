// RUN: bishengir-opt %s --vf-fusion="fusion-mode=ub-aware-op enable-outline-memref=true" | FileCheck %s
//
// Two independent chains (a*b, c*d). Merging all 4 inputs + 2 outputs would
// exceed UB budget (768B > 512B), so the analyzer keeps them as 2 separate VFs.

#map = affine_map<(d0) -> (d0)>

// UB_SIZE = 4096 bits = 512 bytes.  tensor<32xf32> = 128 bytes.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"UB_SIZE", 4096 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {

// Each chain becomes its own fused function with a mulf inside.
// CHECK: func.func private @test_two_chain_fused_0
// CHECK:   arith.mulf
// CHECK: func.func private @test_two_chain_fused_1
// CHECK:   arith.mulf
// CHECK-LABEL: func.func @test_two_chain(
// CHECK:         call @test_two_chain_fused_
// CHECK:         call @test_two_chain_fused_
func.func @test_two_chain(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %a_mr: memref<32xf32>, %b_mr: memref<32xf32>,
    %c_mr: memref<32xf32>, %d_mr: memref<32xf32>,
    %out1_mr: memref<32xf32>, %out2_mr: memref<32xf32>
) attributes {
    SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
    hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
    mix_mode = "aiv", parallel_mode = "simd"
} {
    %alloc0 = memref.alloc() : memref<32xf32>
    memref.copy %a_mr, %alloc0 : memref<32xf32> to memref<32xf32>
    %a = bufferization.to_tensor %alloc0 restrict writable : memref<32xf32>

    %alloc1 = memref.alloc() : memref<32xf32>
    memref.copy %b_mr, %alloc1 : memref<32xf32> to memref<32xf32>
    %b = bufferization.to_tensor %alloc1 restrict writable : memref<32xf32>

    %alloc2 = memref.alloc() : memref<32xf32>
    memref.copy %c_mr, %alloc2 : memref<32xf32> to memref<32xf32>
    %c = bufferization.to_tensor %alloc2 restrict writable : memref<32xf32>

    %alloc3 = memref.alloc() : memref<32xf32>
    memref.copy %d_mr, %alloc3 : memref<32xf32> to memref<32xf32>
    %d = bufferization.to_tensor %alloc3 restrict writable : memref<32xf32>

    %empty1 = tensor.empty() : tensor<32xf32>
    %chain1 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%a, %b : tensor<32xf32>, tensor<32xf32>)
      outs(%empty1 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %prod = arith.mulf %in0, %in1 : f32
      linalg.yield %prod : f32
    } -> tensor<32xf32>

    %empty2 = tensor.empty() : tensor<32xf32>
    %chain2 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%c, %d : tensor<32xf32>, tensor<32xf32>)
      outs(%empty2 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %prod = arith.mulf %in0, %in1 : f32
      linalg.yield %prod : f32
    } -> tensor<32xf32>

    bufferization.materialize_in_destination %chain1 in writable %out1_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    bufferization.materialize_in_destination %chain2 in writable %out2_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    return
}

} // module
