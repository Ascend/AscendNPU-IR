// RUN: bishengir-opt %s --vf-fusion="fusion-mode=ub-aware-op enable-outline-memref=true" | FileCheck %s
//
// canReuseInputForOutput must check the specific init tied to each output
// result, not any_of all inits.  op2 has 2 outputs: result#0 is tied to
// %real_data (not tensor.empty, no reuse) and result#1 to %empty2 (dead).
// Without the fix, any_of sees %empty2 and wrongly grants reuse for result#0,
// underestimating the merged cost (384B vs correct 512B > 448B budget).

#map = affine_map<(d0) -> (d0)>

// UB_SIZE = 3584 bits = 448 bytes.  tensor<32xf32> = 128 bytes.
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 1 : i32>, #dlti.dl_entry<"UB_SIZE", 3584 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {

// op2 (2-output mulf+subf generic) must remain inline — not merged with op1.
// CHECK-LABEL: func.func @test_no_reuse_wrong_init(
// CHECK:         linalg.generic
// CHECK:           arith.mulf
// CHECK:           arith.subf
func.func @test_no_reuse_wrong_init(
    %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
    %in1_mr: memref<32xf32>, %in2_mr: memref<32xf32>,
    %rd_mr: memref<32xf32>,
    %out_mr: memref<32xf32>
) attributes {
    SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
    hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
    mix_mode = "aiv", parallel_mode = "simd"
} {
    %alloc_in1 = memref.alloc() : memref<32xf32>
    memref.copy %in1_mr, %alloc_in1 : memref<32xf32> to memref<32xf32>
    %input1 = bufferization.to_tensor %alloc_in1 restrict writable : memref<32xf32>

    %alloc_in2 = memref.alloc() : memref<32xf32>
    memref.copy %in2_mr, %alloc_in2 : memref<32xf32> to memref<32xf32>
    %input2 = bufferization.to_tensor %alloc_in2 restrict writable : memref<32xf32>

    %alloc_rd = memref.alloc() : memref<32xf32>
    memref.copy %rd_mr, %alloc_rd : memref<32xf32> to memref<32xf32>
    %real_data = bufferization.to_tensor %alloc_rd restrict writable : memref<32xf32>

    // op1: t1 = input1 + input2
    %empty1 = tensor.empty() : tensor<32xf32>
    %t1 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%input1, %input2 : tensor<32xf32>, tensor<32xf32>)
      outs(%empty1 : tensor<32xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %v = arith.addf %in0, %in1 : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>

    // op2: 2-output generic
    //   result#0 = t1 * real_data  (tied to %real_data — NOT tensor.empty)
    //   result#1 = t1 - t1         (tied to %empty2  — dead)
    %empty2 = tensor.empty() : tensor<32xf32>
    %r:2 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]
    } ins(%t1 : tensor<32xf32>)
      outs(%real_data, %empty2 : tensor<32xf32>, tensor<32xf32>) {
    ^bb0(%in: f32, %o0: f32, %o1: f32):
      %v0 = arith.mulf %in, %o0 : f32
      %v1 = arith.subf %in, %in : f32
      linalg.yield %v0, %v1 : f32, f32
    } -> (tensor<32xf32>, tensor<32xf32>)

    bufferization.materialize_in_destination %r#0 in writable %out_mr
        : (tensor<32xf32>, memref<32xf32>) -> ()
    return
}

} // module
