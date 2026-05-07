// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 | FileCheck %s

// CHECK: %2:2 = scf.for %arg10 = %c0 to %c13 step %c1 iter_args(%arg11 = %1, %arg12 = %1) -> (tensor<13x151xi64>, tensor<13x151xi64>) {
// CHECK-NEXT: %3:2 = scf.for %arg13 = %c0 to %c151 step %c64 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<13x151xi64>, tensor<13x151xi64>) {
// CHECK-NEXT:   %4 = affine.min #map(%arg13)
// CHECK-NEXT:   %extracted_slice = tensor.extract_slice %0[%arg10, %arg13] [1, %4] [1, 1] : tensor<13x151xi32> to tensor<1x?xi32>
// CHECK-NEXT:   %5 = scf.for %arg16 = %c0 to %4 step %c64 iter_args(%arg17 = %extracted_slice) -> (tensor<1x?xi32>) {

#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @fn_npu_(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c13_i32 = arith.constant 13 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1057_i32 = arith.constant 1057 : i32
    %c151_i32 = arith.constant 151 : i32
    %0 = tensor.empty() : tensor<13x151xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<13x151xi32>) {
    ^bb0(%out: i32):
      %8 = linalg.index 1 : index
      %9 = arith.index_cast %8 : index to i32
      %10 = linalg.index 0 : index
      %11 = arith.index_cast %10 : index to i32
      %12 = arith.muli %11, %c151_i32 : i32
      %13 = arith.addi %12, %9 : i32
      linalg.yield %13 : i32
    } -> tensor<13x151xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<13x151xi32>) outs(%0 : tensor<13x151xi32>) {
    ^bb0(%in: i32, %out: i32):
      %8 = arith.divsi %in, %c1057_i32 : i32
      %9 = arith.muli %8, %c7_i32 : i32
      linalg.yield %9 : i32
    } -> tensor<13x151xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<13x151xi32>) outs(%0 : tensor<13x151xi32>) {
    ^bb0(%in: i32, %out: i32):
      %8 = arith.subi %c13_i32, %in : i32
      %9 = arith.minsi %8, %c7_i32 : i32
      linalg.yield %9 : i32
    } -> tensor<13x151xi32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<13x151xi32>) outs(%0 : tensor<13x151xi32>) {
    ^bb0(%in: i32, %out: i32):
      %8 = arith.remsi %in, %c1057_i32 : i32
      linalg.yield %8 : i32
    } -> tensor<13x151xi32>
    %5 = tensor.empty() : tensor<13x151xi64>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %4, %3 : tensor<13x151xi32>, tensor<13x151xi32>, tensor<13x151xi32>) outs(%5 : tensor<13x151xi64>) {
    ^bb0(%in: i32, %in_1: i32, %in_2: i32, %out: i64):
      %8 = arith.remsi %in_1, %in_2 : i32
      %9 = arith.addi %in, %8 : i32
      %10 = arith.extsi %9 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : i32 to i64
      linalg.yield %10 : i64
    } -> tensor<13x151xi64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %3 : tensor<13x151xi32>, tensor<13x151xi32>) outs(%5 : tensor<13x151xi64>) {
    ^bb0(%in: i32, %in_1: i32, %out: i64):
      %8 = arith.divsi %in, %in_1 : i32
      %9 = arith.extsi %8 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : i32 to i64
      linalg.yield %9 : i64
    } -> tensor<13x151xi64>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [13, 151], strides: [151, 1] : memref<?xi64> to memref<13x151xi64, strided<[151, 1]>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast : (tensor<13x151xi64>, memref<13x151xi64, strided<[151, 1]>>) -> ()
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13, 151], strides: [151, 1] : memref<?xi64> to memref<13x151xi64, strided<[151, 1]>>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_0 : (tensor<13x151xi64>, memref<13x151xi64, strided<[151, 1]>>) -> ()
    return
  }
}