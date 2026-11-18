// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" 2>&1 | FileCheck %s --implicit-check-not="AutoVectorizeV2 failed" --implicit-check-not="Attempted to vectorize, but failed"

// Original payload from seq.mlir.before. Each scan stage uses its producer
// both as a DPS input and through tensor.extract inside its scalar region.
// Do not clone the full producer into that region when fusing the DPS use.
// Exercise MCF: without the fix, this can assert or fail and retry with MCF
// disabled. Disabling MCF from the outset does not exercise the faulty path.
// Reject failure diagnostics throughout the output, including before the
// function label, so a successful fallback cannot hide the failed MCF attempt.
// CHECK-LABEL: func.func private @moe_fused_index_compute_kernel_fused_7
// CHECK: scf.for

#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950">, ssbuffer.insertionOptimization} {
  func.func private @moe_fused_index_compute_kernel_fused_7(%arg0: index, %arg1: index, %arg2: tensor<128xi32>, %arg3: i32, %arg4: tensor<128xi32>, %arg5: i32, %arg6: i32, %arg7: tensor<128xi1>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) -> tensor<128xi32> attributes {DirectlyUsedGMArgIdxList = [4, 4, 4, 4], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "mix_simd_simt", transform.target_tag = "auto_vectorize_v2.payload.moe_fused_index_compute_kernel_fused_7"} {
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %c15_i32 = arith.constant 15 : i32
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c64_i32 = arith.constant 64 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c-16_i32 = arith.constant -16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c-8_i32 = arith.constant -8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-4_i32 = arith.constant -4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-2_i32 = arith.constant -2 : i32
    %0 = tensor.empty() : tensor<128xi32>
    %1 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-1"} {
    ^bb0(%out: i32):
      %9 = linalg.index 0 : index
      %10 = arith.index_cast %9 : index to i32
      linalg.yield %10 : i32
    } -> tensor<128xi32>
    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg4, %1 : tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-2"} {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %9 = arith.andi %in_0, %c-2_i32 : i32
      %10 = arith.index_castui %9 : i32 to index
      %extracted = tensor.extract %arg4[%10] : tensor<128xi32>
      %11 = arith.andi %in_0, %c1_i32 : i32
      %12 = arith.cmpi ne, %11, %c0_i32 : i32
      %13 = arith.select %12, %extracted, %c0_i32 : i32
      %14 = arith.addi %in, %13 : i32
      linalg.yield %14 : i32
    } -> tensor<128xi32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%2, %1 : tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-3"} {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %9 = arith.andi %in_0, %c-4_i32 : i32
      %10 = arith.addi %9, %c1_i32 : i32
      %11 = arith.index_castui %10 : i32 to index
      %extracted = tensor.extract %2[%11] : tensor<128xi32>
      %12 = arith.andi %in_0, %c2_i32 : i32
      %13 = arith.cmpi ne, %12, %c0_i32 : i32
      %14 = arith.select %13, %extracted, %c0_i32 : i32
      %15 = arith.addi %in, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<128xi32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%3, %1 : tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-4"} {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %9 = arith.andi %in_0, %c-8_i32 : i32
      %10 = arith.addi %9, %c3_i32 : i32
      %11 = arith.index_castui %10 : i32 to index
      %extracted = tensor.extract %3[%11] : tensor<128xi32>
      %12 = arith.andi %in_0, %c4_i32 : i32
      %13 = arith.cmpi ne, %12, %c0_i32 : i32
      %14 = arith.select %13, %extracted, %c0_i32 : i32
      %15 = arith.addi %in, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<128xi32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%4, %1 : tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-5"} {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %9 = arith.andi %in_0, %c-16_i32 : i32
      %10 = arith.addi %9, %c7_i32 : i32
      %11 = arith.index_castui %10 : i32 to index
      %extracted = tensor.extract %4[%11] : tensor<128xi32>
      %12 = arith.andi %in_0, %c8_i32 : i32
      %13 = arith.cmpi ne, %12, %c0_i32 : i32
      %14 = arith.select %13, %extracted, %c0_i32 : i32
      %15 = arith.addi %in, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<128xi32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%5, %1 : tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-6"} {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %9 = arith.andi %in_0, %c-32_i32 : i32
      %10 = arith.addi %9, %c15_i32 : i32
      %11 = arith.index_castui %10 : i32 to index
      %extracted = tensor.extract %5[%11] : tensor<128xi32>
      %12 = arith.andi %in_0, %c16_i32 : i32
      %13 = arith.cmpi ne, %12, %c0_i32 : i32
      %14 = arith.select %13, %extracted, %c0_i32 : i32
      %15 = arith.addi %in, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<128xi32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%6, %1 : tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-7"} {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %9 = arith.andi %in_0, %c-64_i32 : i32
      %10 = arith.addi %9, %c31_i32 : i32
      %11 = arith.index_castui %10 : i32 to index
      %extracted = tensor.extract %6[%11] : tensor<128xi32>
      %12 = arith.andi %in_0, %c32_i32 : i32
      %13 = arith.cmpi ne, %12, %c0_i32 : i32
      %14 = arith.select %13, %extracted, %c0_i32 : i32
      %15 = arith.addi %in, %14 : i32
      linalg.yield %15 : i32
    } -> tensor<128xi32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%7, %1, %arg4 : tensor<128xi32>, tensor<128xi32>, tensor<128xi32>) outs(%0 : tensor<128xi32>) attrs =  {"hfusion-auto-vectorize-target-8"} {
    ^bb0(%in: i32, %in_0: i32, %in_1: i32, %out: i32):
      %9 = arith.andi %in_0, %c-128_i32 : i32
      %10 = arith.addi %9, %c63_i32 : i32
      %11 = arith.index_castui %10 : i32 to index
      %extracted = tensor.extract %7[%11] : tensor<128xi32>
      %12 = arith.andi %in_0, %c64_i32 : i32
      %13 = arith.cmpi ne, %12, %c0_i32 : i32
      %14 = arith.select %13, %extracted, %c0_i32 : i32
      %15 = arith.addi %in, %14 : i32
      %16 = arith.subi %15, %in_1 : i32
      linalg.yield %16 : i32
    } -> tensor<128xi32>
    return %8 : tensor<128xi32>
  }
}
