// RUN: bishengir-opt %s --adapt-triton-kernel | FileCheck %s

// CHECK-LABEL: @indirect_load_adapt
// CHECK: %[[extui_1:.*]] = arith.extui %{{[0-9]+}} : tensor<16x32xi1> to tensor<16x32xi8>
// CHECK: %[[dst_1:.*]] = tensor.empty() : tensor<16x32xf16>
// CHECK: %[[iload_1:.*]] = hfusion.indirect_load ins(%arg3 : memref<?xf16>, %{{[0-9]+}} : tensor<16x32xi64>, %[[extui_1]] : tensor<16x32xi8>, %{{[0-9]+}} : tensor<16x32xf16>) outs(%[[dst_1]] : tensor<16x32xf16>) -> tensor<16x32xf16>
// CHECK: hivm.hir.copy ins(%[[iload_1]] : tensor<16x32xf16>)
// CHECK: %[[extui_2:.*]] = arith.extui %{{[0-9]+}} : tensor<16x32xi1> to tensor<16x32xi8>
// CHECK: %[[dst_2:.*]] = tensor.empty() : tensor<16x32xf16>
// CHECK: %[[iload_2:.*]] = hfusion.indirect_load ins(%arg3 : memref<?xf16>, %{{[0-9]+}} : tensor<16x32xi64>, %[[extui_2]] : tensor<16x32xi8>, %{{[0-9]+}} : tensor<16x32xf16>) outs(%[[dst_2]] : tensor<16x32xf16>) {isVolatile = false} -> tensor<16x32xf16>
// CHECK: hivm.hir.copy ins(%[[iload_2]] : tensor<16x32xf16>)
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func private @triton_indirect_load(memref<?xf16>, tensor<16x32xi64>, tensor<16x32xi1>, tensor<16x32xf16>) -> tensor<16x32xf16>
  func.func @indirect_load_adapt(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32) attributes {DirectlyUsedGMArgIdxList = [3, 3, 2], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "mix_simd_simt"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    scope.scope : () -> () {
      %0 = tensor.empty() : tensor<16x32xf16>
      %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<16x32xf16>) -> tensor<16x32xf16>
      %2 = tensor.empty() : tensor<16x32xi32>
      %3 = linalg.fill ins(%c32_i32 : i32) outs(%2 : tensor<16x32xi32>) -> tensor<16x32xi32>
      %4 = tensor.empty() : tensor<16x32xi1>
      %5 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%3, %3 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%4 : tensor<16x32xi1>) -> tensor<16x32xi1>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      %6 = tensor.empty() : tensor<16x32xi64>
      %7 = linalg.fill ins(%c32_i64 : i64) outs(%6 : tensor<16x32xi64>) -> tensor<16x32xi64>
      %alloc = memref.alloc() : memref<16x32xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x32xf16, #hivm.address_space<cbuf>>
      %alloc_0 = memref.alloc() : memref<16x32xf16, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_0 {ssbuffer.intraDeps = [3 : i32, 1 : i32]} : memref<16x32xf16, #hivm.address_space<ub>> to memref<16x32xf16>
      %8 = func.call @triton_indirect_load(%arg3, %7, %5, %1) : (memref<?xf16>, tensor<16x32xi64>, tensor<16x32xi1>, tensor<16x32xf16>) -> tensor<16x32xf16>
      hivm.hir.copy ins(%8 : tensor<16x32xf16>) outs(%memspacecast : memref<16x32xf16>)
      %9 = func.call @triton_indirect_load(%arg3, %7, %5, %1) {isVolatile = false, ssbuffer.clone = 12 : i32} : (memref<?xf16>, tensor<16x32xi64>, tensor<16x32xi1>, tensor<16x32xf16>) -> tensor<16x32xf16>
      hivm.hir.copy ins(%9 : tensor<16x32xf16>) outs(%alloc : memref<16x32xf16, #hivm.address_space<cbuf>>)
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.muli %arg5, %c16_i32 : i32
      %1 = arith.index_cast %0 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      %alloc = memref.alloc() : memref<16x16xf16>
      memref.copy %reinterpret_cast, %alloc : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
