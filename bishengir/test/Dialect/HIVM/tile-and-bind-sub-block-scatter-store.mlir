// RUN: bishengir-opt %s --mark-simt-scope-no-inline --hivm-bind-sub-block -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @scatter_only_aiv(
// CHECK:           %[[IDX:.*]] = hivm.hir.get_sub_block_idx -> i64
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[IDX]] : i64 to index
// CHECK:           %[[COND:.*]] = arith.cmpi eq, %[[CAST]], %{{.*}} : index
// CHECK:           scf.if %[[COND]] {
// CHECK:             hivm.hir.scatter_store
// CHECK:           } {limit_sub_block_id0}
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @scatter_only_aic(%arg0: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    return
  }
  func.func @scatter_only_aiv(%base: memref<?xf32>, %idx_gm: memref<?xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1_i32 = arith.constant 1 : i32
    %reinterpret_cast = memref.reinterpret_cast %idx_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>)
    %0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.vabs ins(%1 : tensor<8xf32>) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%0 : tensor<8xi64>, %2 : tensor<8xf32>, %c1_i32 : i32) outs(%base : memref<?xf32>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @tiled_store_with_scatter_aiv(
// CHECK:           scf.for
// CHECK:             hivm.hir.store {{.*}} {tiled_op}
// CHECK:             hivm.hir.get_sub_block_idx -> i64
// CHECK:             scf.if
// CHECK:               hivm.hir.scatter_store
// CHECK:             } {limit_sub_block_id0}
// CHECK:           } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @tiled_store_with_scatter_aic(%arg0: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    return
  }
  func.func @tiled_store_with_scatter_aiv(%src: memref<?xf32>, %dst: memref<?xf32>, %base: memref<?xf32>, %idx_gm: memref<?xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1_i32 = arith.constant 1 : i32
    %reinterpret_cast = memref.reinterpret_cast %src to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %dst to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    %0 = tensor.empty() : tensor<64x64xf32>
    %alloc = memref.alloc() : memref<64x64xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<64x64xf32, strided<[64, 1]>>) outs(%alloc : memref<64x64xf32>)
    %1 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %2 = hivm.hir.vabs ins(%1 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    hivm.hir.store ins(%2 : tensor<64x64xf32>) outs(%reinterpret_cast_0 : memref<64x64xf32, strided<[64, 1]>>)
    %reinterpret_cast_1 = memref.reinterpret_cast %idx_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc_2 = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast_1 : memref<8xi64, strided<[1]>>) outs(%alloc_2 : memref<8xi64>)
    %3 = bufferization.to_tensor %alloc_2 restrict writable : memref<8xi64>
    %4 = tensor.empty() : tensor<8xf32>
    %5 = hivm.hir.vabs ins(%4 : tensor<8xf32>) outs(%4 : tensor<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%3 : tensor<8xi64>, %5 : tensor<8xf32>, %c1_i32 : i32) outs(%base : memref<?xf32>)
    return
  }
}

// -----

// CHECK-LABEL:   func.func @scatter_in_simt_scope_aiv(
// CHECK:           scope.scope
// CHECK:             hivm.hir.gather_load
// CHECK-NOT:         scf.if
// CHECK:             hivm.hir.local_store
// CHECK:             hivm.hir.get_sub_block_idx -> i64
// CHECK:             scf.if
// CHECK:               hivm.hir.scatter_store
// CHECK:             } {limit_sub_block_id0}
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @scatter_in_simt_scope_aic(%arg0: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    return
  }
  func.func @scatter_in_simt_scope_aiv(%base: memref<?xf32>, %ub_dst: memref<8xf32>, %idx_gm: memref<?xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1_i32 = arith.constant 1 : i32
    %reinterpret_cast = memref.reinterpret_cast %idx_gm to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %alloc = memref.alloc() : memref<8xi64>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%alloc : memref<8xi64>)
    scope.scope : () -> () {
      %0 = bufferization.to_tensor %alloc restrict writable : memref<8xi64>
      %1 = tensor.empty() : tensor<8xf32>
      %2 = hivm.hir.gather_load ins(%base : memref<?xf32>, %0 : tensor<8xi64>, %c1_i32 : i32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
      hivm.hir.local_store ins(%ub_dst : memref<8xf32>, %2 : tensor<8xf32>)
      hivm.hir.scatter_store ins(%0 : tensor<8xi64>, %2 : tensor<8xf32>, %c1_i32 : i32) outs(%base : memref<?xf32>)
      scope.return
    } {vector_mode = "simt"}
    return
  }
}
