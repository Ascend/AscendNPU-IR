// REQUIRES: regbase
// TODO: enable this testcase after migrating hivm.hir.anchor (AnchorOp) to
// A3's HIVM dialect
// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @calc_cube_vector_mix_aiv
// linalg.map should be vectorized into vector operations

// CHECK: arith.andi
func.func @calc_cube_vector_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  hivm.hir.anchor {id = 0 : i64}
  hivm.hir.set_ctrl false at ctrl[60]
  hivm.hir.set_ctrl true at ctrl[48]
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  hivm.hir.anchor {id = 1 : i64}
  annotation.mark %1 {logical_block_num} : i32
  hivm.hir.anchor {id = 2 : i64}
  hivm.hir.anchor {id = 3 : i64}
  hivm.hir.anchor {id = 4 : i64}
  hivm.hir.anchor {id = 5 : i64}
  hivm.hir.anchor {id = 6 : i64}
  hivm.hir.anchor {id = 7 : i64}
  hivm.hir.anchor {id = 8 : i64}
  hivm.hir.anchor {id = 9 : i64}
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  hivm.hir.anchor {id = 10 : i64}
  %alloc = memref.alloc() : memref<1xi8>
  hivm.hir.anchor {id = 11 : i64}
  hivm.hir.load ins(%reinterpret_cast : memref<1xi8, strided<[1]>>) outs(%alloc : memref<1xi8>) eviction_policy = <EvictFirst> core_type = <VECTOR>
  hivm.hir.anchor {id = 12 : i64}
  %2 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  hivm.hir.anchor {id = 13 : i64}
  %alloc_1 = memref.alloc() : memref<1xi8>
  hivm.hir.anchor {id = 14 : i64}
  hivm.hir.load ins(%reinterpret_cast_0 : memref<1xi8, strided<[1]>>) outs(%alloc_1 : memref<1xi8>) eviction_policy = <EvictFirst> core_type = <VECTOR>
  hivm.hir.anchor {id = 15 : i64}
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<1xi8>
  hivm.hir.anchor {id = 16 : i64}
  %4 = tensor.empty() : tensor<1xi8>
  %mapped = linalg.map { arith.andi } ins(%2, %3 : tensor<1xi8>, tensor<1xi8>) outs(%4 : tensor<1xi8>)
  %reinterpret_cast_2 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  hivm.hir.anchor {id = 17 : i64}
  %5 = hivm.hir.get_sub_block_idx -> i64
  %6 = arith.index_cast %5 : i64 to index
  %7 = arith.cmpi eq, %6, %c0 : index
  scf.if %7 {
    hivm.hir.store ins(%mapped : tensor<1xi8>) outs(%reinterpret_cast_2 : memref<1xi8, strided<[1]>>)
  } {limit_sub_block_id0}
  hivm.hir.set_ctrl true at ctrl[60]
  hivm.hir.anchor {id = 18 : i64}
  return
}
