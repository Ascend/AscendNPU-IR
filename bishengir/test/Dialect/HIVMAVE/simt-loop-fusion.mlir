// RUN: bishengir-opt %s -hivm-lower-to-loops -simt-loop-nests-scheduling -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

module {
  func.func @lanczos_resize_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: f32, %arg11: f32, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32) -> memref<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
    %0 = arith.addi %arg21, %arg20 : i32
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>
    hivm.hir.vadd ins(%alloc, %0 : memref<16x16xi32>, i32) outs(%alloc_2 : memref<16x16xi32>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
    // CHECK: scf.for
    // CHECK: scf.for
    // These two loops should not involve in fusion
    scf.for %arg22 = %c0 to %c16 step %c1 {
      scf.for %arg23 = %c0 to %c16 step %c1 {
        %1 = memref.load %alloc_0[%arg22, %arg23] : memref<16x16xi32>
        %2 = arith.index_cast %1 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %3 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>>
        memref.store %3, %alloc_1[%arg22, %arg23] : memref<16x16xf32>
      } {loop5, map_for_to_forall}
    } {loop1, map_for_to_forall}
    hivm.hir.vmul ins(%alloc_1, %alloc_1 : memref<16x16xf32>, memref<16x16xf32>) outs(%alloc_1 : memref<16x16xf32>)
    // CHECK: scf.for
    // CHECK: scf.for
    // All these loops should be fused
    scf.for %arg22 = %c0 to %c16 step %c1 {
      scf.for %arg23 = %c0 to %c16 step %c1 {
        %1 = memref.load %alloc_2[%arg22, %arg23] : memref<16x16xi32>
        %2 = arith.index_cast %1 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%2], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %3 = memref.load %reinterpret_cast[%c0] : memref<1xf32, strided<[1], offset: ?>>
        memref.store %3, %alloc_3[%arg22, %arg23] : memref<16x16xf32>
      } {loop6, map_for_to_forall}
    } {loop2, map_for_to_forall}
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
    hivm.hir.vmul {hivm.vf_mode = #hivm.vf_mode<SIMT>} ins(%alloc_1, %alloc_3 : memref<16x16xf32>, memref<16x16xf32>) outs(%alloc_4 : memref<16x16xf32>)
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
    scf.for %arg22 = %c0 to %c16 step %c1 {
      scf.for %arg23 = %c0 to %c16 step %c1 {
        %1 = memref.load %alloc_1[%arg22, %arg23] : memref<16x16xf32>
        %2 = memref.load %alloc_3[%arg22, %arg23] : memref<16x16xf32>
        %3 = arith.addf %1, %2 : f32
        memref.store %3, %alloc_5[%arg22, %arg23] : memref<16x16xf32>
      } {loop8, map_for_to_forall}
    } {loop4, map_for_to_forall}
    
    // CHECK-NOT: scf.for
    return %alloc_5 : memref<16x16xf32>
  }
}