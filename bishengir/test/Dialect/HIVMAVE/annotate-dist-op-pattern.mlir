// RUN: bishengir-opt -append-vector-layout -annotate-dist-op-layout -eliminate-vector-layout -split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_gather_using_plt_mask
func.func @test_gather_using_plt_mask(%arg0: memref<23xi64, #hivm.address_space<ub>>, %arg1: memref<21xf16, #hivm.address_space<ub>>, %arg2: memref<23xf16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i64 = arith.constant 0 : i64
  %0 = llvm.mlir.constant(1 : i64) : i64
  %c23 = arith.constant 23 : index
  %c0 = arith.constant 0 : index
  %res, %new_true_shape = ave.hir.plt %c23 {mask_op_idx = 0 : i32} : vector<64xi1>, index
  %1 = llvm.alloca %0 x !llvm.struct<"vector_2xvl_s64", (array<2 x vector<64xi32>>)> : (i64) -> !llvm.ptr
  %cast = memref.cast %arg0 : memref<23xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
  call @vload_NORM_int64_t_rank1(%1, %cast, %c0_i64) : (!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) -> ()
  %2 = builtin.unrealized_conversion_cast %res : vector<64xi1> to vector<256xi1>
  %3 = call @_mlir_ciface_cast_int64_t_to_int32_t(%1, %2) : (!llvm.ptr, vector<256xi1>) -> vector<64xi32>
  %4 = ave.hir.vgather %arg1[%c0] [%3], %res  : memref<21xf16, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xf16>
  // CHECK-NEXT : %[[VAR8:.*]] = ave.hir.vgather %arg1[%[[C0:.*]]] [%[[VAR6:.*]]], %[[VAR7:.*]] {element_alignment_bit_width = 32 : i32} : memref<21xf16, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xf16>
  ave.hir.masked_store <NORM_B16> %arg2[%c0], %res, %4  : memref<23xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  // CHECK-NEXT : ave.hir.masked_store <NORM_B16> %arg2[%[[C0:.*]]], %[[RES:.*]], %[[VAR8:.*]] {element_alignment_bit_width = 32 : i32} : memref<23xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  return
}
func.func private @vload_NORM_int64_t_rank1(!llvm.ptr, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i64) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
func.func private @_mlir_ciface_cast_int64_t_to_int32_t(!llvm.ptr, vector<256xi1>) -> vector<64xi32> attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}

// -----
// CHECK-LABEL : @test_gather_using_pge_mask
func.func @test_gather_using_pge_mask(%arg0: memref<2xi64, #hivm.address_space<ub>>, %arg1: memref<1xf16, #hivm.address_space<ub>>, %arg2: memref<2xf16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %0 = ave.hir.pge <VL2> {mask_op_idx = 0 : i32} : vector<64xi1>
  %res = ave.hir.vload <NORM> %arg0[%c0] : memref<2xi64, #hivm.address_space<ub>> into vector<64xi64>
  %1 = ave.hir.vtrunci %res, false, %0 {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2si>} : vector<64xi64>, vector<64xi32>, vector<64xi1>
  // CHECK-NEXT : %[[VAR2:.*]] = ave.hir.vgather %arg1[%[[C0:.*]]] [%[[VAR1:.*]]], %[[VAR0:.*]] {element_alignment_bit_width = 32 : i32} : memref<1xf16, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xf16>
  // CHECK-NEXT : ave.hir.masked_store <NORM_B16> %arg2[%[[C0:.*]]], %[[VAR0:.*]], %[[VAR2:.*]] {element_alignment_bit_width = 32 : i32} : memref<2xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  %2 = ave.hir.vgather %arg1[%c0] [%1], %0 : memref<1xf16, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xf16>
  ave.hir.masked_store <NORM_B16> %arg2[%c0], %0, %2 : memref<2xf16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf16>
  return
}
