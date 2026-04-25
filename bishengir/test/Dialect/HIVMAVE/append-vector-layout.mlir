// RUN: bishengir-opt -mlir-print-vector-layout-attr -append-vector-layout %s | FileCheck %s
#map = affine_map<()[i, j] -> (i + j)>
#map1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @test_vector_layout_attr
func.func @test_vector_layout_attr(%arg0: i32, %arg1: i32, %arg2: memref<2x512xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %c0_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    %3 = ave.hir.broadcast %c1_i32, %2 : i32, vector<64xi1> -> vector<64xi32>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %4 = arith.index_cast %arg3 : index to i32
      %5 = ave.hir.scalar_broadcast %arg0 : i32 -> vector<64xi32>
      %6 = ave.hir.scalar_broadcast %4 : i32 -> vector<64xi32>
      %7 = ave.hir.pge <ALL> : vector<64xi1>
      %8 = ave.hir.vadd %5, %6, %7 : vector<64xi32>, vector<64xi1>
      %9 = ave.hir.scalar_broadcast %arg1 : i32 -> vector<64xi32>
      %10 = ave.hir.pge <ALL> : vector<64xi1>
      %11 = ave.hir.vcmp <LT> %8, %9, %10 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
      %12 = ave.hir.vsel %11, %3, %1 : vector<64xi1>, vector<64xi32>
      %13 = ave.hir.pge <ALL> : vector<64xi1>
      %14 = ave.hir.vtrunci %12, false, %13 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
      scf.for %arg4 = %c0 to %c512 step %c64 {
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg2 : memref<2x512xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index
        %15 = affine.apply #map()[%arg3, %arg4]
        %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%15], sizes: [64], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<64xi8, #map1, #hivm.address_space<ub>>
        %16 = ave.hir.pge <ALL> : vector<64xi1>
        // CHECK-NEXT : ave.hir.masked_store <NORM_B8> %reinterpret_cast%[[C0:.*]], %[[VAR16:.*]], %[[VAR14:.*]] : memref<64xi8, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<dense>, reg = #ave.vec_reg_type<n>}>>, vector<64xi8, #ave.vector_layout<{mem = #ave.vec_mem_type<dense>, reg = #ave.vec_reg_type<4n0>}>> 
        ave.hir.masked_store <NORM_B8> %reinterpret_cast[%c0], %16, %14 : memref<64xi8, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
      } {element_alignment_bit_width = 8 : i32}
    } {element_alignment_bit_width = 32 : i32}
    return
  }