// RUN: bishengir-opt %s -ave-normalize-ops | FileCheck %s

//===----------------------------------------------------------------------===//
// i1 unaligned store tests
//===----------------------------------------------------------------------===//

func.func @test_i1_store_64_unaligned(%arg0: memref<64xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %val = ave.hir.pge <ALL> {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  // CHECK: preg{{\.}}cast {{.*}} <UNPK4_B8>
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb32>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 8 : i32, hivm.is_continuous, functionType = #ave.func_dist_type<pb8>} : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
  }
 	 
func.func @test_i1_store_128_unaligned(%arg0: memref<128xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>} : vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %val = ave.hir.pge <ALL> {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>} : vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  // CHECK: preg{{\.}}cast {{.*}} <UNPK_B8>
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb16>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 8 : i32, hivm.is_continuous, functionType = #ave.func_dist_type<pb8>} : memref<128xi1, #hivm.address_space<ub>>, vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>, vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
}

//===----------------------------------------------------------------------===//
// i1 unaligned load tests
//===----------------------------------------------------------------------===//

func.func @test_i1_load_64_unaligned(%arg0: memref<64xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK: vload <NORM> {{.*}} functionType = #ave.func_dist_type<pb16>
  // CHECK: preg{{\.}}cast {{.*}} <PK_B16>
  %0 = ave.hir.vload <NORM> %arg0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>, hivm.is_continuous} : memref<64xi1, #hivm.address_space<ub>> into vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %1 = "test.unary"(%0) : (vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>) -> vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
}

func.func @test_i1_load_128_unaligned(%arg0: memref<128xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK: vload <NORM> {{.*}} functionType = #ave.func_dist_type<pb16>
  // CHECK: preg{{\.}}cast {{.*}} <PK_B16>
  %0 = ave.hir.vload <NORM> %arg0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<pb8>, hivm.is_continuous} : memref<128xi1, #hivm.address_space<ub>> into vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %1 = "test.unary"(%0) : (vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>) -> vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
}
