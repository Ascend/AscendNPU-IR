// RUN: bishengir-opt %s -ave-normalize-ops | FileCheck %s
// CHECK-NOT: element_alignment_bit_width

//===----------------------------------------------------------------------===//
// i1 unaligned store tests
//===----------------------------------------------------------------------===//

func.func @test_i1_store_64_unaligned(%arg0: memref<64xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %val = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  // CHECK: preg{{\.}}cast {{.*}} <UNPK4_B8>
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb32>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous, functionType = #ave.func_dist_type<pb8>} : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
  }
 	 
func.func @test_i1_store_128_unaligned(%arg0: memref<128xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %val = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb8>} : vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  // CHECK: preg{{\.}}cast {{.*}} <UNPK_B8>
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb16>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous, functionType = #ave.func_dist_type<pb8>} : memref<128xi1, #hivm.address_space<ub>>, vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>, vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
}

func.func @test_i1_store_64_b16_continuous_unaligned(%arg0: memref<128xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  // CHECK-LABEL: func.func @test_i1_store_64_b16_continuous_unaligned
  // CHECK: preg{{\.}}cast {{.*}} <UNPK_B16>
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb32>
  scf.for %iv = %c0 to %c128 step %c64 {
    %sub = memref.reinterpret_cast %arg0 to offset: [%iv], sizes: [64], strides: [1] : memref<128xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
    %val = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
    ave.hir.masked_store <NORM_B8> %sub[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous, functionType = #ave.func_dist_type<pb16>} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  }
  return
}

func.func @test_i1_store_64_b16_unaligned(%arg0: memref<64xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  %val = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  // CHECK-LABEL: func.func @test_i1_store_64_b16_unaligned
  // CHECK-NOT: preg{{\.}}cast
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb16>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb16>} : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  return
}

func.func @test_i1_store_64_b16_continuous_unaligned_no_loop(%arg0: memref<64xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  %val = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  // CHECK-LABEL: func.func @test_i1_store_64_b16_continuous_unaligned_no_loop
  // CHECK-NOT: preg{{\.}}cast
  // CHECK: masked_store <NORM_B8> {{.*}} functionType = #ave.func_dist_type<pb16>
  ave.hir.masked_store <NORM_B8> %arg0[%c0], %mask, %val {ave.unaligned_ub_access = #ave.unaligned_ub_access, hivm.is_continuous, functionType = #ave.func_dist_type<pb16>} : memref<64xi1, #hivm.address_space<ub>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>, vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b16>}>>
  return
}

//===----------------------------------------------------------------------===//
// i1 unaligned load tests
//===----------------------------------------------------------------------===//

func.func @test_i1_load_64_unaligned(%arg0: memref<64xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK: vload <NORM> {{.*}} functionType = #ave.func_dist_type<pb16>
  // CHECK: preg{{\.}}cast {{.*}} <PK_B16>
  %0 = ave.hir.vload <NORM> %arg0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb8>, hivm.is_continuous} : memref<64xi1, #hivm.address_space<ub>> into vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %1 = "test.unary"(%0) : (vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>) -> vector<64xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
}

func.func @test_i1_load_128_unaligned(%arg0: memref<128xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  // CHECK: vload <NORM> {{.*}} functionType = #ave.func_dist_type<pb16>
  // CHECK: preg{{\.}}cast {{.*}} <PK_B16>
  %0 = ave.hir.vload <NORM> %arg0[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb8>, hivm.is_continuous} : memref<128xi1, #hivm.address_space<ub>> into vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  %1 = "test.unary"(%0) : (vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>) -> vector<128xi1, #ave.vector_layout<{mem = #ave.vec_mem_type<b8>}>>
  return
}
