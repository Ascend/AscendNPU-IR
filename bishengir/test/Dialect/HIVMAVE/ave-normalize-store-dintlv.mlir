// RUN: bishengir-opt -ave-normalize-ops -split-input-file %s | FileCheck %s

// Test that ave-normalize-ops inserts vdintlv ops before store_with_stride
// when the store carries func_dist_type DINTLV2 (1 level of deinterleave).

// CHECK-LABEL: func.func @test_dintlv2_store_with_stride
func.func @test_dintlv2_store_with_stride(%arg0: memref<4x16xbf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1040 = arith.constant 1040 : index
  %val = arith.constant dense<1.0> : vector<128xbf16>
  %mask = arith.constant dense<true> : vector<128xi1>
  // CHECK: ave.hir.vdintlv %{{.*}}, %{{.*}} {element_alignment_bit_width = 16 : i32} : vector<128xbf16>, vector<128xbf16>
  // CHECK-NEXT: ave.hir.store_with_stride
  ave.hir.store_with_stride %arg0[%c0, %c0], %c1040, %mask, %val {element_alignment_bit_width = 16 : i32, functionType = #ave.func_dist_type<dintlv2>} : memref<4x16xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
  return
}

// -----

// CHECK-LABEL: func.func @test_dintlv4_store_with_stride_2x_vdintlv
func.func @test_dintlv4_store_with_stride_2x_vdintlv(%arg0: memref<4x16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1040 = arith.constant 1040 : index
  %val = arith.constant dense<1> : vector<128xi8>
  %mask = arith.constant dense<true> : vector<128xi1>
  // CHECK: ave.hir.vdintlv %{{.*}}, %{{.*}} {element_alignment_bit_width = 8 : i32} : vector<128xi8>, vector<128xi8>
  // CHECK-NEXT: ave.hir.vdintlv %{{.*}}, %{{.*}} {element_alignment_bit_width = 8 : i32} : vector<128xi8>, vector<128xi8>
  // CHECK-NEXT: ave.hir.store_with_stride
  ave.hir.store_with_stride %arg0[%c0, %c0], %c1040, %mask, %val {element_alignment_bit_width = 8 : i32, functionType = #ave.func_dist_type<dintlv4>} : memref<4x16xi8, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi8>
  return
}

// -----

// CHECK-LABEL: func.func @test_no_func_dist_type_no_vdintlv
func.func @test_no_func_dist_type_no_vdintlv(%arg0: memref<4x16xbf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1040 = arith.constant 1040 : index
  %val = arith.constant dense<1.0> : vector<128xbf16>
  %mask = arith.constant dense<true> : vector<128xi1>
  // CHECK-NOT: ave.hir.vdintlv
  // CHECK: ave.hir.store_with_stride
  ave.hir.store_with_stride %arg0[%c0, %c0], %c1040, %mask, %val {element_alignment_bit_width = 16 : i32} : memref<4x16xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
  return
}
