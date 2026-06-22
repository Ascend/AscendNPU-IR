// RUN: bishengir-opt %s -convert-hivm-to-std  -split-input-file | FileCheck %s
 
module {
  func.func @test_reduce_op_check_arith_op() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_sum_r_int32_t
    %src_2 = memref.alloca() : memref<16xi32>
    %dst_2 = memref.alloca() : memref<1xi32>
    hivm.hir.vreduce <sum>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            reduce_dims = [0]
 
    // CHECK: call @reduce_prod_r_int32_t
    hivm.hir.vreduce <prod>  ins(%src_2 : memref<16xi32>)
                             outs(%dst_2 : memref<1xi32>)
                             reduce_dims = [0]
 
    // CHECK: call @reduce_max_r_int32_t
    hivm.hir.vreduce <max>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            reduce_dims = [0]
 
    // CHECK: call @reduce_min_r_int32_t
    hivm.hir.vreduce <min>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            reduce_dims = [0]
    
    // CHECK: call @reduce_andi_r_int32_t
    hivm.hir.vreduce <andi>  ins(%src_2 : memref<16xi32>)
                            outs(%dst_2 : memref<1xi32>)
                            reduce_dims = [0]
 
    return
  }
}
 
// -----
 
module {
  // CHECK: func.func private @reduce_sum_aar_int32_t(memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>, i32)
  func.func @test_reduce_op_4d() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src_2 = memref.alloca() : memref<5x2x3x16xi32>
    %dst_2 = memref.alloca() : memref<5x2x3x1xi32>
    // CHECK: scf.for
    // CHECK: func.call @reduce_sum_aar_int32_t
    hivm.hir.vreduce <sum>  ins(%src_2 : memref<5x2x3x16xi32>)
                            outs(%dst_2 : memref<5x2x3x1xi32>)
                            reduce_dims = [3]
    return
  }
}
 
// -----
 
module {
  func.func @test_reduce_op_check_cross() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @enablevc_reduce_sum_r_half
    %src_2 = memref.alloca() : memref<16xf16>
    %dst_2 = memref.alloca() : memref<1xf16>
    hivm.hir.vreduce <sum>  ins(%src_2 : memref<16xf16>)
                            outs(%dst_2 : memref<1xf16>)
                            reduce_dims = [0]
 
    return
  }
}
 
// -----
 
module {
  func.func @test_reduce_op_check_with_rank_reducing() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK-COUNT-5: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @reduce_max_ra0a1_int32_t
    %src_3 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi32>
    %dst_3 = memref.alloca() : memref<1x16x8x32x8x16x8x32xi32>
    hivm.hir.vreduce <max> ins(%src_3 : memref<16x16x8x32x8x16x8x32xi32>)
                           outs(%dst_3 : memref<1x16x8x32x8x16x8x32xi32>)
                           reduce_dims = [0]
 
    // CHECK-COUNT-5: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @reduce_min_aar_int32_t
    %src_4 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi32>
    %dst_4 = memref.alloca() : memref<16x16x8x32x8x16x8x1xi32>
    hivm.hir.vreduce <min> ins(%src_4 : memref<16x16x8x32x8x16x8x32xi32>)
                           outs(%dst_4 : memref<16x16x8x32x8x16x8x1xi32>)
                           reduce_dims = [7]
 
    return
  }
}
 
// -----
 
module {
  func.func @test_reduce_op_check_with_rank_reducing_dynamic_shape(
                         %src_5 : memref<16x?x?x?x8x?x8x32xf16>,
                         %dst_5 : memref<1x?x?x?x8x?x8x32xf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: memref.dim
    // CHECK: scf.for
    // CHECK-COUNT-2: memref.subview
    // CHECK: call @reduce_max_ra0a1_half
    hivm.hir.vreduce <max> ins(%src_5 : memref<16x?x?x?x8x?x8x32xf16>)
                           outs(%dst_5 : memref<1x?x?x?x8x?x8x32xf16>)
                           reduce_dims = [0]
 
    return
  }
}
 
// -----
 
module {
  func.func @test_reduce_op_check_temp_buffer() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_max_ra0a1_int32_t(%{{.+}}, %{{.+}}, %{{.+}})
    %src_3 = memref.alloca() : memref<16x16x8x32x8x16x8x32xi32>
    %dst_3 = memref.alloca() : memref<1x16x8x32x8x16x8x32xi32>
    %tmp_3 = memref.alloca() : memref<2147483648xi32>
    hivm.hir.vreduce <max> ins(%src_3 : memref<16x16x8x32x8x16x8x32xi32>)
                           outs(%dst_3 : memref<1x16x8x32x8x16x8x32xi32>)
                           temp_buffer (%tmp_3 : memref<2147483648xi32>)
                           reduce_dims = [0]
 
    return
  }
}
 
// -----
 
module {
  func.func @test_reduce_min_with_index() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_min_with_index_left_ar_float
    %src = memref.alloc() : memref<1024x8xf32, #hivm.address_space<ub>>
    %ubC = memref.alloc() : memref<1024x1xf32, #hivm.address_space<ub>>
    %ubC_index = memref.alloc() : memref<1024x1xi32, #hivm.address_space<ub>>
    hivm.hir.vreduce <min_with_index_left>
                     ins(%src : memref<1024x8xf32, #hivm.address_space<ub>>)
                     outs(%ubC, %ubC_index : memref<1024x1xf32, #hivm.address_space<ub>>, memref<1024x1xi32, #hivm.address_space<ub>>)
                     reduce_dims = [1]
    return
  }
}
 
// -----
 
module {
  func.func @test_reduce_sum_mid_axis() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_sum_ara_float
    %src = memref.alloc() : memref<5x16x16xf32, #hivm.address_space<ub>>
    %ubC = memref.alloc() : memref<5x1x16xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum>
                     ins(%src : memref<5x16x16xf32, #hivm.address_space<ub>>)
                     outs(%ubC : memref<5x1x16xf32, #hivm.address_space<ub>>)
                     reduce_dims = [1]
    return
  }
}

// -----

// 2D first-axis reduce (RA).
module {
  func.func @test_reduce_sum_2d_ra() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_sum_ra_int32_t
    %src = memref.alloca() : memref<16x8xi32>
    %dst = memref.alloca() : memref<1x8xi32>
    hivm.hir.vreduce <sum> ins(%src : memref<16x8xi32>)
                       outs(%dst : memref<1x8xi32>)
                       reduce_dims = [0]
    return
  }
}

// -----

// 2D last-axis reduce (AR).
module {
  func.func @test_reduce_sum_2d_ar() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_sum_ar_int32_t
    %src = memref.alloca() : memref<8x16xi32>
    %dst = memref.alloca() : memref<8x1xi32>
    hivm.hir.vreduce <sum> ins(%src : memref<8x16xi32>)
                       outs(%dst : memref<8x1xi32>)
                       reduce_dims = [1]
    return
  }
}

// -----

// 2D first-axis reduce (RA) with float, no enablevc prefix.
module {
  func.func @test_reduce_sum_2d_ra_float() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_sum_ra_float
    %src = memref.alloca() : memref<16x8xf32>
    %dst = memref.alloca() : memref<1x8xf32>
    hivm.hir.vreduce <sum> ins(%src : memref<16x8xf32>)
                       outs(%dst : memref<1x8xf32>)
                       reduce_dims = [0]
    return
  }
}

// -----

// 2D last-axis reduce (AR) with float, enablevc prefix.
module {
  func.func @test_reduce_sum_2d_ar_float() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @enablevc_reduce_sum_ar_float
    %src = memref.alloca() : memref<8x16xf32>
    %dst = memref.alloca() : memref<8x1xf32>
    hivm.hir.vreduce <sum> ins(%src : memref<8x16xf32>)
                       outs(%dst : memref<8x1xf32>)
                       reduce_dims = [1]
    return
  }
}

// -----

// 3D first-axis reduce (RA0A1) explicit 3D.
module {
  func.func @test_reduce_sum_3d_ra0a1() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_sum_ra0a1_float
    %src = memref.alloca() : memref<16x8x4xf32>
    %dst = memref.alloca() : memref<1x8x4xf32>
    hivm.hir.vreduce <sum> ins(%src : memref<16x8x4xf32>)
                       outs(%dst : memref<1x8x4xf32>)
                       reduce_dims = [0]
    return
  }
}

// -----

// 3D last-axis reduce (AAR) explicit 3D with float.
module {
  func.func @test_reduce_sum_3d_aar_float() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @enablevc_reduce_sum_aar_float
    %src = memref.alloca() : memref<4x8x16xf32>
    %dst = memref.alloca() : memref<4x8x1xf32>
    hivm.hir.vreduce <sum> ins(%src : memref<4x8x16xf32>)
                       outs(%dst : memref<4x8x1xf32>)
                       reduce_dims = [2]
    return
  }
}

// -----

// 3D middle-axis reduce (ARA) with max and int32.
module {
  func.func @test_reduce_max_3d_ara() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: call @reduce_max_ara_int32_t
    %src = memref.alloca() : memref<4x16x8xi32>
    %dst = memref.alloca() : memref<4x1x8xi32>
    hivm.hir.vreduce <max> ins(%src : memref<4x16x8xi32>)
                         outs(%dst : memref<4x1x8xi32>)
                         reduce_dims = [1]
    return
  }
}