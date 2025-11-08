// RUN: bishengir-opt -hivm-lift-lowest-stride -split-input-file %s | FileCheck %s

module {
  func.func @test_copy_vadd_static(%arg0: memref<16xi32, strided<[32], offset: 1>, #hivm.address_space<gm>>, %arg1: memref<16xi32, strided<[32]>, #hivm.address_space<gm>>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc() : memref<16x8xi32, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [16, 1] [1, 1] : memref<16x8xi32, #hivm.address_space<ub>> to memref<16xi32, strided<[8]>, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<16x8xi32, #hivm.address_space<ub>>
    %subview_1 = memref.subview %alloc_0[0, 0] [16, 1] [1, 1] : memref<16x8xi32, #hivm.address_space<ub>> to memref<16xi32, strided<[8]>, #hivm.address_space<ub>>

    // CHECK: hivm.hir.load ins(%{{.*}} : memref<16xi32, strided<[32], offset: 1>, #hivm.address_space<gm>>) outs(%{{.*}} : memref<16xi32, strided<[8]>, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg0 : memref<16xi32, strided<[32], offset: 1>, #hivm.address_space<gm>>) outs(%subview : memref<16xi32, strided<[8]>, #hivm.address_space<ub>>)

    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<16x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>, memref<16x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<16x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%subview, %subview : memref<16xi32, strided<[8]>, #hivm.address_space<ub>>, memref<16xi32, strided<[8]>, #hivm.address_space<ub>>)
      outs(%subview_1 : memref<16xi32, strided<[8]>, #hivm.address_space<ub>>)

    // CHECK: hivm.hir.store ins(%{{.*}} : memref<16xi32, strided<[8]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<16xi32, strided<[32]>, #hivm.address_space<gm>>)
    hivm.hir.store ins(%subview_1 : memref<16xi32, strided<[8]>, #hivm.address_space<ub>>) outs(%arg1 : memref<16xi32, strided<[32]>, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_lift_vector_unary_ops(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>,
                                        %dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>,
                                        %dst1 : memref<?xf32, strided<[?]>, #hivm.address_space<ub>>,
                                        %src2 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>,
                                        %dst2 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>) {
    // CHECK:  hivm.hir.vexp ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vabs ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vln ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vrelu ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vrsqrt ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vsqrt ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vtanh ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vsin ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vcos ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vrec ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vcast ins(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf32, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK:  hivm.hir.vnot ins(%{{.*}} : memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>)

    hivm.hir.vexp   ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vabs   ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vln    ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vrelu  ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vrsqrt ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vsqrt  ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vtanh  ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vsin   ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vcos   ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vrec   ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vcast  ins(%src : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst1 : memref<?xf32, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vnot    ins(%src2 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst2 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>)
    return
  }
}

// -----
module {
  func.func @test_lift_vector_binary_ops(
      %cst : f16,
      %src0 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>,
      %src0_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>,
      %src0_i32 : memref<?xi32, strided<[?]>, #hivm.address_space<ub>>,
      %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>,
      %src1_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>,
      %src1_i32 : memref<?xi32, strided<[?]>, #hivm.address_space<ub>>,
      %dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>,
      %dst_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>,
      %dst_i1 : memref<?xi1, strided<[?]>, #hivm.address_space<ub>>,
      %dst_i32 : memref<?xi32, strided<[?]>, #hivm.address_space<ub>>) {
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, f16) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vmul ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vsub ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vdiv ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vmax ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vmin ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vor ins(%{{.*}}, %{{.*}} : memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vand ins(%{{.*}}, %{{.*}} : memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xi16, strided<[?, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.vcmp ins(%{{.*}}, %{{.*}} : memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xf16, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xi1, strided<[?, 1]>, #hivm.address_space<ub>>) compare_mode = <lt>
    // CHECK: hivm.hir.vpow ins(%{{.*}}, %{{.*}} : memref<?x1xi32, strided<[?, 1]>, #hivm.address_space<ub>>, memref<?x1xi32, strided<[?, 1]>, #hivm.address_space<ub>>) outs(%{{.*}} : memref<?x1xi32, strided<[?, 1]>, #hivm.address_space<ub>>)

    hivm.hir.vadd ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%src0, %cst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, f16) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vmul ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vsub ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vdiv ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vmax ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vmin ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vor ins(%src0_i16, %src1_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>, memref<?xi16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vand ins(%src0_i16, %src1_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>, memref<?xi16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst_i16 : memref<?xi16, strided<[?]>, #hivm.address_space<ub>>)
    hivm.hir.vcmp ins(%src0, %src1 : memref<?xf16, strided<[?]>, #hivm.address_space<ub>>, memref<?xf16, strided<[?]>, #hivm.address_space<ub>>) outs(%dst_i1 : memref<?xi1, strided<[?]>, #hivm.address_space<ub>>) compare_mode = <lt>
    hivm.hir.vpow ins(%src0_i32, %src1_i32 : memref<?xi32, strided<[?]>, #hivm.address_space<ub>>, memref<?xi32, strided<[?]>, #hivm.address_space<ub>>) outs(%dst_i32 : memref<?xi32, strided<[?]>, #hivm.address_space<ub>>)

    return
  }
}

// -----
module {
  func.func @test_lift_vector_ternary_ops(
      %src0 : memref<?x16xi1, strided<[512, 32]>>,
      %src1 : memref<?x16xf16, strided<[512, 32]>>,
      %src2 : memref<?x16xf16, strided<[512, 32]>>,
      %dst : memref<?x16xf16, strided<[512, 32]>>) {

    // CHECK: hivm.hir.vsel ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<?x16x1xi1, strided<[512, 32, 1]>>, memref<?x16x1xf16, strided<[512, 32, 1]>>, memref<?x16x1xf16, strided<[512, 32, 1]>>) outs(%{{.*}} : memref<?x16x1xf16, strided<[512, 32, 1]>>)
    hivm.hir.vsel ins(%src0, %src1, %src2 : memref<?x16xi1, strided<[512, 32]>>, memref<?x16xf16, strided<[512, 32]>>, memref<?x16xf16, strided<[512, 32]>>)
                  outs(%dst : memref<?x16xf16, strided<[512, 32]>>)
    return
  }
}

// -----
module {
  func.func @test_lift_vbrc_ops(
      %cst: f32, %arg: memref<32x7xf32, strided<[?, 8]>>,
      %src : memref<1x?xi16, strided<[?, 16]>>,
      %dst : memref<16x?xi16, strided<[?, 16]>>) {
    // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : memref<32x7x1xf32, strided<[?, 8, 1]>>)
    // CHECK: hivm.hir.vbrc ins(%{{.*}} : memref<1x?x1xi16, strided<[?, 16, 1]>>) outs(%{{.*}} : memref<16x?x1xi16, strided<[?, 16, 1]>>) broadcast_dims = [0]
    hivm.hir.vbrc ins(%cst: f32) outs(%arg: memref<32x7xf32, strided<[?, 8]>>)
    hivm.hir.vbrc ins(%src : memref<1x?xi16, strided<[?, 16]>>) outs(%dst : memref<16x?xi16, strided<[?, 16]>>) broadcast_dims = [0]

    return
  }
}

// -----
module {
  func.func @test_lift_vreduce_ops(
      %src : memref<?x32xf16, strided<[?, 16]>>,
      %dst : memref<?x1xf16, strided<[?, 16]>>,
      %dst1 : memref<?x1xi32, strided<[?, 16]>>,
      %alloc_8 : memref<16384xf32>
      ) {
    // CHECK: hivm.hir.vreduce <sum> ins(%{{.*}} : memref<?x32x1xf16, strided<[?, 16, 1]>>) outs(%{{.*}} : memref<?x1x1xf16, strided<[?, 16, 1]>>) reduce_dims = [1]
    hivm.hir.vreduce <sum> ins(%src : memref<?x32xf16, strided<[?, 16]>>) outs(%dst : memref<?x1xf16, strided<[?, 16]>>) reduce_dims = [1]
    // CHECK: hivm.hir.vreduce <max_with_index_left> ins(%{{.*}} : memref<?x32x1xf16, strided<[?, 16, 1]>>) outs(%{{.*}}, %{{.*}} : memref<?x1x1xf16, strided<[?, 16, 1]>>, memref<?x1x1xi32, strided<[?, 16, 1]>>) temp_buffer(%{{.*}} : memref<16384xf32>) reduce_dims = [1]
    hivm.hir.vreduce <max_with_index_left>
      ins(%src : memref<?x32xf16, strided<[?, 16]>>)
      outs(%dst, %dst1 : memref<?x1xf16, strided<[?, 16]>>, memref<?x1xi32, strided<[?, 16]>>)
      temp_buffer(%alloc_8 : memref<16384xf32>)
      reduce_dims = [1]

    return
  }
}

// -----
module {
  func.func @test_lift_vtranspose_ops(%src : memref<6x16x32x8xf32, strided<[?, ?, ?, 8]>>,
      %dst : memref<6x16x8x32xf32, strided<[?, ?, ?, 8]>>) {
    // CHECK: hivm.hir.vtranspose ins(%{{.*}} : memref<6x16x32x8x1xf32, strided<[?, ?, ?, 8, 1]>>) outs(%{{.*}} : memref<6x16x8x32x1xf32, strided<[?, ?, ?, 8, 1]>>) permutation = [0, 1, 3, 2, 4]
    hivm.hir.vtranspose ins(%src :  memref<6x16x32x8xf32, strided<[?, ?, ?, 8]>>)
                        outs(%dst : memref<6x16x8x32xf32, strided<[?, ?, ?, 8]>>)
                        permutation = [0, 1, 3, 2]

    return
  }
}

// -----
module {
  func.func @test_lift_arg_transpose(
      %src0 :  memref<?x?x8xf16, strided<[?, ?, 16]>>,
      %src1 :  memref<?x?x8xf16, strided<[?, ?, 16]>>,
      %dst : memref<?x?x8xf16, strided<[?, ?, 16]>>) {
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<?x?x8x1xf16, strided<[?, ?, 16, 1]>>, memref<?x?x8x1xf16, strided<[?, ?, 16, 1]>>) outs(%{{.*}} : memref<?x?x8x1xf16, strided<[?, ?, 16, 1]>>) transpose = [1, 0, 2, 3]
    hivm.hir.vadd
      ins(%src0, %src1 : memref<?x?x8xf16, strided<[?, ?, 16]>>, memref<?x?x8xf16, strided<[?, ?, 16]>>)
      outs(%dst : memref<?x?x8xf16, strided<[?, ?, 16]>>)
      transpose=[1, 0, 2]

    return
  }
}

// -----
module {
  func.func @test_lift_cumsum_op(%src: memref<2x25xi16, strided<[800, 32]>>, %dst: memref<2x25xi16, strided<[800, 32]>>) {
    // CHECK: hivm.hir.vcumsum ins(%{{.*}} : memref<2x25x1xi16, strided<[800, 32, 1]>>) outs(%{{.*}} : memref<2x25x1xi16, strided<[800, 32, 1]>>) cum_dims = [0]
    hivm.hir.vcumsum ins(%src : memref<2x25xi16, strided<[800, 32]>>)
                     outs(%dst : memref<2x25xi16, strided<[800, 32]>>)
                     cum_dims = [0]
    return
  }
}

// -----
module {
  func.func @test_lift_mul_extended_op(%src0: memref<2x25xi16, strided<[800, 32]>>, %src1: memref<2x25xi16, strided<[800, 32]>>, 
                                       %dst0: memref<2x25xi16, strided<[800, 32]>>, %dst1: memref<2x25xi16, strided<[800, 32]>>) {
    // CHECK: hivm.hir.vmulextended ins(%{{.*}}, %{{.*}} : memref<2x25x1xi16, strided<[800, 32, 1]>>, memref<2x25x1xi16, strided<[800, 32, 1]>>) outs(%{{.*}}, %{{.*}} : memref<2x25x1xi16, strided<[800, 32, 1]>>, memref<2x25x1xi16, strided<[800, 32, 1]>>)
    hivm.hir.vmulextended ins(%src0, %src1 : memref<2x25xi16, strided<[800, 32]>>, memref<2x25xi16, strided<[800, 32]>>)
                          outs(%dst0, %dst1 : memref<2x25xi16, strided<[800, 32]>>, memref<2x25xi16, strided<[800, 32]>>)
    return
  }
}

// -----
module {
  func.func @test_nolift_arg_vbrc_ops(
      %src : memref<?x1xf16, strided<[8, 8]>>,
      %dst : memref<?x16xf16, strided<[16, 1]>>) {
    // CHECK-NOT: memref.extract_strided_metadata
    hivm.hir.vbrc
      ins(%src : memref<?x1xf16, strided<[8, 8]>>)
      outs(%dst : memref<?x16xf16, strided<[16, 1]>>)
      broadcast_dims = [1]

    return
  }
}

// -----

module {
  func.func @test_copy_collapse(%src: memref<16x4xf16, strided<[32, 8]>>, %dst: memref<16x4xf16, strided<[4, 1]>>) {
    // CHECK: memref.extract_strided_metadata
    // CHECK: hivm.hir.copy
    // CHECK-SAME: collapse_reassociation = {{\[}}[0, 1], [2]]
    hivm.hir.copy ins(%src : memref<16x4xf16, strided<[32, 8]>>) outs(%dst : memref<16x4xf16, strided<[4, 1]>>) collapse_reassociation = [[0, 1]]
    return
  }
}
