// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -append-vector-layout -annotate-dist-op-layout -eliminate-vector-layout \
// RUN: -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s

// CHECK-LABEL: @test_preg_arith_op_lowering
func.func @test_preg_arith_op_lowering(%arg0: memref<8xi1, #hivm.address_space<ub>>, %arg1: memref<8x8xi1, strided<[256, 1]>, #hivm.address_space<ub>>, %arg2: memref<8x8xi32, #hivm.address_space<ub>>, %arg3: memref<i32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i16 = arith.constant 0 : i16
  %c1_i8 = arith.constant 1 : i8
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
  %1 = ave.hir.broadcast %c0_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  %2 = ave.hir.pge <ALLF> : vector<64xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b32"
  %res = ave.hir.vload <BRC_B32> %arg3[] : memref<i32, #hivm.address_space<ub>> into vector<1xi32>
  %3 = builtin.unrealized_conversion_cast %res : vector<1xi32> to vector<i32>
  %4 = builtin.unrealized_conversion_cast %3 : vector<i32> to vector<64xi32>
  %5:2 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %3, %arg6 = %4) -> (vector<i32>, vector<64xi32>) {
    %9 = builtin.unrealized_conversion_cast %arg6 : vector<64xi32> to vector<i32>
    %subview = memref.subview %arg1[%arg4, 0] [1, 8] [1, 1] : memref<8x8xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1x8xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg2[%arg4, 0] [1, 8] [1, 1] : memref<8x8xi32, #hivm.address_space<ub>> to memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %res_1 = ave.hir.vload <NORM> %arg0[%c0] : memref<8xi1, #hivm.address_space<ub>> into vector<64xi1>
    %10 = builtin.unrealized_conversion_cast %res_1 : vector<64xi1> to vector<256xi1>
    %11 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
    %12 = ave.hir.scalar_broadcast %c1_i8 : i8 -> vector<256xi8>
    %13 = ave.hir.vsel %10, %12, %11 : vector<256xi1>, vector<256xi8>
    %14 = ave.hir.pge <ALL> : vector<256xi1>
    %15 = arith.addi %arg4, %c1 : index
    %res_2, %new_true_shape = ave.hir.plt %15 : vector<256xi1>, index
    %res_3, %new_true_shape_4 = ave.hir.plt %arg4 : vector<256xi1>, index
    %16 = ave.hir.preg.xor <b8> %res_2, %res_3, %14 : vector<256xi1>
    %17 = ave.hir.pge <ALL> : vector<256xi1>
    %18 = ave.hir.broadcast %c0_i16, %17 : i16, vector<256xi1> -> vector<256xi8>
    %19 = ave.hir.vor %13, %18, %16 : vector<256xi8>, vector<256xi1>
    %20 = ave.hir.pge <ALL> : vector<256xi1>
    %res1, %res2 = ave.hir.vintlv %19, %18 : vector<256xi8>, vector<256xi8>
    %21 = ave.hir.vxor %res1, %res2, %20 : vector<256xi8>, vector<256xi1>
    %res1_5, %res2_6 = ave.hir.vintlv %21, %18 : vector<256xi8>, vector<256xi8>
    %22 = ave.hir.vxor %res1_5, %res2_6, %20 : vector<256xi8>, vector<256xi1>
    %res1_7, %res2_8 = ave.hir.vintlv %22, %18 : vector<256xi8>, vector<256xi8>
    %23 = ave.hir.vxor %res1_7, %res2_8, %20 : vector<256xi8>, vector<256xi1>
    %res1_9, %res2_10 = ave.hir.vintlv %23, %18 : vector<256xi8>, vector<256xi8>
    %24 = ave.hir.vxor %res1_9, %res2_10, %20 : vector<256xi8>, vector<256xi1>
    %res1_11, %res2_12 = ave.hir.vintlv %24, %18 : vector<256xi8>, vector<256xi8>
    %25 = ave.hir.vxor %res1_11, %res2_12, %20 : vector<256xi8>, vector<256xi1>
    %res1_13, %res2_14 = ave.hir.vintlv %25, %18 : vector<256xi8>, vector<256xi8>
    %26 = ave.hir.vxor %res1_13, %res2_14, %20 : vector<256xi8>, vector<256xi1>
    %res1_15, %res2_16 = ave.hir.vintlv %26, %18 : vector<256xi8>, vector<256xi8>
    %27 = ave.hir.vxor %res1_15, %res2_16, %20 : vector<256xi8>, vector<256xi1>
    %res1_17, %res2_18 = ave.hir.vintlv %27, %18 : vector<256xi8>, vector<256xi8>
    %28 = ave.hir.vxor %res1_17, %res2_18, %20 : vector<256xi8>, vector<256xi1>
    %29 = ave.hir.vector_broadcast %28, %14, true : vector<256xi8>, vector<256xi1> -> vector<256xi8>
    %30 = ave.hir.scalar_broadcast %c0_i8 : i8 -> vector<256xi8>
    %31 = ave.hir.vcmp <NE> %29, %30, %14 : vector<256xi8>, vector<256xi1> -> vector<256xi1>
    %32 = builtin.unrealized_conversion_cast %31 : vector<256xi1> to vector<64xi1>
    %33 = ave.hir.pge <VL8> : vector<64xi1>
    %subview_19 = memref.subview %subview[0, 0] [1, 8] [1, 1] : memref<1x8xi1, strided<[256, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_20 = ave.hir.vload <NORM> %subview_19[%c0] : memref<8xi1, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi1>
    // CHECK : "hivm_regbaseintrins.intr.hivm.plds.b8"
    %subview_21 = memref.subview %subview_0[0, 0] [1, 8] [1, 1] : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_22 = ave.hir.vload <NORM> %subview_21[%c0] : memref<8xi32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi32>
    %34 = ave.hir.pge <ALL> : vector<64xi1>
    %35 = ave.hir.preg.xor <b8> %32, %2, %34 : vector<64xi1>
    // CHECK : "hivm_regbaseintrins.intr.hivm.pxor.z"
    %36 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b32"
    %37 = ave.hir.preg.or <b8> %35, %res_20, %36 : vector<64xi1>
    // CHECK : "hivm_regbaseintrins.intr.hivm.pge.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pintlv.b8"
    // CHECK : "hivm_regbaseintrins.intr.hivm.pintlv.b8"
    // CHECK: "hivm_regbaseintrins.intr.hivm.por.z"
    %38 = ave.hir.vsel %37, %res_22, %1 : vector<64xi1>, vector<64xi32>
    %39 = builtin.unrealized_conversion_cast %9 : vector<i32> to i32
    %40 = ave.hir.vsel %33, %38, %1 : vector<64xi1>, vector<64xi32>
    %41 = ave.hir.pge <ALL> : vector<64xi1>
    %42 = builtin.unrealized_conversion_cast %39 : i32 to vector<1xi32>
    %43 = builtin.unrealized_conversion_cast %42 : vector<1xi32> to vector<64xi32>
    %44 = ave.hir.reduction <add>, %40, %41 : vector<64xi32>, vector<64xi1> -> vector<64xi32>
    %45 = ave.hir.pge <ALL> : vector<64xi1>
    %46 = ave.hir.vadd %43, %44, %45 : vector<64xi32>, vector<64xi1>
    %47 = builtin.unrealized_conversion_cast %46 : vector<64xi32> to vector<1xi32>
    %48 = builtin.unrealized_conversion_cast %47 : vector<1xi32> to i32
    %49 = builtin.unrealized_conversion_cast %48 : i32 to vector<i32>
    %50 = builtin.unrealized_conversion_cast %49 : vector<i32> to vector<64xi32>
    scf.yield %49, %50 : vector<i32>, vector<64xi32>
  } {element_alignment_bit_width = -1 : i32}
  %6 = builtin.unrealized_conversion_cast %5#1 : vector<64xi32> to vector<i32>
  %7 = builtin.unrealized_conversion_cast %6 : vector<i32> to vector<1xi32>
  %8 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %arg3[], %8, %7 : memref<i32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi32>
  return
}