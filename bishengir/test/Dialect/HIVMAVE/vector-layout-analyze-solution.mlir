// RUN: bishengir-opt -analyze-vector-layout %s -split-input-file | FileCheck %s


// CHECK-LABEL: @test_big_vf_with_mask_ops
func.func @test_big_vf_with_mask_ops(%arg0: memref<2048xf32, #hivm.address_space<ub>>, %arg1: memref<320xi32, #hivm.address_space<ub>>, %arg2: memref<2048xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c64 = arith.constant 64 : index
  %c23_i32 = arith.constant 23 : i32
  %c255_i32 = arith.constant 255 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c40_i32 = arith.constant 40 : i32
  %c8388607_i32 = arith.constant 8388607 : i32
  %c8388608_i32 = arith.constant 8388608 : i32
  %c16_i32 = arith.constant 16 : i32
  %c65535_i32 = arith.constant 65535 : i32
  %c31_i32 = arith.constant 31 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %cst_1 = arith.constant 4.65661287E-10 : f32
  %cst_2 = arith.constant 3.14159274 : f32
  %cst_3 = arith.constant 1.57079637 : f32
  %cst_4 = arith.constant 2.75573188E-6 : f32
  %cst_5 = arith.constant -1.98412701E-4 : f32
  %cst_6 = arith.constant 0.00833333377 : f32
  %cst_7 = arith.constant -0.166666672 : f32
  %cst_8 = arith.constant 0x7FC00000 : f32
  %cst_9 = arith.constant 0.000000e+00 : f32
  %0 = ave.hir.vci %cst_9, <INCREASE> : f32, vector<64xf32>
  %1 = ave.hir.pge <ALL> : vector<64xi1>
  %2 = ave.hir.vmuls %0, %cst_8, %1 : vector<64xf32>, f32, vector<64xi1>
  %3 = ave.hir.vadds %2, %cst_8, %1 : vector<64xf32>, f32, vector<64xi1>
  %4 = ave.hir.pge <ALL> : vector<64xi1>
  %5 = ave.hir.broadcast %c2147483647_i32, %4 : i32, vector<64xi1> -> vector<64xi32>
  %6 = ave.hir.pge <ALL> : vector<64xi1>
  %7 = ave.hir.broadcast %cst_0, %6 : f32, vector<64xi1> -> vector<64xf32>
  %8 = ave.hir.pge <ALL> : vector<64xi1>
  %9 = ave.hir.broadcast %cst, %8 : f32, vector<64xi1> -> vector<64xf32>
  %10 = ave.hir.pge <ALL> : vector<64xi1>
  %11 = ave.hir.broadcast %c1_i32, %10 : i32, vector<64xi1> -> vector<64xi32>
  %12 = ave.hir.pge <ALL> : vector<64xi1>
  %13 = ave.hir.broadcast %c65535_i32, %12 : i32, vector<64xi1> -> vector<64xi32>
  %14 = ave.hir.pge <ALL> : vector<64xi1>
  %15 = ave.hir.broadcast %c8388607_i32, %14 : i32, vector<64xi1> -> vector<64xi32>
  %16 = ave.hir.pge <ALL> : vector<64xi1>
  %17 = ave.hir.broadcast %c0_i32, %16 : i32, vector<64xi1> -> vector<64xi32>
  %18 = ave.hir.pge <ALL> : vector<64xi1>
  %19 = ave.hir.pge <ALL> : vector<64xi1>
  %20 = ave.hir.broadcast %c255_i32, %19 : i32, vector<64xi1> -> vector<64xi32>
  scf.for %arg3 = %c0 to %c2048 step %c64 {
    %subview = memref.subview %arg0[%arg3] [64] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_10 = memref.subview %arg2[%arg3] [64] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %21 = vector.bitcast %res : vector<64xf32> to vector<64xi32>
    %22 = ave.hir.pge <ALL> : vector<64xi1>
    %23 = ave.hir.vshrs %21, %c23_i32, %22, %false : vector<64xi32>, i32, vector<64xi1>
    %24 = ave.hir.pge <ALL> : vector<64xi1>
    %25 = ave.hir.vand %23, %20, %24 : vector<64xi32>, vector<64xi1>
    %26 = ave.hir.pge <ALL> : vector<64xi1>
    %27 = ave.hir.vadds %25, %c8_i32, %26 : vector<64xi32>, i32, vector<64xi1>
    %28 = ave.hir.vgather %arg1[%c0] [%27], %18 : memref<320xi32, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xi32>
    %29 = ave.hir.pge <ALL> : vector<64xi1>
    %30 = ave.hir.vadds %25, %c40_i32, %29 : vector<64xi32>, i32, vector<64xi1>
    %31 = ave.hir.vgather %arg1[%c0] [%30], %18 : memref<320xi32, #hivm.address_space<ub>>, vector<64xi32>, vector<64xi1> into vector<64xi32>
    %32 = ave.hir.pge <ALL> : vector<64xi1>
    %33 = ave.hir.vand %21, %15, %32 : vector<64xi32>, vector<64xi1>
    %34 = ave.hir.pge <ALL> : vector<64xi1>
    %35 = ave.hir.vadds %33, %c8388608_i32, %34 : vector<64xi32>, i32, vector<64xi1>
    %36 = ave.hir.pge <ALL> : vector<64xi1>
    %37 = ave.hir.vshrs %35, %c16_i32, %36, %false : vector<64xi32>, i32, vector<64xi1>
    %38 = ave.hir.pge <ALL> : vector<64xi1>
    %39 = ave.hir.vand %28, %13, %38 : vector<64xi32>, vector<64xi1>
    %40 = ave.hir.pge <ALL> : vector<64xi1>
    %41 = ave.hir.vshrs %31, %c16_i32, %40, %false : vector<64xi32>, i32, vector<64xi1>
    %42 = ave.hir.pge <ALL> : vector<64xi1>
    %43 = ave.hir.vmul %37, %39, %42 : vector<64xi32>, vector<64xi1>
    %44 = ave.hir.pge <ALL> : vector<64xi1>
    %45 = ave.hir.vand %43, %13, %44 : vector<64xi32>, vector<64xi1>
    %46 = ave.hir.pge <ALL> : vector<64xi1>
    %47 = ave.hir.vshls %45, %c16_i32, %46, %false : vector<64xi32>, i32, vector<64xi1>
    %48 = ave.hir.pge <ALL> : vector<64xi1>
    %49 = ave.hir.vmul %37, %41, %48 : vector<64xi32>, vector<64xi1>
    %50 = ave.hir.pge <ALL> : vector<64xi1>
    %51 = ave.hir.vadd %49, %47, %50 : vector<64xi32>, vector<64xi1>
    %52 = ave.hir.pge <ALL> : vector<64xi1>
    %53 = ave.hir.vand %35, %13, %52 : vector<64xi32>, vector<64xi1>
    %54 = ave.hir.pge <ALL> : vector<64xi1>
    %55 = ave.hir.vand %31, %13, %54 : vector<64xi32>, vector<64xi1>
    %56 = ave.hir.pge <ALL> : vector<64xi1>
    %57 = ave.hir.vmul %37, %55, %56 : vector<64xi32>, vector<64xi1>
    %58 = ave.hir.pge <ALL> : vector<64xi1>
    %59 = ave.hir.vshrs %57, %c16_i32, %58, %false : vector<64xi32>, i32, vector<64xi1>
    %60 = ave.hir.pge <ALL> : vector<64xi1>
    %61 = ave.hir.vshrs %28, %c16_i32, %60, %false : vector<64xi32>, i32, vector<64xi1>
    %62 = ave.hir.pge <ALL> : vector<64xi1>
    %63 = ave.hir.vmul %53, %61, %62 : vector<64xi32>, vector<64xi1>
    %64 = ave.hir.pge <ALL> : vector<64xi1>
    %65 = ave.hir.vand %63, %13, %64 : vector<64xi32>, vector<64xi1>
    %66 = ave.hir.pge <ALL> : vector<64xi1>
    %67 = ave.hir.vshls %65, %c16_i32, %66, %false : vector<64xi32>, i32, vector<64xi1>
    %68 = ave.hir.pge <ALL> : vector<64xi1>
    %69 = ave.hir.vadd %67, %59, %68 : vector<64xi32>, vector<64xi1>
    %70 = ave.hir.pge <ALL> : vector<64xi1>
    %71 = ave.hir.vmul %53, %39, %70 : vector<64xi32>, vector<64xi1>
    %72 = ave.hir.pge <ALL> : vector<64xi1>
    %73 = ave.hir.vmul %53, %41, %72 : vector<64xi32>, vector<64xi1>
    %74 = ave.hir.pge <ALL> : vector<64xi1>
    %75 = ave.hir.vshrs %73, %c16_i32, %74, %false : vector<64xi32>, i32, vector<64xi1>
    %76 = ave.hir.pge <ALL> : vector<64xi1>
    %77 = ave.hir.vadd %75, %71, %76 : vector<64xi32>, vector<64xi1>
    %78 = ave.hir.pge <ALL> : vector<64xi1>
    %79 = ave.hir.vadd %77, %69, %78 : vector<64xi32>, vector<64xi1>
    %80 = ave.hir.pge <ALL> : vector<64xi1>
    %81 = ave.hir.vadd %79, %51, %80 : vector<64xi32>, vector<64xi1>
    %82 = ave.hir.pge <ALL> : vector<64xi1>
    %83 = ave.hir.vshrs %81, %c31_i32, %82, %false : vector<64xi32>, i32, vector<64xi1>
    %84 = ave.hir.pge <ALL> : vector<64xi1>
    %85 = ave.hir.vand %83, %11, %84 : vector<64xi32>, vector<64xi1>
    %86 = ave.hir.pge <ALL> : vector<64xi1>
    %87 = ave.hir.vcmp <EQ> %85, %17, %86 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %88 = ave.hir.vsel %87, %9, %7 : vector<64xi1>, vector<64xf32>
    %89 = ave.hir.pge <ALL> : vector<64xi1>
    %90 = ave.hir.vand %81, %5, %89 : vector<64xi32>, vector<64xi1>
    %91 = ave.hir.pge <ALL> : vector<64xi1>
    %92 = ave.hir.vsitofp %90, %91 {rnd = #hivm.round_mode<rint>} : vector<64xi32>, vector<64xi1>, vector<64xf32>
    %93 = ave.hir.pge <ALL> : vector<64xi1>
    %94 = ave.hir.vmuls %92, %cst_1, %93 : vector<64xf32>, f32, vector<64xi1>
    %95 = ave.hir.pge <ALL> : vector<64xi1>
    %96 = ave.hir.vmuls %94, %cst_2, %95 : vector<64xf32>, f32, vector<64xi1>
    %97 = ave.hir.pge <ALL> : vector<64xi1>
    %98 = ave.hir.vmuls %96, %cst_0, %97 : vector<64xf32>, f32, vector<64xi1>
    %99 = ave.hir.pge <ALL> : vector<64xi1>
    %100 = ave.hir.vadds %98, %cst_3, %99 : vector<64xf32>, f32, vector<64xi1>
    %101 = ave.hir.pge <ALL> : vector<64xi1>
    %102 = ave.hir.vmul %100, %100, %101 : vector<64xf32>, vector<64xi1>
    %103 = ave.hir.pge <ALL> : vector<64xi1>
    %104 = ave.hir.vmuls %102, %cst_4, %103 : vector<64xf32>, f32, vector<64xi1>
    %105 = ave.hir.pge <ALL> : vector<64xi1>
    %106 = ave.hir.vadds %104, %cst_5, %105 : vector<64xf32>, f32, vector<64xi1>
    %107 = ave.hir.pge <ALL> : vector<64xi1>
    %108 = ave.hir.vmul %106, %102, %107 : vector<64xf32>, vector<64xi1>
    %109 = ave.hir.pge <ALL> : vector<64xi1>
    %110 = ave.hir.vadds %108, %cst_6, %109 : vector<64xf32>, f32, vector<64xi1>
    %111 = ave.hir.pge <ALL> : vector<64xi1>
    %112 = ave.hir.vmul %110, %102, %111 : vector<64xf32>, vector<64xi1>
    %113 = ave.hir.pge <ALL> : vector<64xi1>
    %114 = ave.hir.vadds %112, %cst_7, %113 : vector<64xf32>, f32, vector<64xi1>
    %115 = ave.hir.pge <ALL> : vector<64xi1>
    %116 = ave.hir.vmul %114, %102, %115 : vector<64xf32>, vector<64xi1>
    %117 = ave.hir.pge <ALL> : vector<64xi1>
    %118 = ave.hir.vadds %116, %cst, %117 : vector<64xf32>, f32, vector<64xi1>
    %119 = ave.hir.pge <ALL> : vector<64xi1>
    %120 = ave.hir.vmul %118, %100, %119 : vector<64xf32>, vector<64xi1>
    %121 = ave.hir.pge <ALL> : vector<64xi1>
    %122 = ave.hir.vmul %120, %88, %121 : vector<64xf32>, vector<64xi1>
    %123 = ave.hir.pge <ALL> : vector<64xi1>
    %124 = ave.hir.vcmp <EQ> %25, %20, %123 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %125 = ave.hir.vsel %124, %3, %122 : vector<64xi1>, vector<64xf32>
    %126 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_10[%c0], %126, %125 {hivm.is_continuous} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}

// -----
// CHECK-LABEL:@test_vcmp_16_2VL
func.func @test_vcmp_16_2VL(%arg0: memref<1000xf32, #hivm.address_space<ub>>, %arg1: memref<1000xf32, #hivm.address_space<ub>>, %arg2: memref<1000xf32, #hivm.address_space<ub>>, %arg3: f32, %arg4: f32, %arg5: memref<1000xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %c64 = arith.constant 64 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i16 = arith.constant 0 : i16
  %c1_i16 = arith.constant 1 : i16
  %c0_i32 = arith.constant 0 : i32
  %0 = ave.hir.vci %c0_i32, <INCREASE> : i32, vector<64xi32>
  %1 = ave.hir.pge <ALL> : vector<64xi1>
  %2 = ave.hir.broadcast %c1_i16, %1 : i16, vector<64xi1> -> vector<64xi16>
  %3 = ave.hir.pge <ALL> : vector<64xi1>
  %4 = ave.hir.broadcast %c0_i16, %3 : i16, vector<64xi1> -> vector<64xi16>
  %5 = ave.hir.pge <ALL> : vector<64xi1>
  %6 = ave.hir.broadcast %c1_i32, %5 : i32, vector<64xi1> -> vector<64xi32>
  %7 = ave.hir.scalar_broadcast %arg3 : f32 -> vector<64xf32>
  %8 = ave.hir.scalar_broadcast %arg4 : f32 -> vector<64xf32>
  scf.for %arg6 = %c0 to %c1000 step %c64 {
    %9 = affine.min affine_map<(d0) -> (-d0 + 1000, 64)>(%arg6)
    %subview = memref.subview %arg0[%arg6] [%9] [1] : memref<1000xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg6] [%9] [1] : memref<1000xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg6] [%9] [1] : memref<1000xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg5[%arg6] [%9] [1] : memref<1000xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %9 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %res_3 = ave.hir.vload <NORM> %subview[%c0] : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res_4 = ave.hir.vload <NORM> %subview_0[%c0] : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res_5 = ave.hir.vload <NORM> %subview_1[%c0] : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %10 = ave.hir.vsub %res_3, %res_4, %res : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.vcmp <GT> %10, %7, %res : vector<64xf32>, vector<64xi1> -> vector<64xi1>
    %12 = ave.hir.vsel %11, %2, %4 : vector<64xi1>, vector<64xi16>
    %13 = arith.index_cast %arg6 : index to i32
    %14 = ave.hir.scalar_broadcast %13 : i32 -> vector<64xi32>
    %15 = ave.hir.vadd %14, %0, %res : vector<64xi32>, vector<64xi1>
    %16 = ave.hir.vcmp <LT> %15, %6, %res : vector<64xi32>, vector<64xi1> -> vector<64xi1>
    %17 = ave.hir.vsel %16, %4, %12 : vector<64xi1>, vector<64xi16>
    %18 = ave.hir.vcmp <NE> %17, %4, %res : vector<64xi16>, vector<64xi1> -> vector<64xi1>
    %19 = ave.hir.vsel %18, %8, %res_5 : vector<64xi1>, vector<64xf32>
    ave.hir.masked_store <NORM_B32> %subview_2[%c0], %res, %19 {hivm.is_continuous} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}

// -----
// CHECK-LABEL: @test_vf_with_dangling_vector_of_for_op_res
func.func @test_vf_with_dangling_vector_of_for_op_res(%arg0: memref<2x2xf32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>>, %arg3: memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>>, %arg4: memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = ave.hir.pge <VL2> {mask_op_idx = 0 : i32} : vector<64xi1>
  scf.for %arg5 = %c0 to %c2 step %c1 {
    scf.for %arg6 = %c0 to %c2 step %c1 {
      scf.for %arg7 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg0[%arg6, %arg7] [1, 1] [1, 1] : memref<2x2xf32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        scf.for %arg8 = %c0 to %c2 step %c1 {
          scf.for %arg9 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, %arg8, %arg9, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
            %res = ave.hir.vload <NORM> %subview_0[%c0, %c0, %c0, %c0, %c0] : memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
            %res_2 = ave.hir.vload <BRC_B32> %subview[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x1xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
            %subview_3 = memref.subview %subview_1[0, 0, 0, 0, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
            %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi16>
            %1 = ave.hir.vtrunci %res, false, %0 {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2si>} : vector<64xi16>, vector<64xi8>, vector<64xi1>
            %2 = ave.hir.vsitofp %1, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi8>, vector<64xi1>, vector<64xf16>
            %3 = ave.hir.vextf %2, <part_even>, %0 : vector<64xf16>, vector<64xf32>, vector<64xi1>
            %4 = ave.hir.vmul %3, %res_2, %0 : vector<64xf32>, vector<64xi1>
            %5 = ave.hir.vfptosi %4, <truncwithoverflow>, %0 {sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi32>
            %6 = ave.hir.vtrunci %5, false, %0 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
            %7 = ave.hir.vextsi %6, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi8>, vector<64xi16>, vector<64xi1>
            %8 = ave.hir.vadd %res_4, %7, %0 : vector<64xi16>, vector<64xi1>
            %9 = ave.hir.vsel %0, %8, %7 : vector<64xi1>, vector<64xi16>
            ave.hir.masked_store <NORM_B16> %subview_3[%c0], %0, %9 : memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
          }
        }
      }
    }
  }
  scf.for %arg5 = %c0 to %c2 step %c1 {
    scf.for %arg6 = %c0 to %c2 step %c1 {
      scf.for %arg7 = %c0 to %c2 step %c1 {
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg7, %arg8] [1, 1] [1, 1] : memref<2x2xf32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          scf.for %arg9 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg2[%arg5, %arg6, %arg8, %arg9, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg9, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
            %res = ave.hir.vload <NORM> %subview_0[%c0, %c0, %c0, %c0, %c0] : memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
            %res_2 = ave.hir.vload <BRC_B32> %subview[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x1xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
            %subview_3 = memref.subview %subview_1[0, 0, 0, 0, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
            %res_4 = ave.hir.vload <NORM> %subview_3[%c0] : memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi16>
            %1 = ave.hir.vtrunci %res, false, %0 {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2si>} : vector<64xi16>, vector<64xi8>, vector<64xi1>
            %2 = ave.hir.vsitofp %1, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi8>, vector<64xi1>, vector<64xf16>
            %3 = ave.hir.vextf %2, <part_even>, %0 : vector<64xf16>, vector<64xf32>, vector<64xi1>
            %4 = ave.hir.vmul %3, %res_2, %0 : vector<64xf32>, vector<64xi1>
            %5 = ave.hir.vfptosi %4, <truncwithoverflow>, %0 {sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi32>
            %6 = ave.hir.vtrunci %5, false, %0 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
            %7 = ave.hir.vextsi %6, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi8>, vector<64xi16>, vector<64xi1>
            %8 = ave.hir.vadd %res_4, %7, %0 : vector<64xi16>, vector<64xi1>
            %9 = ave.hir.vsel %0, %8, %7 : vector<64xi1>, vector<64xi16>
            ave.hir.masked_store <NORM_B16> %subview_3[%c0], %0, %9 : memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
          }
        }
      }
    }
  }
  scf.for %arg5 = %c0 to %c2 step %c1 {
    scf.for %arg6 = %c0 to %c2 step %c1 {
      scf.for %arg7 = %c0 to %c2 step %c1 {
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %subview[0, 0, 0, 0, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
          %res = ave.hir.vload <NORM> %subview_0[%c0] : memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xi16>
          %1 = builtin.unrealized_conversion_cast %res : vector<64xi16> to vector<128xi16>
          %2:2 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %res, %arg11 = %1) -> (vector<64xi16>, vector<128xi16>) {
            %4 = builtin.unrealized_conversion_cast %arg11 : vector<128xi16> to vector<64xi16>
            %subview_1 = memref.subview %arg0[%arg8, %arg9] [1, 1] [1, 1] : memref<2x2xf32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg9, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x2x2x2xi16, strided<[128, 64, 32, 16, 1]>, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
            %res_3 = ave.hir.vload <NORM> %subview_2[%c0, %c0, %c0, %c0, %c0] : memref<1x1x1x1x2xi16, strided<[128, 64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
            %res_4 = ave.hir.vload <BRC_B32> %subview_1[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x1xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
            %5 = ave.hir.pge <ALL> : vector<64xi1>
            %6 = ave.hir.vtrunci %res_3, false, %5 {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2si>} : vector<64xi16>, vector<64xi8>, vector<64xi1>
            %7 = ave.hir.pge <ALL> : vector<64xi1>
            %8 = ave.hir.vsitofp %6, %7 {part = #ave.vcvt_part_type<part_even>} : vector<64xi8>, vector<64xi1>, vector<64xf16>
            %9 = ave.hir.pge <ALL> : vector<64xi1>
            %10 = ave.hir.vextf %8, <part_even>, %9 : vector<64xf16>, vector<64xf32>, vector<64xi1>
            %11 = ave.hir.pge <ALL> : vector<64xi1>
            %12 = ave.hir.vmul %10, %res_4, %11 : vector<64xf32>, vector<64xi1>
            %13 = ave.hir.pge <ALL> : vector<64xi1>
            %14 = ave.hir.vfptosi %12, <truncwithoverflow>, %13 {sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi32>
            %15 = ave.hir.pge <ALL> : vector<64xi1>
            %16 = ave.hir.vtrunci %14, false, %15 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
            %17 = ave.hir.pge <ALL> : vector<64xi1>
            %18 = ave.hir.vextsi %16, %17 {part = #ave.vcvt_part_type<part_even>} : vector<64xi8>, vector<64xi16>, vector<64xi1>
            %19 = ave.hir.pge <ALL> : vector<64xi1>
            %20 = ave.hir.vadd %4, %18, %19 : vector<64xi16>, vector<64xi1>
            %21 = ave.hir.vsel %0, %20, %18 : vector<64xi1>, vector<64xi16>
            %22 = builtin.unrealized_conversion_cast %21 : vector<64xi16> to vector<128xi16>
            scf.yield %21, %22 : vector<64xi16>, vector<128xi16>
          }
          %3 = builtin.unrealized_conversion_cast %2#1 : vector<128xi16> to vector<64xi16>
          ave.hir.masked_store <NORM_B16> %subview_0[%c0], %0, %3 : memref<2xi16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
        }
      }
    }
  }
  return
}

// -----
// CHECK-LABEL:@test_multi_dangling_vector
func.func @test_multi_dangling_vector(%arg0: memref<64x40xbf16, #hivm.address_space<ub>>, %arg1: memref<64x20xbf16, #hivm.address_space<ub>>, %arg2: memref<64x40xbf16, #hivm.address_space<ub>>, %arg3: memref<64x20xf32, #hivm.address_space<ub>>, %arg4: memref<20xf32, #hivm.address_space<ub>>, %arg5: memref<20xf32, #hivm.address_space<ub>>, %arg6: memref<64x40xbf16, #hivm.address_space<ub>>, %arg7: memref<64x40xbf16, #hivm.address_space<ub>>, %arg8: memref<1x20xf32, #hivm.address_space<ub>>, %arg9: memref<1x20xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c20 = arith.constant 20 : index
  %c40 = arith.constant 40 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant -1.000000e+00 : f32
  %res, %new_true_shape = ave.hir.plt %c20 : vector<128xi1>, index
  scf.for %arg10 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg10, 0] [1, 40] [1, 1] : memref<64x40xbf16, #hivm.address_space<ub>> to memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>>
    %res_2 = ave.hir.vload <NORM> %subview[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
    %res_3 = ave.hir.vload <NORM> %subview[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
    %res1, %res2 = ave.hir.vdintlv %res_2, %res_3 : vector<128xbf16>, vector<128xbf16>
    ave.hir.masked_store <NORM_B16> %arg1[%arg10, %c0], %res, %res2 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64x20xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
  }
  %res_0, %new_true_shape_1 = ave.hir.plt %c20 {mask_op_idx = 3 : i32} : vector<64xi1>, index
  scf.for %arg10 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg6[%arg10, 0] [1, 40] [1, 1] : memref<64x40xbf16, #hivm.address_space<ub>> to memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg2[%arg10, 0] [1, 40] [1, 1] : memref<64x40xbf16, #hivm.address_space<ub>> to memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>>
    %res_3 = ave.hir.vload <NORM> %subview_2[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
    %res_4 = ave.hir.vload <NORM> %subview_2[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
    %res1, %res2 = ave.hir.vdintlv %res_3, %res_4 : vector<128xbf16>, vector<128xbf16>
    %0 = builtin.unrealized_conversion_cast %res1 : vector<128xbf16> to vector<1x128xbf16>
    %subview_5 = memref.subview %arg3[%arg10, 0] [1, 20] [1, 1] : memref<64x20xf32, #hivm.address_space<ub>> to memref<1x20xf32, strided<[20, 1], offset: ?>, #hivm.address_space<ub>>
    %1 = builtin.unrealized_conversion_cast %0 : vector<1x128xbf16> to vector<1x64xbf16>
    %2 = builtin.unrealized_conversion_cast %1 : vector<1x64xbf16> to vector<64xbf16>
    %3 = ave.hir.vextf %2, <part_even>, %res_0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %4 = builtin.unrealized_conversion_cast %res2 : vector<128xbf16> to vector<1x128xbf16>
    %5 = builtin.unrealized_conversion_cast %4 : vector<1x128xbf16> to vector<1x64xbf16>
    %6 = builtin.unrealized_conversion_cast %5 : vector<1x64xbf16> to vector<64xbf16>
    %7 = ave.hir.vextf %6, <part_even>, %res_0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %res_6 = ave.hir.vload <NORM> %arg4[%c0] : memref<20xf32, #hivm.address_space<ub>> into vector<64xf32>
    %8 = ave.hir.vmuls %res_6, %cst, %res_0 : vector<64xf32>, f32, vector<64xi1>
    %res_7 = ave.hir.vload <NORM> %arg5[%c0] : memref<20xf32, #hivm.address_space<ub>> into vector<64xf32>
    %9 = ave.hir.vmul %7, %8, %res_0 : vector<64xf32>, vector<64xi1>
    %10 = ave.hir.vmuls %9, %cst, %res_0 : vector<64xf32>, f32, vector<64xi1>
    %11 = ave.hir.vmul %3, %res_7, %res_0 : vector<64xf32>, vector<64xi1>
    %12 = ave.hir.vadd %11, %10, %res_0 : vector<64xf32>, vector<64xi1>
    %subview_8 = memref.subview %arg8[0, 0] [1, 20] [1, 1] : memref<1x20xf32, #hivm.address_space<ub>> to memref<20xf32, strided<[1]>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_8[%c0], %res_0, %12 : memref<20xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %13 = ave.hir.vmul %7, %res_7, %res_0 : vector<64xf32>, vector<64xi1>
    %14 = ave.hir.vmul %3, %8, %res_0 : vector<64xf32>, vector<64xi1>
    %15 = ave.hir.vadd %14, %13, %res_0 : vector<64xf32>, vector<64xi1>
    %subview_9 = memref.subview %subview_5[0, 0] [1, 20] [1, 1] : memref<1x20xf32, strided<[20, 1], offset: ?>, #hivm.address_space<ub>> to memref<20xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_9[%c0], %res_0, %15 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<20xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %res_10 = ave.hir.vload <NORM> %arg8[%c0, %c0] : memref<1x20xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_11 = ave.hir.vload <NORM> %subview_5[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x20xf32, strided<[20, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res1_12, %res2_13 = ave.hir.vintlv %res_10, %res_11 : vector<64xf32>, vector<64xf32>
    %16 = ave.hir.pge <ALL> : vector<64xi1>
    %17 = ave.hir.vtruncf %res1_12, <rint>, false, <part_even>, %16 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %res_14, %new_true_shape_15 = ave.hir.plt %c40 {mask_op_idx = 4 : i32} : vector<64xi1>, index
    %subview_16 = memref.subview %subview[0, 0] [1, 40] [1, 1] : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> to memref<40xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B16> %subview_16[%c0], %res_14, %17 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<40xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
    %subview_17 = memref.subview %arg7[%arg10, 0] [1, 40] [1, 1] : memref<64x40xbf16, #hivm.address_space<ub>> to memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %arg0[%arg10, 0] [1, 40] [1, 1] : memref<64x40xbf16, #hivm.address_space<ub>> to memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>>
    %res_19 = ave.hir.vload <NORM> %subview_18[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
    %res_20 = ave.hir.vload <NORM> %subview_18[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> into vector<128xbf16>
    %res1_21, %res2_22 = ave.hir.vdintlv %res_19, %res_20 : vector<128xbf16>, vector<128xbf16>
    %18 = builtin.unrealized_conversion_cast %res1_21 : vector<128xbf16> to vector<1x128xbf16>
    %19 = builtin.unrealized_conversion_cast %18 : vector<1x128xbf16> to vector<1x64xbf16>
    %20 = builtin.unrealized_conversion_cast %19 : vector<1x64xbf16> to vector<64xbf16>
    %21 = ave.hir.vextf %20, <part_even>, %res_0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %subview_23 = memref.subview %arg1[%arg10, 0] [1, 20] [1, 1] : memref<64x20xbf16, #hivm.address_space<ub>> to memref<1x20xbf16, strided<[20, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_24 = memref.subview %subview_23[0, 0] [1, 20] [1, 1] : memref<1x20xbf16, strided<[20, 1], offset: ?>, #hivm.address_space<ub>> to memref<20xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %res_25 = ave.hir.vload <NORM> %subview_24[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<20xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xbf16>
    %22 = ave.hir.vextf %res_25, <part_even>, %res_0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %23 = ave.hir.vmul %22, %8, %res_0 : vector<64xf32>, vector<64xi1>
    %24 = ave.hir.vmuls %23, %cst, %res_0 : vector<64xf32>, f32, vector<64xi1>
    %25 = ave.hir.vmul %21, %res_7, %res_0 : vector<64xf32>, vector<64xi1>
    %26 = ave.hir.vadd %25, %24, %res_0 : vector<64xf32>, vector<64xi1>
    %subview_26 = memref.subview %arg9[0, 0] [1, 20] [1, 1] : memref<1x20xf32, #hivm.address_space<ub>> to memref<20xf32, strided<[1]>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_26[%c0], %res_0, %26 : memref<20xf32, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %27 = ave.hir.vmul %22, %res_7, %res_0 : vector<64xf32>, vector<64xi1>
    %28 = ave.hir.vmul %21, %8, %res_0 : vector<64xf32>, vector<64xi1>
    %29 = ave.hir.vadd %28, %27, %res_0 : vector<64xf32>, vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_9[%c0], %res_0, %29 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<20xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %res_27 = ave.hir.vload <NORM> %arg9[%c0, %c0] : memref<1x20xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_28 = ave.hir.vload <NORM> %subview_5[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<1x20xf32, strided<[20, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res1_29, %res2_30 = ave.hir.vintlv %res_27, %res_28 : vector<64xf32>, vector<64xf32>
    %30 = ave.hir.vtruncf %res1_29, <rint>, false, <part_even>, %res_14 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
    %subview_31 = memref.subview %subview_17[0, 0] [1, 40] [1, 1] : memref<1x40xbf16, strided<[40, 1], offset: ?>, #hivm.address_space<ub>> to memref<40xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B16> %subview_31[%c0], %res_14, %30 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<40xbf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xbf16>
  }
  return
}

// -----
// CHECK-LABEL: @test_vsmin
func.func @test_vsmin(%arg0: memref<2xf32, #hivm.address_space<ub>>, %arg1: memref<i16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant 1.000000e+00 : f16
  %c32767_i16 = arith.constant 32767 : i16
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c32767_i16, %0 : i16, vector<64xi1> -> vector<64xi16>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %cst_1, %2 : f16, vector<64xi1> -> vector<64xf16>
  %4 = ave.hir.pge <ALL> : vector<64xi1>
  %5 = ave.hir.broadcast %cst_0, %4 : f16, vector<64xi1> -> vector<64xf16>
  %6 = ave.hir.pge <ALL> : vector<64xi1>
  %7 = ave.hir.broadcast %cst, %6 : f32, vector<64xi1> -> vector<64xf32>
  %res = ave.hir.vload <BRC_B16> %arg1[] : memref<i16, #hivm.address_space<ub>> into vector<1xi16>
  %8 = builtin.unrealized_conversion_cast %res : vector<1xi16> to vector<i16>
  %9 = ave.hir.pge <VL2> : vector<64xi1>
  %res_2 = ave.hir.vload <NORM> %arg0[%c0] : memref<2xf32, #hivm.address_space<ub>> into vector<64xf32>
  %10 = ave.hir.pge <ALL> : vector<64xi1>
  %11 = ave.hir.vcmp <NE> %res_2, %7, %10 : vector<64xf32>, vector<64xi1> -> vector<64xi1>
  %12 = ave.hir.vsel %11, %3, %5 : vector<64xi1>, vector<64xf16>
  %13 = ave.hir.pge <ALL> : vector<64xi1>
  %14 = ave.hir.vfptosi %12, <trunc>, %13 {sat = true} : vector<64xf16>, vector<64xi1>, vector<64xi16>
  %15 = builtin.unrealized_conversion_cast %8 : vector<i16> to i16
  %16 = ave.hir.vsel %9, %14, %1 : vector<64xi1>, vector<64xi16>
  %17 = ave.hir.pge <ALL> : vector<64xi1>
  %18 = builtin.unrealized_conversion_cast %15 : i16 to vector<1xi16>
  %19 = builtin.unrealized_conversion_cast %18 : vector<1xi16> to vector<64xi16>
  %20 = ave.hir.reduction <min>, %16, %17 : vector<64xi16>, vector<64xi1> -> vector<64xi16>
  %21 = ave.hir.pge <ALL> : vector<64xi1>
  %22 = ave.hir.vsmin %19, %20, %21 : vector<64xi16>, vector<64xi1>
  %23 = builtin.unrealized_conversion_cast %22 : vector<64xi16> to vector<1xi16>
  %24 = builtin.unrealized_conversion_cast %23 : vector<1xi16> to i16
  %25 = builtin.unrealized_conversion_cast %24 : i16 to vector<i16>
  %26 = builtin.unrealized_conversion_cast %25 : vector<i16> to vector<1xi16>
  %27 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B16> %arg1[], %27, %26 : memref<i16, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi16>
  return
}

// -----
// CHECK-LABEL: @test_broadcast_scalar_i1
func.func @test_broadcast_scalar_i1(%arg0: memref<128xi32, #hivm.address_space<ub>>, %arg1: i64, %arg2: memref<128xi1, #hivm.address_space<ub>>, %arg3: i1, %arg4: memref<128xi32, #hivm.address_space<ub>>, %arg5: memref<128xi64, #hivm.address_space<ub>>, %arg6: memref<128xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %1 = ave.hir.broadcast %c1_i32, %0 : i32, vector<64xi1> -> vector<64xi32>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %3 = ave.hir.broadcast %c0_i32, %2 : i32, vector<64xi1> -> vector<64xi32>
  %4 = ave.hir.scalar_broadcast %arg1 : i64 -> vector<64xi64>
  %5 = ave.hir.scalar_broadcast %arg3 : i1 -> vector<64xi1>
  scf.for %arg7 = %c0 to %c128 step %c64 {
    %subview = memref.subview %arg4[%arg7] [64] [1] : memref<128xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview[%c0], %6, %3 {hivm.is_continuous} : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
    %subview_0 = memref.subview %arg0[%arg7] [64] [1] : memref<128xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg5[%arg7] [64] [1] : memref<128xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vextsi %res, %7 {part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi64>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    %10 = ave.hir.vadd %4, %8, %9 : vector<64xi64>, vector<64xi1>
    %11 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B64> %subview_1[%c0], %11, %10 {hivm.is_continuous} : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    %subview_2 = memref.subview %arg2[%arg7] [64] [1] : memref<128xi1, #hivm.address_space<ub>> to memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_3 = memref.subview %arg6[%arg7] [64] [1] : memref<128xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res_4 = ave.hir.vload <NORM> %subview_2[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi1>
    %12 = ave.hir.pge <ALL> : vector<64xi1>
    %13 = ave.hir.preg.and <b8> %res_4, %5, %12 : vector<64xi1>
    %14 = ave.hir.vsel %13, %1, %3 : vector<64xi1>, vector<64xi32>
    %15 = ave.hir.pge <ALL> : vector<64xi1>
    %16 = ave.hir.vtrunci %14, false, %15 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
    %17 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B8> %subview_3[%c0], %17, %16 {hivm.is_continuous} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
  }
  return
}