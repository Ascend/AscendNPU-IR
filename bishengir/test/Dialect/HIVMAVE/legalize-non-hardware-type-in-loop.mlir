// RUN: bishengir-opt -hivm-legalize-loop-iter-arg -scf-for-loop-canonicalization -cse -canonicalize -split-input-file %s | FileCheck %s

func.func @entry_outlined_vf_1(%arg0: memref<?x128xf16, #hivm.address_space<ub>>, %arg1: memref<?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: memref<?x128xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %arg4 = %c0 to %arg2 step %c1 {
    %subview = memref.subview %arg0[%arg4, 0] [1, 128] [1, 1] : memref<?x128xf16, #hivm.address_space<ub>> to memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg4] [1] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_1[%c0], %cst {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
    // CHECK: unrealized_conversion_cast
    // CHECK-NOT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<1xf32>)
    %1 = scf.for %arg5 = %c0 to %c128 step %c64 iter_args(%arg6 = %0) -> (vector<1xf32>) {
      %subview_2 = memref.subview %subview[0, %arg5] [1, 64] [1, 1] : memref<1x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview_2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf16>
      %3 = arith.extf %2 : vector<1x64xf16> to vector<1x64xf32>
      %4 = vector.multi_reduction <add>, %3, %arg6 [1] : vector<1x64xf32> to vector<1xf32>
      // CHECK: unrealized_conversion_cast
      // CHECK-NOT: scf.yield %{{.*}} : vector<1xf32>
      scf.yield %4 : vector<1xf32>
    }
    // CHECK: unrealized_conversion_cast
    vector.transfer_write %1, %subview_1[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg5 = %c0 to %c128 step %c64 {
      %subview_2 = memref.subview %arg3[%arg4, %arg5] [1, 64] [1, 1] : memref<?x128xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview_1[%c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %3 = arith.truncf %2 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %3, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----
// CHECK: builtin.unrealized_conversion_cast %[[FOR_RES:.*]]#1 : vector<64xf32> to vector<f32>
func.func @fused_sigmoid_gating_delta_rule_update_kernel_outlined_vf_4(%arg0: memref<128xbf16, #hivm.address_space<ub>>, %arg1: memref<128xbf16, #hivm.address_space<ub>>, %arg2: memref<f32, #hivm.address_space<ub>>, %arg3: memref<f32, #hivm.address_space<ub>>, %arg4: memref<128xf32, #hivm.address_space<ub>>, %arg5: memref<128xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %res = ave.hir.vload <BRC_B32> %arg2[] : memref<f32, #hivm.address_space<ub>> into vector<1xf32>
  %0 = builtin.unrealized_conversion_cast %res : vector<1xf32> to vector<f32>
  %res_0 = ave.hir.vload <BRC_B32> %arg3[] : memref<f32, #hivm.address_space<ub>> into vector<1xf32>
  %1 = builtin.unrealized_conversion_cast %res_0 : vector<1xf32> to vector<f32>
  %2:2 = scf.for %arg6 = %c0 to %c128 step %c64 iter_args(%arg7 = %0, %arg8 = %1) -> (vector<f32>, vector<f32>) {
    %subview = memref.subview %arg0[%arg6] [64] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg4[%arg6] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res_2 = ave.hir.vload <NORM> %subview[%c0] : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xbf16>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vextf %res_2, <part_even>, %7 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_1[%c0], %9, %8 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %10 = ave.hir.pge <ALL> : vector<64xi1>
    %11 = ave.hir.vmul %8, %8, %10 : vector<64xf32>, vector<64xi1>
    %12 = builtin.unrealized_conversion_cast %arg7 : vector<f32> to f32
    %13 = ave.hir.pge <ALL> : vector<64xi1>
    %14 = builtin.unrealized_conversion_cast %12 : f32 to vector<1xf32>
    %15 = builtin.unrealized_conversion_cast %14 : vector<1xf32> to vector<64xf32>
    %16 = ave.hir.reduction <add>, %11, %13 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %17 = ave.hir.pge <ALL> : vector<64xi1>
    %18 = ave.hir.vadd %15, %16, %17 : vector<64xf32>, vector<64xi1>
    %19 = builtin.unrealized_conversion_cast %18 : vector<64xf32> to vector<1xf32>
    %20 = builtin.unrealized_conversion_cast %19 : vector<1xf32> to f32
    %21 = builtin.unrealized_conversion_cast %20 : f32 to vector<f32>
    %subview_3 = memref.subview %arg1[%arg6] [64] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %arg5[%arg6] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res_5 = ave.hir.vload <NORM> %subview_3[%c0] : memref<64xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xbf16>
    %22 = ave.hir.pge <ALL> : vector<64xi1>
    %23 = ave.hir.vextf %res_5, <part_even>, %22 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
    %24 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_4[%c0], %24, %23 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %25 = ave.hir.pge <ALL> : vector<64xi1>
    %26 = ave.hir.vmul %23, %23, %25 : vector<64xf32>, vector<64xi1>
    %27 = builtin.unrealized_conversion_cast %arg8 : vector<f32> to f32
    %28 = ave.hir.pge <ALL> : vector<64xi1>
    %29 = builtin.unrealized_conversion_cast %27 : f32 to vector<1xf32>
    %30 = builtin.unrealized_conversion_cast %29 : vector<1xf32> to vector<64xf32>
    %31 = ave.hir.reduction <add>, %26, %28 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %32 = ave.hir.pge <ALL> : vector<64xi1>
    %33 = ave.hir.vadd %30, %31, %32 : vector<64xf32>, vector<64xi1>
    %34 = builtin.unrealized_conversion_cast %33 : vector<64xf32> to vector<1xf32>
    %35 = builtin.unrealized_conversion_cast %34 : vector<1xf32> to f32
    %36 = builtin.unrealized_conversion_cast %35 : f32 to vector<f32>
    scf.yield %21, %36 : vector<f32>, vector<f32>
  } {element_alignment_bit_width = 32 : i32}
  %3 = builtin.unrealized_conversion_cast %2#1 : vector<f32> to vector<1xf32>
  %4 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %arg3[], %4, %3 : memref<f32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  %5 = builtin.unrealized_conversion_cast %2#0 : vector<f32> to vector<1xf32>
  %6 = ave.hir.pge <ALL> : vector<1xi1>
  ave.hir.masked_store <ONEPT_B32> %arg2[], %6, %5 : memref<f32, #hivm.address_space<ub>>, vector<1xi1>, vector<1xf32>
  return
}