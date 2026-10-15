// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 \
// RUN: -convert-hivmave-to-ave-intrin -canonicalize -cse | FileCheck %s --implicit-check-not=_packed_stores --implicit-check-not=hivm.pdintlv --implicit-check-not=hivm.psts --implicit-check-not=hivm.ppack --implicit-check-not=hivm.pstu
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 \
// RUN: -convert-hivmave-to-ave-intrin -convert-hivmave-to-ave-intrin \
// RUN: -canonicalize -cse | FileCheck %s --implicit-check-not=_packed_stores --implicit-check-not=hivm.pdintlv --implicit-check-not=hivm.psts --implicit-check-not=hivm.ppack --implicit-check-not=hivm.pstu

// The complete original VF loop, including both vsel users and intervening
// f32 loads/stores. The caller retains the planned UB addresses and double
// buffering from bwd_qkv_kernel. Input i8 and output i1 SHARE the same storage;
// replacing the output with a fresh allocation would miss this regression.
// Outlined VFs are only called by their kernel. Optimize the original VF
// in place after proving all call sites, preserving its name and call target.
// CHECK-DAG: #[[$ROWOFFSET:map[0-9]*]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK-LABEL: func.func @bwd_qkv_kernel_mix_aiv_outlined_vf_3(
// CHECK-DAG: %[[PACK:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-DAG: %[[DIVISOR:.*]] = llvm.mlir.constant(8 : index)
// CHECK: scf.for
// CHECK-NOT: hivm.pstu
// CHECK: %[[FIRST:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[ROW:.*]] = affine.apply #[[$ROWOFFSET]]
// CHECK: "hivm_regbaseintrins.intr.hivm.vsel"({{.*}}, %[[FIRST]])
// CHECK: %[[SECOND:.*]] = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"
// CHECK: %[[PAIR:.*]] = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%[[FIRST]], %[[SECOND]])
// CHECK: %[[EVEN:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK: to offset: [%[[ROW]]], sizes: [128], strides: [1]
// CHECK: llvm.udiv {{.*}}, %[[DIVISOR]]
// CHECK: "hivm_regbaseintrins.intr.hivm.psts.b8"(%[[EVEN]], {{.*}}, {{.*}}, %[[PACK]], {{.*}})
// CHECK-NOT: hivm.psts
// CHECK-NOT: hivm.pstu
// CHECK-NOT: hivm.vstas
// CHECK: "hivm_regbaseintrins.intr.hivm.vsel"({{.*}}, %[[SECOND]])
// CHECK-NOT: hivm.psts
// CHECK-NOT: hivm.pstu
// CHECK: return
#map12 = affine_map<(d0)[s0] -> (d0 + s0)>
#map13 = affine_map<()[s0] -> (s0 * 128)>
#map14 = affine_map<()[s0] -> (s0 * 256)>
#map15 = affine_map<()[s0] -> (s0 * 16)>
#map16 = affine_map<(d0, d1)[s0] -> (d0 * 1040 + d1 + s0)>
#map17 = affine_map<()[s0] -> (s0 * 128 + 64)>
#map18 = affine_map<()[s0] -> (s0 * 256 + 64)>
  func.func @bwd_qkv_kernel_mix_aiv_outlined_vf_3(%arg0: memref<64x1xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xi8, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>, %arg3: f32, %arg4: memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, %arg5: memref<64x128xf32, #hivm.address_space<ub>>, %arg6: memref<64x128xi1, strided<[256, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c1040 = arith.constant 1040 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.44269502 : f32
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant 0.693147182 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst_1, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.broadcast %c0_i8, %2 : i8, vector<64xi1> -> vector<64xi8>
    scf.for %arg7 = %c0 to %c64 step %c1 {
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<64x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg7], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      %res = ave.hir.vload <BRC_B32> %reinterpret_cast[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %5 = ave.hir.vmuls %res, %cst, %4 : vector<64xf32>, f32, vector<64xi1>
      %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg1 : memref<64x128xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index
      %6 = affine.apply #map13()[%arg7]
      %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [%6], sizes: [64], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<64xi8, #map12, #hivm.address_space<ub>>
      %res_7 = ave.hir.vload <UNPK4_B8> %reinterpret_cast_6[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, #map12, #hivm.address_space<ub>> into vector<64xi8>
      %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %8 = ave.hir.vcmp <NE> %res_7, %3, %7 : vector<64xi8>, vector<64xi1> -> vector<64xi1>
      %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %arg6 : memref<64x128xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
      %9 = affine.apply #map14()[%arg7]
      %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [%9], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map12, #hivm.address_space<ub>>
      %10 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B8> %reinterpret_cast_12[%c0], %10, %8 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map12, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
      %base_buffer_13, %offset_14, %sizes_15:2, %strides_16:2 = memref.extract_strided_metadata %arg2 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %11 = affine.apply #map13()[%arg7]
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [%11], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map12, #hivm.address_space<ub>>
      %res_18 = ave.hir.vload <NORM> %reinterpret_cast_17[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map12, #hivm.address_space<ub>> into vector<64xf32>
      %12 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %13 = ave.hir.vmuls %res_18, %arg3, %12 : vector<64xf32>, f32, vector<64xi1>
      %14 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %15 = ave.hir.vsub %13, %5, %14 : vector<64xf32>, vector<64xi1>
      %16 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %17 = ave.hir.vmuls %15, %cst_0, %16 : vector<64xf32>, f32, vector<64xi1>
      %18 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %19 = ave.hir.vexp %17, %18 : vector<64xf32>, vector<64xi1>
      %20 = ave.hir.vsel %8, %19, %1 : vector<64xi1>, vector<64xf32>
      %base_buffer_19, %offset_20, %sizes_21:2, %strides_22:2 = memref.extract_strided_metadata %arg5 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %21 = affine.apply #map13()[%arg7]
      %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer_19 to offset: [%21], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map12, #hivm.address_space<ub>>
      %22 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast_23[%c0], %22, %20 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map12, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %23 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %24 = ave.hir.vtruncf %20, <rint>, false, <part_even>, %23 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xbf16>, vector<64xi1>
      %base_buffer_24, %offset_25, %sizes_26:3, %strides_27:3 = memref.extract_strided_metadata %arg4 : memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %25 = affine.apply #map15()[%arg7]
      %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [%25], sizes: [4, 16], strides: [1040, 1] : memref<bf16, #hivm.address_space<ub>> to memref<4x16xbf16, #map16, #hivm.address_space<ub>>
      %base_buffer_29, %offset_30, %sizes_31:2, %strides_32:2 = memref.extract_strided_metadata %arg1 : memref<64x128xi8, #hivm.address_space<ub>> -> memref<i8, #hivm.address_space<ub>>, index, index, index, index, index
      %26 = affine.apply #map17()[%arg7]
      %reinterpret_cast_33 = memref.reinterpret_cast %base_buffer_29 to offset: [%26], sizes: [64], strides: [1] : memref<i8, #hivm.address_space<ub>> to memref<64xi8, #map12, #hivm.address_space<ub>>
      %res_34 = ave.hir.vload <UNPK4_B8> %reinterpret_cast_33[%c0] {functionType = #ave.func_dist_type<pk4>} : memref<64xi8, #map12, #hivm.address_space<ub>> into vector<64xi8>
      %27 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %28 = ave.hir.vcmp <NE> %res_34, %3, %27 : vector<64xi8>, vector<64xi1> -> vector<64xi1>
      %base_buffer_35, %offset_36, %sizes_37:2, %strides_38:2 = memref.extract_strided_metadata %arg6 : memref<64x128xi1, strided<[256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index
      %29 = affine.apply #map18()[%arg7]
      %reinterpret_cast_39 = memref.reinterpret_cast %base_buffer_35 to offset: [%29], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map12, #hivm.address_space<ub>>
      %30 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B8> %reinterpret_cast_39[%c0], %30, %28 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map12, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
      %base_buffer_40, %offset_41, %sizes_42:2, %strides_43:2 = memref.extract_strided_metadata %arg2 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %31 = affine.apply #map17()[%arg7]
      %reinterpret_cast_44 = memref.reinterpret_cast %base_buffer_40 to offset: [%31], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map12, #hivm.address_space<ub>>
      %res_45 = ave.hir.vload <NORM> %reinterpret_cast_44[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map12, #hivm.address_space<ub>> into vector<64xf32>
      %32 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %33 = ave.hir.vmuls %res_45, %arg3, %32 : vector<64xf32>, f32, vector<64xi1>
      %34 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %35 = ave.hir.vsub %33, %5, %34 : vector<64xf32>, vector<64xi1>
      %36 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %37 = ave.hir.vmuls %35, %cst_0, %36 : vector<64xf32>, f32, vector<64xi1>
      %38 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %39 = ave.hir.vexp %37, %38 : vector<64xf32>, vector<64xi1>
      %40 = ave.hir.vsel %28, %39, %1 : vector<64xi1>, vector<64xf32>
      %base_buffer_46, %offset_47, %sizes_48:2, %strides_49:2 = memref.extract_strided_metadata %arg5 : memref<64x128xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index
      %41 = affine.apply #map17()[%arg7]
      %reinterpret_cast_50 = memref.reinterpret_cast %base_buffer_46 to offset: [%41], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map12, #hivm.address_space<ub>>
      %42 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast_50[%c0], %42, %40 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map12, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %43 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %44 = ave.hir.vtruncf %40, <rint>, false, <part_even>, %43 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xbf16>, vector<64xi1>
      %45 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      %res1, %res2 = ave.hir.vdintlv %24, %44 {layout_change = #ave<layout_change DENSE>} : vector<64xbf16>, vector<128xbf16>
      ave.hir.store_with_stride %reinterpret_cast_28[%c0, %c0], %c1040, %45, %res1 {functionType = #ave.func_dist_type<norm>} : memref<4x16xbf16, #map16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
    }
    return
  }

// CHECK-LABEL: func.func @original_caller(
// CHECK: call @bwd_qkv_kernel_mix_aiv_outlined_vf_3(
// CHECK: return
func.func @original_caller(%ping: i1, %scale: f32) {
  %c0 = arith.constant 0 : i64
  %c8192 = arith.constant 8192 : i64
  %c16384 = arith.constant 16384 : i64
  %c16640 = arith.constant 16640 : i64
  %c16896 = arith.constant 16896 : i64
  %c50688 = arith.constant 50688 : i64
  %c181760 = arith.constant 181760 : i64
  %c214528 = arith.constant 214528 : i64
  %a0 = hivm.hir.pointer_cast(%c16384) : memref<64xf32, #hivm.address_space<ub>>
  %a1 = hivm.hir.pointer_cast(%c16640) : memref<64xf32, #hivm.address_space<ub>>
  %a = arith.select %ping, %a1, %a0 : memref<64xf32, #hivm.address_space<ub>>
  %abase, %ao, %asz, %ast = memref.extract_strided_metadata %a : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
  %arg0 = memref.reinterpret_cast %abase to offset: [0], sizes: [64, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<64x1xf32, #hivm.address_space<ub>>
  %b0 = hivm.hir.pointer_cast(%c0) : memref<64x128xi8, #hivm.address_space<ub>>
  %b1 = hivm.hir.pointer_cast(%c8192) : memref<64x128xi8, #hivm.address_space<ub>>
  %arg1 = arith.select %ping, %b1, %b0 : memref<64x128xi8, #hivm.address_space<ub>>
  %f0 = hivm.hir.pointer_cast(%c214528) : memref<64x128xf32, #hivm.address_space<ub>>
  %f1 = hivm.hir.pointer_cast(%c181760) : memref<64x128xf32, #hivm.address_space<ub>>
  %arg2 = arith.select %ping, %f1, %f0 : memref<64x128xf32, #hivm.address_space<ub>>
  %arg4 = hivm.hir.pointer_cast(%c16896) : memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>
  %arg5 = hivm.hir.pointer_cast(%c50688) : memref<64x128xf32, #hivm.address_space<ub>>
  %d0 = hivm.hir.pointer_cast(%c0) : memref<64x256x1xi1, #hivm.address_space<ub>>
  %d1 = hivm.hir.pointer_cast(%c8192) : memref<64x256x1xi1, #hivm.address_space<ub>>
  %d = arith.select %ping, %d1, %d0 : memref<64x256x1xi1, #hivm.address_space<ub>>
  %dbase, %do, %dsz:3, %dst:3 = memref.extract_strided_metadata %d : memref<64x256x1xi1, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
  %arg6 = memref.reinterpret_cast %dbase to offset: [0], sizes: [64, 128], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<64x128xi1, strided<[256, 1]>, #hivm.address_space<ub>>
  func.call @bwd_qkv_kernel_mix_aiv_outlined_vf_3(%arg0, %arg1, %arg2, %scale, %arg4, %arg5, %arg6) : (memref<64x1xf32, #hivm.address_space<ub>>, memref<64x128xi8, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, f32, memref<8x64x16xbf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xi1, strided<[256, 1]>, #hivm.address_space<ub>>) -> ()
  return
}
