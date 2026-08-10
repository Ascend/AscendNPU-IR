// RUN: bishengir-opt -hivm-enable-stride-align %s | FileCheck %s

// Regression test: EnableStrideAlign must run to completion (no crash/assert)
// on a real median kernel consisting of 43 outlined vector functions operating
// on strided subviews of UB memrefs. The pass introduces explicit strided
// memref argument types and the `hivm.storage_aligned` function attribute.
// This test only guards against crashes / silent skips: it asserts the pass
// produces output and applies the attribute on the first function.

// CHECK-LABEL: func.func @median_small_flat_kernel_fused_0_outlined_vf_0
// CHECK-SAME: strided<[256, 128, 64, 32, 16, 1]>
// CHECK-SAME: hivm.storage_aligned
// CHECK-NOT: unrealized_conversion_cast

#map = affine_map<(d0) -> (0, d0)>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map3 = affine_map<(d0, d1) -> (0, d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map5 = affine_map<(d0, d1) -> (0, d0, d1, 0)>
#map6 = affine_map<(d0, d1) -> (0, d0, 0, d1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map8 = affine_map<(d0, d1) -> (0, d0, d1, 0, 0)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>
#map11 = affine_map<(d0, d1) -> (0, d0, 0, d1, 0)>
#map12 = affine_map<(d0, d1) -> (0, d0, 0, d1, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, 0, d4, d5)>
#map14 = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map16 = affine_map<(d0, d1) -> (d0, 0, d1, 0, 0)>
#map17 = affine_map<(d0, d1) -> (d0, 0, d1, 0)>
#map18 = affine_map<(d0, d1) -> (d0, 0, d1)>
#map19 = affine_map<(d0) -> (d0, 0, 0)>
#map20 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map21 = affine_map<(d0) -> (0, d0, 0, 0)>
#map22 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map23 = affine_map<(d0) -> (0, d0, 0)>
#map24 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @median_small_flat_kernel_fused_0_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : vector<1x1x1x1x1x64xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x64xi1>
    %1 = vector.constant_mask [2] : vector<64xi1>
    %2 = vector.transfer_read %arg0[%c0], %c0_i32, %1 {in_bounds = [true, true], permutation_map = #map} : memref<2xi32, #hivm.address_space<ub>>, vector<1x64xi32>
    %3 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            scf.for %arg7 = %c0 to %c2 step %c1 {
              %subview_1 = memref.subview %arg1[%arg3, %arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              vector.transfer_write %cst, %subview_1[%c0, %c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x1x64xi16>, memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            }
          }
        }
      }
      %subview = memref.subview %arg0[%arg3] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg2[%arg3, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %4 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true, true], permutation_map = #map1} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi32>
      %5 = arith.xori %4, %2 : vector<1x64xi32>
      vector.transfer_write %5, %subview_0[%c0, %c0], %3 {in_bounds = [true, true]} : vector<1x64xi32>, memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @median_small_flat_kernel_fused_0_outlined_vf_1(%arg0: memref<32x2x1xi16, #hivm.address_space<ub>>, %arg1: memref<32x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>, %arg3: memref<32x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<32x2xi16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x64xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c32 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview_0 = memref.subview %arg0[%arg5, %arg6, 0] [1, 1, 1] [1, 1, 1] : memref<32x2x1xi16, #hivm.address_space<ub>> to memref<1x1x1xi16, strided<[2, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg1[%arg5, %arg6, 0] [1, 1, 2] [1, 1, 1] : memref<32x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg2[%arg6, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_3 = memref.subview %arg3[%arg5, %arg6, 0] [1, 1, 2] [1, 1, 1] : memref<32x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %2 = vector.transfer_read %subview_1[%c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %3 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true], permutation_map = #map2} : memref<1x1x1xi16, strided<[2, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %4 = vector.transfer_read %subview_2[%c0, %c0], %c0_i32, %1 {in_bounds = [true, true, true], permutation_map = #map3} : memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
        %5 = arith.xori %2, %3 : vector<1x1x64xi16>
        %6 = arith.cmpi sgt, %2, %5 : vector<1x1x64xi16>
        %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xi1> to vector<1x1x64xi32>
        %8 = arith.cmpi ne, %7, %4 : vector<1x1x64xi32>
        %9 = arith.select %8, %5, %2 : vector<1x1x64xi1>, vector<1x1x64xi16>
        vector.transfer_write %9, %subview_3[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : vector<1x1x64xi16>, memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
      }
      %subview = memref.subview %arg4[%arg5, 0] [1, 2] [1, 1] : memref<32x2xi16, #hivm.address_space<ub>> to memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xi16>, memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @median_small_flat_kernel_fused_0_outlined_vf_2(%arg0: memref<32x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<32x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x128xi1>
    %2 = vector.shape_cast %0 : vector<1x1x128xi1> to vector<1x128xi1>
    scf.for %arg2 = %c0 to %c32 step %c1 {
      %subview = memref.subview %arg1[%arg2, 0] [1, 2] [1, 1] : memref<32x2xi16, #hivm.address_space<ub>> to memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %3 = vector.transfer_read %subview[%c0, %c0], %c0_i16, %1 {in_bounds = [true, true]} : memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xi16>
      %4 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %3) -> (vector<1x128xi16>) {
        %subview_0 = memref.subview %arg0[%arg2, %arg3, 0] [1, 1, 2] [1, 1, 1] : memref<32x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = arith.select %1, %arg4, %cst : vector<1x128xi1>, vector<1x128xi16>
        %7 = vector.shape_cast %5 : vector<1x1x128xi16> to vector<1x128xi16>
        %8 = arith.xori %6, %7 : vector<1x128xi16>
        %9 = arith.select %2, %8, %7 : vector<1x128xi1>, vector<1x128xi16>
        scf.yield %9 : vector<1x128xi16>
      }
      vector.transfer_write %4, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x128xi16>, memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @median_small_flat_kernel_fused_1_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<16x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<16x2x1x2xi16, #hivm.address_space<ub>>, %arg3: memref<16x2x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    scf.for %arg4 = %c0 to %c16 step %c1 {
      scf.for %arg5 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg2[%arg4, %arg5, 0, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x1x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[4, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %2 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map4} : memref<1x1x1x2xi16, strided<[4, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg0[%arg5, %arg6] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg1[%arg4, %arg5, %arg6, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg4, %arg5, %arg6, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %3 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %4 = vector.transfer_read %subview_0[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map5} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
          %5 = arith.xori %3, %2 : vector<1x1x1x64xi16>
          %6 = arith.cmpi sgt, %3, %5 : vector<1x1x1x64xi16>
          %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %8 = arith.cmpi ne, %7, %4 : vector<1x1x1x64xi32>
          %9 = arith.select %8, %5, %3 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %9, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_1_outlined_vf_1(%arg0: memref<16x2x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<16x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    scf.for %arg2 = %c0 to %c16 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg1[%arg2, %arg3, %arg4] [1, 1, 1] [1, 1, 1] : memref<16x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %1 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %2 = arith.select %0, %1, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
          %3 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true]} : memref<1x1x1xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1xi16>
          %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [3] : vector<1x1x1x128xi16> to vector<1x1x1xi16>
          vector.transfer_write %4, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x1xi16>, memref<1x1x1xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_2_outlined_vf_0(%arg0: memref<16x2x2x1xi16, #hivm.address_space<ub>>, %arg1: memref<16x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>, %arg3: memref<16x2x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<16x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg2[%arg6, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32, %1 {in_bounds = [true, true, true, true], permutation_map = #map6} : memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg7, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<16x2x2x1xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true], permutation_map = #map7} : memref<1x1x1x1xi16, strided<[4, 2, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg4[%arg5, %arg7, 0] [1, 1, 2] [1, 1, 1] : memref<16x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
          %7 = vector.shape_cast %5 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %8 = arith.xori %6, %7 : vector<1x1x128xi16>
          %9 = arith.select %4, %8, %7 : vector<1x1x128xi1>, vector<1x1x128xi16>
          vector.transfer_write %9, %subview_0[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_3_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<8x2x1x2x2xi16, #hivm.address_space<ub>>, %arg3: memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<8x2x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c8 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg6, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map8} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, 0, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x1x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[8, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map9} : memref<1x1x1x1x2xi16, strided<[8, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %8 = arith.xori %6, %7 : vector<1x1x1x1x64xi16>
            %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x64xi16>
            %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x64xi32>
            %12 = arith.select %11, %8, %6 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x128xi1> to vector<1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c8 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<8x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %5) -> (vector<1x1x1x128xi16>) {
            %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %8 = arith.select %3, %arg9, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            %9 = vector.shape_cast %7 : vector<1x1x1x1x128xi16> to vector<1x1x1x128xi16>
            %10 = arith.xori %8, %9 : vector<1x1x1x128xi16>
            %11 = arith.select %4, %10, %9 : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            scf.yield %11 : vector<1x1x1x128xi16>
          }
          vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xi16>, memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_4_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<8x2x2x1x2xi16, #hivm.address_space<ub>>, %arg3: memref<8x2x2x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c2 step %c1 {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg2[%arg4, %arg5, %arg6, 0, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x1x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[8, 4, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %2 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map10} : memref<1x1x1x1x2xi16, strided<[8, 4, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg0[%arg5, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg1[%arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %3 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %4 = vector.transfer_read %subview_0[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map11} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
            %5 = arith.xori %3, %2 : vector<1x1x1x1x64xi16>
            %6 = arith.cmpi sgt, %3, %5 : vector<1x1x1x1x64xi16>
            %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %8 = arith.cmpi ne, %7, %4 : vector<1x1x1x1x64xi32>
            %9 = arith.select %8, %5, %3 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %9, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_4_outlined_vf_1(%arg0: memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<8x2x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    scf.for %arg2 = %c0 to %c8 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          scf.for %arg5 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<8x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %1 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %2 = arith.select %0, %1, %cst : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
            %3 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true]} : memref<1x1x1x1xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1xi16>
            %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [4] : vector<1x1x1x1x128xi16> to vector<1x1x1x1xi16>
            vector.transfer_write %4, %subview[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xi16>, memref<1x1x1x1xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_5_outlined_vf_0(%arg0: memref<8x2x4x1xi16, #hivm.address_space<ub>>, %arg1: memref<8x2x4x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>, %arg3: memref<8x2x4x2xi16, #hivm.address_space<ub>>, %arg4: memref<8x4x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c8 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg2[%arg6, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32, %1 {in_bounds = [true, true, true, true], permutation_map = #map6} : memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c4 step %c1 {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg7, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<8x2x4x1xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[8, 4, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<8x2x4x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<8x2x4x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true], permutation_map = #map7} : memref<1x1x1x1xi16, strided<[8, 4, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c8 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          %subview = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<8x2x4x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg4[%arg5, %arg7, 0] [1, 1, 2] [1, 1, 1] : memref<8x4x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[8, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[8, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
          %7 = vector.shape_cast %5 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %8 = arith.xori %6, %7 : vector<1x1x128xi16>
          %9 = arith.select %4, %8, %7 : vector<1x1x128xi1>, vector<1x1x128xi16>
          vector.transfer_write %9, %subview_0[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x2xi16, strided<[8, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_6_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<4x2x2x2x4xi16, #hivm.address_space<ub>>, %arg2: memref<4x2x1x2x4xi16, #hivm.address_space<ub>>, %arg3: memref<4x2x2x2x4xi16, #hivm.address_space<ub>>, %arg4: memref<4x2x2x4xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 4] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 4] : vector<1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg6, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map8} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<4x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, 0, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<4x2x1x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[16, 8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<4x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map9} : memref<1x1x1x1x4xi16, strided<[16, 8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %8 = arith.xori %6, %7 : vector<1x1x1x1x64xi16>
            %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x64xi16>
            %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x64xi32>
            %12 = arith.select %11, %8, %6 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 4] : vector<1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 4] : vector<1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x128xi1> to vector<1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<4x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true]} : memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %5) -> (vector<1x1x1x128xi16>) {
            %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<4x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %8 = arith.select %3, %arg9, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            %9 = vector.shape_cast %7 : vector<1x1x1x1x128xi16> to vector<1x1x1x128xi16>
            %10 = arith.xori %8, %9 : vector<1x1x1x128xi16>
            %11 = arith.select %4, %10, %9 : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            scf.yield %11 : vector<1x1x1x128xi16>
          }
          vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xi16>, memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_7_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<4x2x2x1x2x2xi16, #hivm.address_space<ub>>, %arg3: memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<4x2x2x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg0[%arg6, %arg8] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true, true], permutation_map = #map12} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi32>
            scf.for %arg9 = %c0 to %c2 step %c1 {
              %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %subview_1 = memref.subview %arg2[%arg5, %arg6, %arg7, 0, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<4x2x2x1x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[16, 8, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi16>
              %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true, true], permutation_map = #map13} : memref<1x1x1x1x1x2xi16, strided<[16, 8, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi16>
              %8 = arith.xori %6, %7 : vector<1x1x1x1x1x64xi16>
              %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x1x64xi16>
              %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x1x64xi1> to vector<1x1x1x1x1x64xi32>
              %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x1x64xi32>
              %12 = arith.select %11, %8, %6 : vector<1x1x1x1x1x64xi1>, vector<1x1x1x1x1x64xi16>
              vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x1x64xi16>, memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            }
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x1x128xi1> to vector<1x1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg4[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<4x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %6 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %5) -> (vector<1x1x1x1x128xi16>) {
              %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x2xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x128xi16>
              %8 = arith.select %3, %arg10, %cst : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
              %9 = vector.shape_cast %7 : vector<1x1x1x1x1x128xi16> to vector<1x1x1x1x128xi16>
              %10 = arith.xori %8, %9 : vector<1x1x1x1x128xi16>
              %11 = arith.select %4, %10, %9 : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
              scf.yield %11 : vector<1x1x1x1x128xi16>
            }
            vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x128xi16>, memref<1x1x1x1x2xi16, strided<[16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_8_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<4x2x4x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<4x2x4x1x2xi16, #hivm.address_space<ub>>, %arg3: memref<4x2x4x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    scf.for %arg4 = %c0 to %c4 step %c1 {
      scf.for %arg5 = %c0 to %c2 step %c1 {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          %subview = memref.subview %arg2[%arg4, %arg5, %arg6, 0, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<4x2x4x1x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[16, 8, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %2 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map10} : memref<1x1x1x1x2xi16, strided<[16, 8, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg0[%arg5, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg1[%arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<4x2x4x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<4x2x4x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %3 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %4 = vector.transfer_read %subview_0[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map11} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
            %5 = arith.xori %3, %2 : vector<1x1x1x1x64xi16>
            %6 = arith.cmpi sgt, %3, %5 : vector<1x1x1x1x64xi16>
            %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %8 = arith.cmpi ne, %7, %4 : vector<1x1x1x1x64xi32>
            %9 = arith.select %8, %5, %3 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %9, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_8_outlined_vf_1(%arg0: memref<4x2x4x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<4x2x4x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          scf.for %arg5 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x2x4x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<4x2x4x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %1 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %2 = arith.select %0, %1, %cst : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
            %3 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true]} : memref<1x1x1x1xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1xi16>
            %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [4] : vector<1x1x1x1x128xi16> to vector<1x1x1x1xi16>
            vector.transfer_write %4, %subview[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xi16>, memref<1x1x1x1xi16, strided<[16, 8, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_9_outlined_vf_0(%arg0: memref<4x2x8x1xi16, #hivm.address_space<ub>>, %arg1: memref<4x2x8x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>, %arg3: memref<4x2x8x2xi16, #hivm.address_space<ub>>, %arg4: memref<4x8x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg2[%arg6, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32, %1 {in_bounds = [true, true, true, true], permutation_map = #map6} : memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c8 step %c1 {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg7, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<4x2x8x1xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[16, 8, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<4x2x8x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<4x2x8x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true], permutation_map = #map7} : memref<1x1x1x1xi16, strided<[16, 8, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c8 step %c1 {
          %subview = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<4x2x8x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg4[%arg5, %arg7, 0] [1, 1, 2] [1, 1, 1] : memref<4x8x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
          %7 = vector.shape_cast %5 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %8 = arith.xori %6, %7 : vector<1x1x128xi16>
          %9 = arith.select %4, %8, %7 : vector<1x1x128xi1>, vector<1x1x128xi16>
          vector.transfer_write %9, %subview_0[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x2xi16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_10_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, %arg2: memref<2x2x1x2x8xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, %arg4: memref<2x2x2x8xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 8] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 8] : vector<1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg6, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map8} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, 0, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x1x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[32, 16, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map9} : memref<1x1x1x1x8xi16, strided<[32, 16, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %8 = arith.xori %6, %7 : vector<1x1x1x1x64xi16>
            %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x64xi16>
            %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x64xi32>
            %12 = arith.select %11, %8, %6 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 8] : vector<1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 8] : vector<1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x128xi1> to vector<1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true]} : memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %5) -> (vector<1x1x1x128xi16>) {
            %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %8 = arith.select %3, %arg9, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            %9 = vector.shape_cast %7 : vector<1x1x1x1x128xi16> to vector<1x1x1x128xi16>
            %10 = arith.xori %8, %9 : vector<1x1x1x128xi16>
            %11 = arith.select %4, %10, %9 : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            scf.yield %11 : vector<1x1x1x128xi16>
          }
          vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xi16>, memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_11_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>>, %arg2: memref<2x2x2x1x2x4xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>>, %arg4: memref<2x2x2x2x4xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 4] : vector<1x1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 1, 4] : vector<1x1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg0[%arg6, %arg8] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true, true], permutation_map = #map12} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi32>
            scf.for %arg9 = %c0 to %c2 step %c1 {
              %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x4xi16, strided<[64, 32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
              %subview_1 = memref.subview %arg2[%arg5, %arg6, %arg7, 0, %arg9, 0] [1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x2x1x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x4xi16, strided<[32, 16, 8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
              %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x4xi16, strided<[64, 32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
              %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x4xi16, strided<[64, 32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi16>
              %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true, true], permutation_map = #map13} : memref<1x1x1x1x1x4xi16, strided<[32, 16, 8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi16>
              %8 = arith.xori %6, %7 : vector<1x1x1x1x1x64xi16>
              %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x1x64xi16>
              %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x1x64xi1> to vector<1x1x1x1x1x64xi32>
              %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x1x64xi32>
              %12 = arith.select %11, %8, %6 : vector<1x1x1x1x1x64xi1>, vector<1x1x1x1x1x64xi16>
              vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x1x64xi16>, memref<1x1x1x1x1x4xi16, strided<[64, 32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            }
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 1, 4] : vector<1x1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 1, 4] : vector<1x1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x1x128xi1> to vector<1x1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg4[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<2x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %6 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %5) -> (vector<1x1x1x1x128xi16>) {
              %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x4xi16, strided<[64, 32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
              %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x4xi16, strided<[64, 32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x128xi16>
              %8 = arith.select %3, %arg10, %cst : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
              %9 = vector.shape_cast %7 : vector<1x1x1x1x1x128xi16> to vector<1x1x1x1x128xi16>
              %10 = arith.xori %8, %9 : vector<1x1x1x1x128xi16>
              %11 = arith.select %4, %10, %9 : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
              scf.yield %11 : vector<1x1x1x1x128xi16>
            }
            vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x128xi16>, memref<1x1x1x1x4xi16, strided<[32, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_12_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2x4x1x2x2xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<2x2x4x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg0[%arg6, %arg8] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true, true], permutation_map = #map12} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi32>
            scf.for %arg9 = %c0 to %c2 step %c1 {
              %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[64, 32, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %subview_1 = memref.subview %arg2[%arg5, %arg6, %arg7, 0, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x1x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[32, 16, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[64, 32, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x2xi16, strided<[64, 32, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi16>
              %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true, true], permutation_map = #map13} : memref<1x1x1x1x1x2xi16, strided<[32, 16, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x64xi16>
              %8 = arith.xori %6, %7 : vector<1x1x1x1x1x64xi16>
              %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x1x64xi16>
              %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x1x64xi1> to vector<1x1x1x1x1x64xi32>
              %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x1x64xi32>
              %12 = arith.select %11, %8, %6 : vector<1x1x1x1x1x64xi1>, vector<1x1x1x1x1x64xi16>
              vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x1x64xi16>, memref<1x1x1x1x1x2xi16, strided<[64, 32, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            }
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x1x128xi1> to vector<1x1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg4[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x4x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %6 = scf.for %arg9 = %c0 to %c2 step %c1 iter_args(%arg10 = %5) -> (vector<1x1x1x1x128xi16>) {
              %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, %arg9, 0] [1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x2xi16, strided<[64, 32, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x2xi16, strided<[64, 32, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x128xi16>
              %8 = arith.select %3, %arg10, %cst : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
              %9 = vector.shape_cast %7 : vector<1x1x1x1x1x128xi16> to vector<1x1x1x1x128xi16>
              %10 = arith.xori %8, %9 : vector<1x1x1x1x128xi16>
              %11 = arith.select %4, %10, %9 : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
              scf.yield %11 : vector<1x1x1x1x128xi16>
            }
            vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x128xi16>, memref<1x1x1x1x2xi16, strided<[32, 16, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_13_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x8x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2x8x1x2xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x8x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    scf.for %arg4 = %c0 to %c2 step %c1 {
      scf.for %arg5 = %c0 to %c2 step %c1 {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %subview = memref.subview %arg2[%arg4, %arg5, %arg6, 0, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x8x1x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[32, 16, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %2 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map10} : memref<1x1x1x1x2xi16, strided<[32, 16, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg0[%arg5, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg1[%arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x8x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[64, 32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x8x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[64, 32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %3 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[64, 32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %4 = vector.transfer_read %subview_0[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map11} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
            %5 = arith.xori %3, %2 : vector<1x1x1x1x64xi16>
            %6 = arith.cmpi sgt, %3, %5 : vector<1x1x1x1x64xi16>
            %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %8 = arith.cmpi ne, %7, %4 : vector<1x1x1x1x64xi32>
            %9 = arith.select %8, %5, %3 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %9, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x2xi16, strided<[64, 32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_13_outlined_vf_1(%arg0: memref<2x2x8x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<2x2x8x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c8 step %c1 {
          scf.for %arg5 = %c0 to %c2 step %c1 {
            %subview = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5] [1, 1, 1, 1] [1, 1, 1, 1] : memref<2x2x8x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x2x8x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[64, 32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %1 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[64, 32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %2 = arith.select %0, %1, %cst : vector<1x1x1x1x128xi1>, vector<1x1x1x1x128xi16>
            %3 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true]} : memref<1x1x1x1xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1xi16>
            %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [4] : vector<1x1x1x1x128xi16> to vector<1x1x1x1xi16>
            vector.transfer_write %4, %subview[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x1xi16>, memref<1x1x1x1xi16, strided<[32, 16, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_14_outlined_vf_0(%arg0: memref<2x2x16x1xi16, #hivm.address_space<ub>>, %arg1: memref<2x2x16x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>, %arg3: memref<2x2x16x2xi16, #hivm.address_space<ub>>, %arg4: memref<2x16x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg2[%arg6, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32, %1 {in_bounds = [true, true, true, true], permutation_map = #map6} : memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c16 step %c1 {
          %subview_0 = memref.subview %arg0[%arg5, %arg6, %arg7, 0] [1, 1, 1, 1] [1, 1, 1, 1] : memref<2x2x16x1xi16, #hivm.address_space<ub>> to memref<1x1x1x1xi16, strided<[32, 16, 1, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x2x16x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[64, 32, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x2x16x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[64, 32, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[64, 32, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true], permutation_map = #map7} : memref<1x1x1x1xi16, strided<[32, 16, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[64, 32, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c16 step %c1 {
          %subview = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x2x16x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[64, 32, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg4[%arg5, %arg7, 0] [1, 1, 2] [1, 1, 1] : memref<2x16x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[32, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[64, 32, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[32, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
          %7 = vector.shape_cast %5 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %8 = arith.xori %6, %7 : vector<1x1x128xi16>
          %9 = arith.select %4, %8, %7 : vector<1x1x128xi1>, vector<1x1x128xi16>
          vector.transfer_write %9, %subview_0[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x2xi16, strided<[32, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_15_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x16xi16, #hivm.address_space<ub>>, %arg2: memref<2x1x2x16xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x2x16xi16, #hivm.address_space<ub>>, %arg4: memref<2x2x16xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 16] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 16] : vector<1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg0[%arg5, %arg6] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map14} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg2[%arg5, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x1x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[32, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map15} : memref<1x1x1x16xi16, strided<[32, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 16] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 16] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg4[%arg5, %arg6, 0] [1, 1, 16] [1, 1, 1] : memref<2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x16xi16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x16xi16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %5) -> (vector<1x1x128xi16>) {
          %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %8 = arith.select %3, %arg8, %cst : vector<1x1x128xi1>, vector<1x1x128xi16>
          %9 = vector.shape_cast %7 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %10 = arith.xori %8, %9 : vector<1x1x128xi16>
          %11 = arith.select %4, %10, %9 : vector<1x1x128xi1>, vector<1x1x128xi16>
          scf.yield %11 : vector<1x1x128xi16>
        }
        vector.transfer_write %6, %subview[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x16xi16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_16_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, %arg2: memref<2x2x1x2x8xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, %arg4: memref<2x2x2x8xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 8] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 8] : vector<1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg5, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map16} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, 0, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x1x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[32, 16, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map9} : memref<1x1x1x1x8xi16, strided<[32, 16, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %8 = arith.xori %6, %7 : vector<1x1x1x1x64xi16>
            %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x64xi16>
            %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x64xi32>
            %12 = arith.select %11, %8, %6 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 8] : vector<1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 8] : vector<1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x128xi1> to vector<1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true]} : memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %5) -> (vector<1x1x1x128xi16>) {
            %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 8] [1, 1, 1, 1, 1] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x8xi16, strided<[64, 32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %8 = arith.select %3, %arg9, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            %9 = vector.shape_cast %7 : vector<1x1x1x1x128xi16> to vector<1x1x1x128xi16>
            %10 = arith.xori %8, %9 : vector<1x1x1x128xi16>
            %11 = arith.select %4, %10, %9 : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            scf.yield %11 : vector<1x1x1x128xi16>
          }
          vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xi16>, memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_17_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x4x2x2x4xi16, #hivm.address_space<ub>>, %arg2: memref<2x4x1x2x4xi16, #hivm.address_space<ub>>, %arg3: memref<2x4x2x2x4xi16, #hivm.address_space<ub>>, %arg4: memref<2x4x2x4xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 4] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 4] : vector<1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c4 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg5, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map16} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<2x4x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[64, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, 0, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<2x4x1x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[32, 8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<2x4x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[64, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x4xi16, strided<[64, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map9} : memref<1x1x1x1x4xi16, strided<[32, 8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %8 = arith.xori %6, %7 : vector<1x1x1x1x64xi16>
            %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x64xi16>
            %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x64xi32>
            %12 = arith.select %11, %8, %6 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x4xi16, strided<[64, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 4] : vector<1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 4] : vector<1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x128xi1> to vector<1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c4 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<2x4x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x4xi16, strided<[32, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true]} : memref<1x1x1x4xi16, strided<[32, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %5) -> (vector<1x1x1x128xi16>) {
            %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1] : memref<2x4x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x1x4xi16, strided<[64, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x4xi16, strided<[64, 16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %8 = arith.select %3, %arg9, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            %9 = vector.shape_cast %7 : vector<1x1x1x1x128xi16> to vector<1x1x1x128xi16>
            %10 = arith.xori %8, %9 : vector<1x1x1x128xi16>
            %11 = arith.select %4, %10, %9 : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            scf.yield %11 : vector<1x1x1x128xi16>
          }
          vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xi16>, memref<1x1x1x4xi16, strided<[32, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_18_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x8x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x8x1x2x2xi16, #hivm.address_space<ub>>, %arg3: memref<2x8x2x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<2x8x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c8 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg0[%arg5, %arg7] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true, true], permutation_map = #map16} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi32>
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x8x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[64, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_1 = memref.subview %arg2[%arg5, %arg6, 0, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x8x1x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[32, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x8x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[64, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[64, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true, true], permutation_map = #map9} : memref<1x1x1x1x2xi16, strided<[32, 4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x64xi16>
            %8 = arith.xori %6, %7 : vector<1x1x1x1x64xi16>
            %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x1x64xi16>
            %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x1x64xi1> to vector<1x1x1x1x64xi32>
            %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x1x64xi32>
            %12 = arith.select %11, %8, %6 : vector<1x1x1x1x64xi1>, vector<1x1x1x1x64xi16>
            vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : vector<1x1x1x1x64xi16>, memref<1x1x1x1x2xi16, strided<[64, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          }
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 1, 2] : vector<1x1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x1x128xi1> to vector<1x1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c8 step %c1 {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg4[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x8x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %5 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %6 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %5) -> (vector<1x1x1x128xi16>) {
            %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, %arg8, 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : memref<2x8x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x2xi16, strided<[64, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
            %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true, true]} : memref<1x1x1x1x2xi16, strided<[64, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x128xi16>
            %8 = arith.select %3, %arg9, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            %9 = vector.shape_cast %7 : vector<1x1x1x1x128xi16> to vector<1x1x1x128xi16>
            %10 = arith.xori %8, %9 : vector<1x1x1x128xi16>
            %11 = arith.select %4, %10, %9 : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
            scf.yield %11 : vector<1x1x1x128xi16>
          }
          vector.transfer_write %6, %subview[%c0, %c0, %c0, %c0], %3 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xi16>, memref<1x1x1x2xi16, strided<[32, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_19_outlined_vf_0(%arg0: memref<2x2xi32, #hivm.address_space<ub>>, %arg1: memref<2x16x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x16x1x2xi16, #hivm.address_space<ub>>, %arg3: memref<2x16x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    scf.for %arg4 = %c0 to %c2 step %c1 {
      scf.for %arg5 = %c0 to %c16 step %c1 {
        %subview = memref.subview %arg2[%arg4, %arg5, 0, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x16x1x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[32, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %2 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map4} : memref<1x1x1x2xi16, strided<[32, 2, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg0[%arg4, %arg6] [1, 1] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg1[%arg4, %arg5, %arg6, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x16x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[64, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg4, %arg5, %arg6, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x16x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[64, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %3 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[64, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %4 = vector.transfer_read %subview_0[%c0, %c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map17} : memref<1x1xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
          %5 = arith.xori %3, %2 : vector<1x1x1x64xi16>
          %6 = arith.cmpi sgt, %3, %5 : vector<1x1x1x64xi16>
          %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %8 = arith.cmpi ne, %7, %4 : vector<1x1x1x64xi32>
          %9 = arith.select %8, %5, %3 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %9, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[64, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_19_outlined_vf_1(%arg0: memref<2x16x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<2x16x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %subview = memref.subview %arg1[%arg2, %arg3, %arg4] [1, 1, 1] [1, 1, 1] : memref<2x16x2xi16, #hivm.address_space<ub>> to memref<1x1x1xi16, strided<[32, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<2x16x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[64, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %1 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[64, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %2 = arith.select %0, %1, %cst : vector<1x1x1x128xi1>, vector<1x1x1x128xi16>
          %3 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true]} : memref<1x1x1xi16, strided<[32, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1xi16>
          %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [3] : vector<1x1x1x128xi16> to vector<1x1x1xi16>
          vector.transfer_write %4, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x1x1xi16>, memref<1x1x1xi16, strided<[32, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_20_outlined_vf_0(%arg0: memref<2x32x1xi16, #hivm.address_space<ub>>, %arg1: memref<2x32x2xi16, #hivm.address_space<ub>>, %arg2: memref<2x2xi32, #hivm.address_space<ub>>, %arg3: memref<2x32x2xi16, #hivm.address_space<ub>>, %arg4: memref<32x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %subview = memref.subview %arg2[%arg5, 0] [1, 2] [1, 1] : memref<2x2xi32, #hivm.address_space<ub>> to memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview[%c0, %c0], %c0_i32, %1 {in_bounds = [true, true, true], permutation_map = #map18} : memref<1x2xi32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
      scf.for %arg6 = %c0 to %c32 step %c1 {
        %subview_0 = memref.subview %arg0[%arg5, %arg6, 0] [1, 1, 1] [1, 1, 1] : memref<2x32x1xi16, #hivm.address_space<ub>> to memref<1x1x1xi16, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg1[%arg5, %arg6, 0] [1, 1, 2] [1, 1, 1] : memref<2x32x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[64, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 1, 2] [1, 1, 1] : memref<2x32x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[64, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[64, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true], permutation_map = #map2} : memref<1x1x1xi16, strided<[32, 1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %8 = arith.xori %6, %7 : vector<1x1x64xi16>
        %9 = arith.cmpi sgt, %6, %8 : vector<1x1x64xi16>
        %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xi1> to vector<1x1x64xi32>
        %11 = arith.cmpi ne, %10, %5 : vector<1x1x64xi32>
        %12 = arith.select %11, %8, %6 : vector<1x1x64xi1>, vector<1x1x64xi16>
        vector.transfer_write %12, %subview_2[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : vector<1x1x64xi16>, memref<1x1x2xi16, strided<[64, 2, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    %2 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %3 = vector.constant_mask [1, 2] : vector<1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x128xi1> to vector<1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c32 step %c1 {
        %subview = memref.subview %arg3[%arg5, %arg6, 0] [1, 1, 2] [1, 1, 1] : memref<2x32x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[64, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg4[%arg6, 0] [1, 2] [1, 1] : memref<32x2xi16, #hivm.address_space<ub>> to memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[64, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = vector.transfer_read %subview_0[%c0, %c0], %c0_i16, %3 {in_bounds = [true, true]} : memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xi16>
        %7 = vector.shape_cast %5 : vector<1x1x128xi16> to vector<1x128xi16>
        %8 = arith.xori %6, %7 : vector<1x128xi16>
        %9 = arith.select %4, %8, %7 : vector<1x128xi1>, vector<1x128xi16>
        vector.transfer_write %9, %subview_0[%c0, %c0], %3 {in_bounds = [true, true]} : vector<1x128xi16>, memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_21_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x32xi16, #hivm.address_space<ub>>, %arg2: memref<1x2x32xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x32xi16, #hivm.address_space<ub>>, %arg4: memref<2x32xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 32] : vector<1x1x64xi1>
    %1 = vector.constant_mask [1, 32] : vector<1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %subview = memref.subview %arg0[%arg5] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true, true, true], permutation_map = #map19} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview_0 = memref.subview %arg1[%arg5, %arg6, 0] [1, 1, 32] [1, 1, 1] : memref<2x2x32xi16, #hivm.address_space<ub>> to memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[0, %arg6, 0] [1, 1, 32] [1, 1, 1] : memref<1x2x32xi16, #hivm.address_space<ub>> to memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 1, 32] [1, 1, 1] : memref<2x2x32xi16, #hivm.address_space<ub>> to memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>
        %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true]} : memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %7 = vector.transfer_read %subview_1[%c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true], permutation_map = #map20} : memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %8 = arith.xori %6, %7 : vector<1x1x64xi16>
        %9 = arith.cmpi sgt, %6, %8 : vector<1x1x64xi16>
        %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xi1> to vector<1x1x64xi32>
        %11 = arith.cmpi ne, %10, %5 : vector<1x1x64xi32>
        %12 = arith.select %11, %8, %6 : vector<1x1x64xi1>, vector<1x1x64xi16>
        vector.transfer_write %12, %subview_2[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : vector<1x1x64xi16>, memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    %2 = vector.constant_mask [1, 1, 32] : vector<1x1x128xi1>
    %3 = vector.constant_mask [1, 32] : vector<1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x128xi1> to vector<1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %subview = memref.subview %arg4[%arg5, 0] [1, 32] [1, 1] : memref<2x32xi16, #hivm.address_space<ub>> to memref<1x32xi16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview[%c0, %c0], %c0_i16, %3 {in_bounds = [true, true]} : memref<1x32xi16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xi16>
      %6 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %5) -> (vector<1x128xi16>) {
        %subview_0 = memref.subview %arg3[%arg5, %arg6, 0] [1, 1, 32] [1, 1, 1] : memref<2x2x32xi16, #hivm.address_space<ub>> to memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>
        %7 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true]} : memref<1x1x32xi16, strided<[64, 32, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %8 = arith.select %3, %arg7, %cst : vector<1x128xi1>, vector<1x128xi16>
        %9 = vector.shape_cast %7 : vector<1x1x128xi16> to vector<1x128xi16>
        %10 = arith.xori %8, %9 : vector<1x128xi16>
        %11 = arith.select %4, %10, %9 : vector<1x128xi1>, vector<1x128xi16>
        scf.yield %11 : vector<1x128xi16>
      }
      vector.transfer_write %6, %subview[%c0, %c0], %3 {in_bounds = [true, true]} : vector<1x128xi16>, memref<1x32xi16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @median_small_flat_kernel_fused_22_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<2x2x2x16xi16, #hivm.address_space<ub>>, %arg2: memref<2x1x2x16xi16, #hivm.address_space<ub>>, %arg3: memref<2x2x2x16xi16, #hivm.address_space<ub>>, %arg4: memref<2x2x16xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 16] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 16] : vector<1x1x64xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg0[%arg6] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map21} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg2[%arg5, 0, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x1x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[32, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map15} : memref<1x1x1x16xi16, strided<[32, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 16] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 16] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c2 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg4[%arg5, %arg6, 0] [1, 1, 16] [1, 1, 1] : memref<2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x16xi16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x16xi16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %5) -> (vector<1x1x128xi16>) {
          %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<2x2x2x16xi16, #hivm.address_space<ub>> to memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x16xi16, strided<[64, 32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %8 = arith.select %3, %arg8, %cst : vector<1x1x128xi1>, vector<1x1x128xi16>
          %9 = vector.shape_cast %7 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %10 = arith.xori %8, %9 : vector<1x1x128xi16>
          %11 = arith.select %4, %10, %9 : vector<1x1x128xi1>, vector<1x1x128xi16>
          scf.yield %11 : vector<1x1x128xi16>
        }
        vector.transfer_write %6, %subview[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x16xi16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_23_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<4x2x2x8xi16, #hivm.address_space<ub>>, %arg2: memref<4x1x2x8xi16, #hivm.address_space<ub>>, %arg3: memref<4x2x2x8xi16, #hivm.address_space<ub>>, %arg4: memref<4x2x8xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 8] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 8] : vector<1x1x64xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg0[%arg6] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map21} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<4x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg2[%arg5, 0, %arg7, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<4x1x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x8xi16, strided<[16, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<4x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map15} : memref<1x1x1x8xi16, strided<[16, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 8] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 8] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c4 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg4[%arg5, %arg6, 0] [1, 1, 8] [1, 1, 1] : memref<4x2x8xi16, #hivm.address_space<ub>> to memref<1x1x8xi16, strided<[16, 8, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x8xi16, strided<[16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %5) -> (vector<1x1x128xi16>) {
          %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<4x2x2x8xi16, #hivm.address_space<ub>> to memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x8xi16, strided<[32, 16, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %8 = arith.select %3, %arg8, %cst : vector<1x1x128xi1>, vector<1x1x128xi16>
          %9 = vector.shape_cast %7 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %10 = arith.xori %8, %9 : vector<1x1x128xi16>
          %11 = arith.select %4, %10, %9 : vector<1x1x128xi1>, vector<1x1x128xi16>
          scf.yield %11 : vector<1x1x128xi16>
        }
        vector.transfer_write %6, %subview[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x8xi16, strided<[16, 8, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_24_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<8x2x2x4xi16, #hivm.address_space<ub>>, %arg2: memref<8x1x2x4xi16, #hivm.address_space<ub>>, %arg3: memref<8x2x2x4xi16, #hivm.address_space<ub>>, %arg4: memref<8x2x4xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 4] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 4] : vector<1x1x64xi1>
    scf.for %arg5 = %c0 to %c8 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg0[%arg6] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map21} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<8x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg2[%arg5, 0, %arg7, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<8x1x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x4xi16, strided<[8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<8x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map15} : memref<1x1x1x4xi16, strided<[8, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 4] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 4] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c8 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg4[%arg5, %arg6, 0] [1, 1, 4] [1, 1, 1] : memref<8x2x4xi16, #hivm.address_space<ub>> to memref<1x1x4xi16, strided<[8, 4, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x4xi16, strided<[8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %5) -> (vector<1x1x128xi16>) {
          %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 4] [1, 1, 1, 1] : memref<8x2x2x4xi16, #hivm.address_space<ub>> to memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x4xi16, strided<[16, 8, 4, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %8 = arith.select %3, %arg8, %cst : vector<1x1x128xi1>, vector<1x1x128xi16>
          %9 = vector.shape_cast %7 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %10 = arith.xori %8, %9 : vector<1x1x128xi16>
          %11 = arith.select %4, %10, %9 : vector<1x1x128xi1>, vector<1x1x128xi16>
          scf.yield %11 : vector<1x1x128xi16>
        }
        vector.transfer_write %6, %subview[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x4xi16, strided<[8, 4, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_25_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<16x2x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<16x1x2x2xi16, #hivm.address_space<ub>>, %arg3: memref<16x2x2x2xi16, #hivm.address_space<ub>>, %arg4: memref<16x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x128xi16>
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x64xi1>
    %1 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg0[%arg6] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true, true, true, true], permutation_map = #map21} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi32>
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %subview_0 = memref.subview %arg1[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %arg2[%arg5, 0, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x1x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %subview_2 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true, true], permutation_map = #map15} : memref<1x1x1x2xi16, strided<[4, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x64xi16>
          %8 = arith.xori %6, %7 : vector<1x1x1x64xi16>
          %9 = arith.cmpi sgt, %6, %8 : vector<1x1x1x64xi16>
          %10 = arith.extui %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x1x64xi1> to vector<1x1x1x64xi32>
          %11 = arith.cmpi ne, %10, %5 : vector<1x1x1x64xi32>
          %12 = arith.select %11, %8, %6 : vector<1x1x1x64xi1>, vector<1x1x1x64xi16>
          vector.transfer_write %12, %subview_2[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : vector<1x1x1x64xi16>, memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        }
      }
    }
    %2 = vector.constant_mask [1, 1, 1, 2] : vector<1x1x1x128xi1>
    %3 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    %4 = vector.shape_cast %2 : vector<1x1x1x128xi1> to vector<1x1x128xi1>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      scf.for %arg6 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg4[%arg5, %arg6, 0] [1, 1, 2] [1, 1, 1] : memref<16x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %3 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %6 = scf.for %arg7 = %c0 to %c2 step %c1 iter_args(%arg8 = %5) -> (vector<1x1x128xi16>) {
          %subview_0 = memref.subview %arg3[%arg5, %arg6, %arg7, 0] [1, 1, 1, 2] [1, 1, 1, 1] : memref<16x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
          %7 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %c0_i16, %2 {in_bounds = [true, true, true, true]} : memref<1x1x1x2xi16, strided<[8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xi16>
          %8 = arith.select %3, %arg8, %cst : vector<1x1x128xi1>, vector<1x1x128xi16>
          %9 = vector.shape_cast %7 : vector<1x1x1x128xi16> to vector<1x1x128xi16>
          %10 = arith.xori %8, %9 : vector<1x1x128xi16>
          %11 = arith.select %4, %10, %9 : vector<1x1x128xi1>, vector<1x1x128xi16>
          scf.yield %11 : vector<1x1x128xi16>
        }
        vector.transfer_write %6, %subview[%c0, %c0, %c0], %3 {in_bounds = [true, true, true]} : vector<1x1x128xi16>, memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_26_outlined_vf_0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<32x2x2xi16, #hivm.address_space<ub>>, %arg2: memref<32x1x2xi16, #hivm.address_space<ub>>, %arg3: memref<32x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg4 = %c0 to %c32 step %c1 {
      %subview = memref.subview %arg2[%arg4, 0, 0] [1, 1, 2] [1, 1, 1] : memref<32x1x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[2, 2, 1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i16, %1 {in_bounds = [true, true, true], permutation_map = #map22} : memref<1x1x2xi16, strided<[2, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
      scf.for %arg5 = %c0 to %c2 step %c1 {
        %subview_0 = memref.subview %arg0[%arg5] [1] [1] : memref<2xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg1[%arg4, %arg5, 0] [1, 1, 2] [1, 1, 1] : memref<32x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg4, %arg5, 0] [1, 1, 2] [1, 1, 1] : memref<32x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %3 = vector.transfer_read %subview_1[%c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi16>
        %4 = vector.transfer_read %subview_0[%c0], %c0_i32 {in_bounds = [true, true, true], permutation_map = #map23} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
        %5 = arith.xori %3, %2 : vector<1x1x64xi16>
        %6 = arith.cmpi sgt, %3, %5 : vector<1x1x64xi16>
        %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xi1> to vector<1x1x64xi32>
        %8 = arith.cmpi ne, %7, %4 : vector<1x1x64xi32>
        %9 = arith.select %8, %5, %3 : vector<1x1x64xi1>, vector<1x1x64xi16>
        vector.transfer_write %9, %subview_2[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : vector<1x1x64xi16>, memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_26_outlined_vf_1(%arg0: memref<32x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<32x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 2] : vector<1x1x128xi1>
    scf.for %arg2 = %c0 to %c32 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %subview = memref.subview %arg1[%arg2, %arg3] [1, 1] [1, 1] : memref<32x2xi16, #hivm.address_space<ub>> to memref<1x1xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg0[%arg2, %arg3, 0] [1, 1, 2] [1, 1, 1] : memref<32x2x2xi16, #hivm.address_space<ub>> to memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true]} : memref<1x1x2xi16, strided<[4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x128xi16>
        %2 = arith.select %0, %1, %cst : vector<1x1x128xi1>, vector<1x1x128xi16>
        %3 = vector.transfer_read %subview[%c0, %c0], %c0_i16 {in_bounds = [true, true]} : memref<1x1xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1xi16>
        %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [2] : vector<1x1x128xi16> to vector<1x1xi16>
        vector.transfer_write %4, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xi16>, memref<1x1xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @median_small_flat_kernel_fused_27_outlined_vf_0(%arg0: memref<64x1xi16, #hivm.address_space<ub>>, %arg1: memref<64x2xi16, #hivm.address_space<ub>>, %arg2: memref<2xi32, #hivm.address_space<ub>>, %arg3: memref<64x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 2] : vector<1x64xi1>
    %1 = vector.constant_mask [2] : vector<64xi1>
    %2 = vector.transfer_read %arg2[%c0], %c0_i32, %1 {in_bounds = [true, true], permutation_map = #map} : memref<2xi32, #hivm.address_space<ub>>, vector<1x64xi32>
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4, 0] [1, 1] [1, 1] : memref<64x1xi16, #hivm.address_space<ub>> to memref<1x1xi16, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg4, 0] [1, 2] [1, 1] : memref<64x2xi16, #hivm.address_space<ub>> to memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg3[%arg4, 0] [1, 2] [1, 1] : memref<64x2xi16, #hivm.address_space<ub>> to memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %3 = vector.transfer_read %subview_0[%c0, %c0], %c0_i16, %0 {in_bounds = [true, true]} : memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi16>
      %4 = vector.transfer_read %subview[%c0, %c0], %c0_i16 {in_bounds = [true, true], permutation_map = #map24} : memref<1x1xi16, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi16>
      %5 = arith.xori %3, %4 : vector<1x64xi16>
      %6 = arith.cmpi sgt, %3, %5 : vector<1x64xi16>
      %7 = arith.extui %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xi1> to vector<1x64xi32>
      %8 = arith.cmpi ne, %7, %2 : vector<1x64xi32>
      %9 = arith.select %8, %5, %3 : vector<1x64xi1>, vector<1x64xi16>
      vector.transfer_write %9, %subview_1[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x64xi16>, memref<1x2xi16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @median_small_flat_kernel_fused_28_outlined_vf_0(%arg0: memref<128xi16, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg1[], %c0_i32 : memref<i32, #hivm.address_space<ub>>, vector<i32>
    %1 = scf.for %arg2 = %c0 to %c128 step %c64 iter_args(%arg3 = %0) -> (vector<i32>) {
      %subview = memref.subview %arg0[%arg2] [64] [1] : memref<128xi16, #hivm.address_space<ub>> to memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview[%c0], %c0_i16 {in_bounds = [true]} : memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi16>
      %3 = vector.extractelement %arg3[] : vector<i32>
      %4 = arith.extsi %2 {round_mode = #hfusion.round_mode<rint>} : vector<64xi16> to vector<64xi32>
      %5 = vector.multi_reduction <add>, %4, %3 [0] : vector<64xi32> to i32
      %6 = vector.broadcast %5 : i32 to vector<i32>
      scf.yield %6 : vector<i32>
    }
    vector.transfer_write %1, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
    return
  }
  func.func @median_small_flat_kernel_outlined_vf_0(%arg0: memref<44xi16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<128xi16>
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [44] : vector<128xi1>
    vector.transfer_write %cst, %arg0[%c0], %0 {in_bounds = [true]} : vector<128xi16>, memref<44xi16, #hivm.address_space<ub>>
    return
  }
  func.func @median_small_flat_kernel_outlined_vf_1(%arg0: memref<128xi16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<128xi16>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<128xi16>, memref<128xi16, #hivm.address_space<ub>>
    return
  }
  func.func @median_small_flat_kernel_outlined_vf_2(%arg0: memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<32767> : vector<1x1x1x1x1x1x128xi16>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x1x128xi1>
    scf.for %arg1 = %c0 to %c2 step %c1 {
      scf.for %arg2 = %c0 to %c2 step %c1 {
        scf.for %arg3 = %c0 to %c2 step %c1 {
          scf.for %arg4 = %c0 to %c2 step %c1 {
            scf.for %arg5 = %c0 to %c2 step %c1 {
              scf.for %arg6 = %c0 to %c2 step %c1 {
                %subview = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, 0] [1, 1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x1x2xi16, strided<[64, 32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
                vector.transfer_write %cst, %subview[%c0, %c0, %c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true, true, true]} : vector<1x1x1x1x1x1x128xi16>, memref<1x1x1x1x1x1x2xi16, strided<[64, 32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              }
            }
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel_outlined_vf_3(%arg0: memref<2xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<64xindex>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xindex>
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [2] : vector<64xi1>
    %1 = arith.addi %cst, %cst_0 : vector<64xindex>
    %2 = arith.index_cast %1 : vector<64xindex> to vector<64xi32>
    vector.transfer_write %2, %arg0[%c0], %0 {in_bounds = [true]} : vector<64xi32>, memref<2xi32, #hivm.address_space<ub>>
    return
  }
  func.func @median_small_flat_kernel_outlined_vf_4(%arg0: memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>, %arg1: memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x1x1x1x1x1x128xi16>
    %c0_i16 = arith.constant 0 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 1, 1, 1, 1, 2] : vector<1x1x1x1x1x1x128xi1>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          scf.for %arg5 = %c0 to %c2 step %c1 {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %subview = memref.subview %arg1[%arg2, %arg3, %arg4, %arg5, %arg6, %arg7] [1, 1, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x1xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
                %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4, %arg5, %arg6, %arg7, 0] [1, 1, 1, 1, 1, 1, 2] [1, 1, 1, 1, 1, 1, 1] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> to memref<1x1x1x1x1x1x2xi16, strided<[64, 32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
                %1 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0, %c0, %c0, %c0], %c0_i16, %0 {in_bounds = [true, true, true, true, true, true, true]} : memref<1x1x1x1x1x1x2xi16, strided<[64, 32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x1x128xi16>
                %2 = arith.select %0, %1, %cst : vector<1x1x1x1x1x1x128xi1>, vector<1x1x1x1x1x1x128xi16>
                %3 = vector.transfer_read %subview[%c0, %c0, %c0, %c0, %c0, %c0], %c0_i16 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x1x1x1x1xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x1x1x1xi16>
                %4 = vector.multi_reduction <xor>, %2, %3 {withoutInitMergeOp} [6] : vector<1x1x1x1x1x1x128xi16> to vector<1x1x1x1x1x1xi16>
                vector.transfer_write %4, %subview[%c0, %c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x1x1xi16>, memref<1x1x1x1x1x1xi16, strided<[32, 16, 8, 4, 2, 1], offset: ?>, #hivm.address_space<ub>>
              }
            }
          }
        }
      }
    }
    return
  }
  func.func @median_small_flat_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg4, %arg5 : i32
    %1 = arith.muli %0, %arg6 : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<44xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_outlined_vf_0(%alloc) {hivm.vector_function, no_inline} : (memref<44xi16, #hivm.address_space<ub>>) -> ()
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_outlined_vf_1(%alloc_0) {hivm.vector_function, no_inline} : (memref<128xi16, #hivm.address_space<ub>>) -> ()
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<?xi16, #hivm.address_space<gm>> to memref<128xi16, strided<[1]>, #hivm.address_space<gm>>
    %alloc_1 = memref.alloc() : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1, 2, 3, 4, 5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<128xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_outlined_vf_2(%alloc_1) {hivm.vector_function, no_inline} : (memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %subview = memref.subview %reinterpret_cast[0] [90] [1] : memref<128xi16, strided<[1]>, #hivm.address_space<gm>> to memref<90xi16, #hivm.address_space<gm>>
    %subview_2 = memref.subview %collapse_shape[0] [90] [1] : memref<128xi16, #hivm.address_space<ub>> to memref<90xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%subview : memref<90xi16, #hivm.address_space<gm>>) outs(%subview_2 : memref<90xi16, #hivm.address_space<ub>>) left_padding_num = %c0 : index eviction_policy = <EvictFirst> core_type = <VECTOR>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2xi32, #hivm.address_space<ub>>
    call @median_small_flat_kernel_outlined_vf_3(%alloc_3) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>) -> ()
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_outlined_vf_4(%alloc_1, %alloc_4) {hivm.vector_function, no_inline} : (memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>, memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_5 = memref.collapse_shape %alloc_1 [[0, 1, 2, 3, 4], [5], [6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<32x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %alloc_4 [[0, 1, 2, 3, 4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<32x2xi16, #hivm.address_space<ub>>
    %expand_shape = memref.expand_shape %collapse_shape_6 [[0], [1, 2]] output_shape [32, 2, 1] : memref<32x2xi16, #hivm.address_space<ub>> into memref<32x2x1xi16, #hivm.address_space<ub>>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<2x2xi32, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_0_outlined_vf_0(%alloc_3, %alloc_7, %alloc_8) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>) -> ()
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32x2x2xi16, #hivm.address_space<ub>>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_0_outlined_vf_1(%expand_shape, %collapse_shape_5, %alloc_8, %alloc_9, %alloc_10) {hivm.vector_function, no_inline} : (memref<32x2x1xi16, #hivm.address_space<ub>>, memref<32x2x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>, memref<32x2x2xi16, #hivm.address_space<ub>>, memref<32x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_11 = memref.expand_shape %alloc_9 [[0, 1], [2], [3]] output_shape [16, 2, 2, 2] : memref<32x2x2xi16, #hivm.address_space<ub>> into memref<16x2x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_0_outlined_vf_2(%alloc_9, %alloc_10) {hivm.vector_function, no_inline} : (memref<32x2x2xi16, #hivm.address_space<ub>>, memref<32x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_12 = memref.expand_shape %alloc_10 [[0, 1, 2], [3]] output_shape [16, 2, 1, 2] : memref<32x2xi16, #hivm.address_space<ub>> into memref<16x2x1x2xi16, #hivm.address_space<ub>>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<16x2x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_1_outlined_vf_0(%alloc_8, %expand_shape_11, %expand_shape_12, %alloc_13) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<16x2x2x2xi16, #hivm.address_space<ub>>, memref<16x2x1x2xi16, #hivm.address_space<ub>>, memref<16x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<16x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_1_outlined_vf_1(%alloc_13, %alloc_14) {hivm.vector_function, no_inline} : (memref<16x2x2x2xi16, #hivm.address_space<ub>>, memref<16x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_15 = memref.collapse_shape %alloc_7 [[0, 1, 2, 3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<16x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_15 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<16x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_16 = memref.expand_shape %alloc_14 [[0], [1], [2, 3]] output_shape [16, 2, 2, 1] : memref<16x2x2xi16, #hivm.address_space<ub>> into memref<16x2x2x1xi16, #hivm.address_space<ub>>
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<16x2x2x2xi16, #hivm.address_space<ub>>
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<16x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_18 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<16x2x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_15 : memref<16x2x2xi16, #hivm.address_space<ub>>) outs(%alloc_18 : memref<16x2x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_2_outlined_vf_0(%expand_shape_16, %alloc_13, %alloc_8, %alloc_17, %alloc_18) {hivm.vector_function, no_inline} : (memref<16x2x2x1xi16, #hivm.address_space<ub>>, memref<16x2x2x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>, memref<16x2x2x2xi16, #hivm.address_space<ub>>, memref<16x2x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_19 = memref.expand_shape %alloc_17 [[0, 1], [2], [3], [4]] output_shape [8, 2, 2, 2, 2] : memref<16x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %alloc_7 [[0, 1, 2], [3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_20 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_21 = memref.expand_shape %alloc_18 [[0, 1, 2], [3], [4]] output_shape [8, 2, 1, 2, 2] : memref<16x2x2xi16, #hivm.address_space<ub>> into memref<8x2x1x2x2xi16, #hivm.address_space<ub>>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<8x2x2x2x2xi16, #hivm.address_space<ub>>
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<8x2x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_23 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x2x2x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_20 : memref<8x2x2x2xi16, #hivm.address_space<ub>>) outs(%alloc_23 : memref<8x2x2x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_3_outlined_vf_0(%alloc_8, %expand_shape_19, %expand_shape_21, %alloc_22, %alloc_23) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, memref<8x2x1x2x2xi16, #hivm.address_space<ub>>, memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, memref<8x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_24 = memref.expand_shape %alloc_23 [[0], [1], [2, 3], [4]] output_shape [8, 2, 2, 1, 2] : memref<8x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x2x1x2xi16, #hivm.address_space<ub>>
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<8x2x2x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_4_outlined_vf_0(%alloc_8, %alloc_22, %expand_shape_24, %alloc_25) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, memref<8x2x2x1x2xi16, #hivm.address_space<ub>>, memref<8x2x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_26 = memref.collapse_shape %alloc_25 [[0], [1], [2, 3], [4]] : memref<8x2x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x4x2xi16, #hivm.address_space<ub>>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<8x2x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_4_outlined_vf_1(%alloc_25, %alloc_27) {hivm.vector_function, no_inline} : (memref<8x2x2x2x2xi16, #hivm.address_space<ub>>, memref<8x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_28 = memref.collapse_shape %alloc_27 [[0], [1], [2, 3]] : memref<8x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %alloc_7 [[0, 1, 2], [3, 4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<8x4x2xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_29 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x4x2xi16, #hivm.address_space<ub>>
    %expand_shape_30 = memref.expand_shape %collapse_shape_28 [[0], [1], [2, 3]] output_shape [8, 2, 4, 1] : memref<8x2x4xi16, #hivm.address_space<ub>> into memref<8x2x4x1xi16, #hivm.address_space<ub>>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<8x2x4x2xi16, #hivm.address_space<ub>>
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<8x4x2xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_32 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x4x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_29 : memref<8x4x2xi16, #hivm.address_space<ub>>) outs(%alloc_32 : memref<8x4x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_5_outlined_vf_0(%expand_shape_30, %collapse_shape_26, %alloc_8, %alloc_31, %alloc_32) {hivm.vector_function, no_inline} : (memref<8x2x4x1xi16, #hivm.address_space<ub>>, memref<8x2x4x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>, memref<8x2x4x2xi16, #hivm.address_space<ub>>, memref<8x4x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_33 = memref.expand_shape %alloc_31 [[0, 1, 2], [3], [4, 5], [6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<8x2x4x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_34 = memref.expand_shape %alloc_32 [[0, 1, 2], [3, 4], [5]] output_shape [2, 2, 2, 2, 2, 2] : memref<8x4x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_35 = memref.collapse_shape %alloc_7 [[0, 1], [2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_35 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x2x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %expand_shape_33 [[0, 1], [2], [3], [4], [5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x2x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_37 = memref.collapse_shape %expand_shape_34 [[0, 1], [2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x2x4xi16, #hivm.address_space<ub>>
    %expand_shape_38 = memref.expand_shape %collapse_shape_37 [[0], [1, 2], [3], [4]] output_shape [4, 2, 1, 2, 4] : memref<4x2x2x4xi16, #hivm.address_space<ub>> into memref<4x2x1x2x4xi16, #hivm.address_space<ub>>
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x2x4xi16, #hivm.address_space<ub>>
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_40 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x2x2x4xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_35 : memref<4x2x2x4xi16, #hivm.address_space<ub>>) outs(%alloc_40 : memref<4x2x2x4xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_6_outlined_vf_0(%alloc_8, %collapse_shape_36, %expand_shape_38, %alloc_39, %alloc_40) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<4x2x2x2x4xi16, #hivm.address_space<ub>>, memref<4x2x1x2x4xi16, #hivm.address_space<ub>>, memref<4x2x2x2x4xi16, #hivm.address_space<ub>>, memref<4x2x2x4xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_41 = memref.expand_shape %alloc_39 [[0], [1], [2], [3], [4, 5]] output_shape [4, 2, 2, 2, 2, 2] : memref<4x2x2x2x4xi16, #hivm.address_space<ub>> into memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_42 = memref.collapse_shape %alloc_7 [[0, 1], [2], [3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x2x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_42 {hivm.stride_align_dims = array<i32: 4>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_43 = memref.expand_shape %alloc_40 [[0], [1], [2, 3], [4, 5]] output_shape [4, 2, 2, 1, 2, 2] : memref<4x2x2x4xi16, #hivm.address_space<ub>> into memref<4x2x2x1x2x2xi16, #hivm.address_space<ub>>
    %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %alloc_45 = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_45 {hivm.stride_align_dims = array<i32: 4>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4x2x2x2x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_42 : memref<4x2x2x2x2xi16, #hivm.address_space<ub>>) outs(%alloc_45 : memref<4x2x2x2x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_7_outlined_vf_0(%alloc_8, %expand_shape_41, %expand_shape_43, %alloc_44, %alloc_45) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>>, memref<4x2x2x1x2x2xi16, #hivm.address_space<ub>>, memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>>, memref<4x2x2x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_46 = memref.collapse_shape %alloc_45 [[0], [1], [2, 3], [4]] : memref<4x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x4x2xi16, #hivm.address_space<ub>>
    %collapse_shape_47 = memref.collapse_shape %alloc_44 [[0], [1], [2, 3], [4], [5]] : memref<4x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x4x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_48 = memref.expand_shape %collapse_shape_46 [[0], [1], [2, 3], [4]] output_shape [4, 2, 4, 1, 2] : memref<4x2x4x2xi16, #hivm.address_space<ub>> into memref<4x2x4x1x2xi16, #hivm.address_space<ub>>
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<4x2x4x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_8_outlined_vf_0(%alloc_8, %collapse_shape_47, %expand_shape_48, %alloc_49) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<4x2x4x2x2xi16, #hivm.address_space<ub>>, memref<4x2x4x1x2xi16, #hivm.address_space<ub>>, memref<4x2x4x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_50 = memref.collapse_shape %alloc_49 [[0], [1], [2, 3], [4]] : memref<4x2x4x2x2xi16, #hivm.address_space<ub>> into memref<4x2x8x2xi16, #hivm.address_space<ub>>
    %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<4x2x4x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_8_outlined_vf_1(%alloc_49, %alloc_51) {hivm.vector_function, no_inline} : (memref<4x2x4x2x2xi16, #hivm.address_space<ub>>, memref<4x2x4x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_52 = memref.collapse_shape %alloc_51 [[0], [1], [2, 3]] : memref<4x2x4x2xi16, #hivm.address_space<ub>> into memref<4x2x8xi16, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %alloc_7 [[0, 1], [2, 3, 4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x8x2xi16, #hivm.address_space<ub>>
    %expand_shape_54 = memref.expand_shape %collapse_shape_52 [[0], [1], [2, 3]] output_shape [4, 2, 8, 1] : memref<4x2x8xi16, #hivm.address_space<ub>> into memref<4x2x8x1xi16, #hivm.address_space<ub>>
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<4x2x8x2xi16, #hivm.address_space<ub>>
    %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<4x8x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_53 : memref<4x8x2xi16, #hivm.address_space<ub>>) outs(%alloc_56 : memref<4x8x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_9_outlined_vf_0(%expand_shape_54, %collapse_shape_50, %alloc_8, %alloc_55, %alloc_56) {hivm.vector_function, no_inline} : (memref<4x2x8x1xi16, #hivm.address_space<ub>>, memref<4x2x8x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>, memref<4x2x8x2xi16, #hivm.address_space<ub>>, memref<4x8x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_57 = memref.expand_shape %alloc_55 [[0, 1], [2], [3, 4, 5], [6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<4x2x8x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_58 = memref.expand_shape %alloc_56 [[0, 1], [2, 3, 4], [5]] output_shape [2, 2, 2, 2, 2, 2] : memref<4x8x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_59 = memref.collapse_shape %alloc_7 [[0], [1], [2], [3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x2x8xi16, #hivm.address_space<ub>>
    %collapse_shape_60 = memref.collapse_shape %expand_shape_57 [[0], [1], [2], [3], [4, 5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x8xi16, #hivm.address_space<ub>>
    %collapse_shape_61 = memref.collapse_shape %expand_shape_58 [[0], [1], [2], [3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x2x8xi16, #hivm.address_space<ub>>
    %expand_shape_62 = memref.expand_shape %collapse_shape_61 [[0], [1, 2], [3], [4]] output_shape [2, 2, 1, 2, 8] : memref<2x2x2x8xi16, #hivm.address_space<ub>> into memref<2x2x1x2x8xi16, #hivm.address_space<ub>>
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x8xi16, #hivm.address_space<ub>>
    %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x8xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_59 : memref<2x2x2x8xi16, #hivm.address_space<ub>>) outs(%alloc_64 : memref<2x2x2x8xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_10_outlined_vf_0(%alloc_8, %collapse_shape_60, %expand_shape_62, %alloc_63, %alloc_64) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, memref<2x2x1x2x8xi16, #hivm.address_space<ub>>, memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, memref<2x2x2x8xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_65 = memref.expand_shape %alloc_63 [[0], [1], [2], [3], [4, 5]] output_shape [2, 2, 2, 2, 2, 4] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_66 = memref.collapse_shape %alloc_7 [[0], [1], [2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_66 {hivm.stride_align_dims = array<i32: 4>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x2x4xi16, #hivm.address_space<ub>>
    %expand_shape_67 = memref.expand_shape %alloc_64 [[0], [1], [2, 3], [4, 5]] output_shape [2, 2, 2, 1, 2, 4] : memref<2x2x2x8xi16, #hivm.address_space<ub>> into memref<2x2x2x1x2x4xi16, #hivm.address_space<ub>>
    %alloc_68 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>>
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_69 {hivm.stride_align_dims = array<i32: 4>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x2x2x4xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_66 : memref<2x2x2x2x4xi16, #hivm.address_space<ub>>) outs(%alloc_69 : memref<2x2x2x2x4xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_11_outlined_vf_0(%alloc_8, %expand_shape_65, %expand_shape_67, %alloc_68, %alloc_69) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>>, memref<2x2x2x1x2x4xi16, #hivm.address_space<ub>>, memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>>, memref<2x2x2x2x4xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_70 = memref.expand_shape %alloc_68 [[0], [1], [2], [3], [4], [5, 6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<2x2x2x2x2x4xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_71 = memref.expand_shape %alloc_69 [[0], [1], [2], [3], [4, 5]] output_shape [2, 2, 2, 2, 2, 2] : memref<2x2x2x2x4xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_72 = memref.collapse_shape %alloc_7 [[0], [1], [2, 3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x4x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_72 {hivm.stride_align_dims = array<i32: 4>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x4x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_73 = memref.collapse_shape %expand_shape_70 [[0], [1], [2, 3], [4], [5], [6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_74 = memref.collapse_shape %expand_shape_71 [[0], [1], [2, 3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x4x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_75 = memref.expand_shape %collapse_shape_74 [[0], [1], [2, 3], [4], [5]] output_shape [2, 2, 4, 1, 2, 2] : memref<2x2x4x2x2xi16, #hivm.address_space<ub>> into memref<2x2x4x1x2x2xi16, #hivm.address_space<ub>>
    %alloc_76 = memref.alloc() {alignment = 64 : i64} : memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>>
    %alloc_77 = memref.alloc() {alignment = 64 : i64} : memref<2x2x4x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_77 {hivm.stride_align_dims = array<i32: 4>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x2x4x2x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_72 : memref<2x2x4x2x2xi16, #hivm.address_space<ub>>) outs(%alloc_77 : memref<2x2x4x2x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_12_outlined_vf_0(%alloc_8, %collapse_shape_73, %expand_shape_75, %alloc_76, %alloc_77) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>>, memref<2x2x4x1x2x2xi16, #hivm.address_space<ub>>, memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>>, memref<2x2x4x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_78 = memref.collapse_shape %alloc_77 [[0], [1], [2, 3], [4]] : memref<2x2x4x2x2xi16, #hivm.address_space<ub>> into memref<2x2x8x2xi16, #hivm.address_space<ub>>
    %collapse_shape_79 = memref.collapse_shape %alloc_76 [[0], [1], [2, 3], [4], [5]] : memref<2x2x4x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x8x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_80 = memref.expand_shape %collapse_shape_78 [[0], [1], [2, 3], [4]] output_shape [2, 2, 8, 1, 2] : memref<2x2x8x2xi16, #hivm.address_space<ub>> into memref<2x2x8x1x2xi16, #hivm.address_space<ub>>
    %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<2x2x8x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_13_outlined_vf_0(%alloc_8, %collapse_shape_79, %expand_shape_80, %alloc_81) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x8x2x2xi16, #hivm.address_space<ub>>, memref<2x2x8x1x2xi16, #hivm.address_space<ub>>, memref<2x2x8x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_82 = memref.collapse_shape %alloc_81 [[0], [1], [2, 3], [4]] : memref<2x2x8x2x2xi16, #hivm.address_space<ub>> into memref<2x2x16x2xi16, #hivm.address_space<ub>>
    %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<2x2x8x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_13_outlined_vf_1(%alloc_81, %alloc_83) {hivm.vector_function, no_inline} : (memref<2x2x8x2x2xi16, #hivm.address_space<ub>>, memref<2x2x8x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_84 = memref.collapse_shape %alloc_83 [[0], [1], [2, 3]] : memref<2x2x8x2xi16, #hivm.address_space<ub>> into memref<2x2x16xi16, #hivm.address_space<ub>>
    %collapse_shape_85 = memref.collapse_shape %alloc_7 [[0], [1, 2, 3, 4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x16x2xi16, #hivm.address_space<ub>>
    %expand_shape_86 = memref.expand_shape %collapse_shape_84 [[0], [1], [2, 3]] output_shape [2, 2, 16, 1] : memref<2x2x16xi16, #hivm.address_space<ub>> into memref<2x2x16x1xi16, #hivm.address_space<ub>>
    %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<2x2x16x2xi16, #hivm.address_space<ub>>
    %alloc_88 = memref.alloc() {alignment = 64 : i64} : memref<2x16x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_85 : memref<2x16x2xi16, #hivm.address_space<ub>>) outs(%alloc_88 : memref<2x16x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_14_outlined_vf_0(%expand_shape_86, %collapse_shape_82, %alloc_8, %alloc_87, %alloc_88) {hivm.vector_function, no_inline} : (memref<2x2x16x1xi16, #hivm.address_space<ub>>, memref<2x2x16x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x16x2xi16, #hivm.address_space<ub>>, memref<2x16x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_89 = memref.expand_shape %alloc_87 [[0], [1], [2, 3, 4, 5], [6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<2x2x16x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_90 = memref.expand_shape %alloc_88 [[0], [1, 2, 3, 4], [5]] output_shape [2, 2, 2, 2, 2, 2] : memref<2x16x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_91 = memref.collapse_shape %alloc_7 [[0], [1], [2, 3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x16xi16, #hivm.address_space<ub>>
    %collapse_shape_92 = memref.collapse_shape %expand_shape_89 [[0], [1], [2], [3, 4, 5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x2x16xi16, #hivm.address_space<ub>>
    %collapse_shape_93 = memref.collapse_shape %expand_shape_90 [[0], [1], [2, 3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x16xi16, #hivm.address_space<ub>>
    %expand_shape_94 = memref.expand_shape %collapse_shape_93 [[0, 1], [2], [3]] output_shape [2, 1, 2, 16] : memref<2x2x16xi16, #hivm.address_space<ub>> into memref<2x1x2x16xi16, #hivm.address_space<ub>>
    %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x16xi16, #hivm.address_space<ub>>
    %alloc_96 = memref.alloc() {alignment = 64 : i64} : memref<2x2x16xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_91 : memref<2x2x16xi16, #hivm.address_space<ub>>) outs(%alloc_96 : memref<2x2x16xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_15_outlined_vf_0(%alloc_8, %collapse_shape_92, %expand_shape_94, %alloc_95, %alloc_96) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x2x16xi16, #hivm.address_space<ub>>, memref<2x1x2x16xi16, #hivm.address_space<ub>>, memref<2x2x2x16xi16, #hivm.address_space<ub>>, memref<2x2x16xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_97 = memref.expand_shape %alloc_95 [[0], [1], [2], [3, 4]] output_shape [2, 2, 2, 2, 8] : memref<2x2x2x16xi16, #hivm.address_space<ub>> into memref<2x2x2x2x8xi16, #hivm.address_space<ub>>
    %expand_shape_98 = memref.expand_shape %alloc_96 [[0], [1, 2], [3, 4]] output_shape [2, 2, 1, 2, 8] : memref<2x2x16xi16, #hivm.address_space<ub>> into memref<2x2x1x2x8xi16, #hivm.address_space<ub>>
    %alloc_99 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x8xi16, #hivm.address_space<ub>>
    %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x8xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_59 : memref<2x2x2x8xi16, #hivm.address_space<ub>>) outs(%alloc_100 : memref<2x2x2x8xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_16_outlined_vf_0(%alloc_8, %expand_shape_97, %expand_shape_98, %alloc_99, %alloc_100) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, memref<2x2x1x2x8xi16, #hivm.address_space<ub>>, memref<2x2x2x2x8xi16, #hivm.address_space<ub>>, memref<2x2x2x8xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_101 = memref.expand_shape %alloc_99 [[0], [1], [2], [3], [4, 5, 6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<2x2x2x2x8xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_102 = memref.expand_shape %alloc_100 [[0], [1], [2], [3, 4, 5]] output_shape [2, 2, 2, 2, 2, 2] : memref<2x2x2x8xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_103 = memref.collapse_shape %alloc_7 [[0], [1, 2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x4x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_103 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x4x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_104 = memref.collapse_shape %expand_shape_101 [[0], [1, 2], [3], [4], [5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x4x2x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_105 = memref.collapse_shape %expand_shape_102 [[0], [1, 2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x4x2x4xi16, #hivm.address_space<ub>>
    %expand_shape_106 = memref.expand_shape %collapse_shape_105 [[0], [1, 2], [3], [4]] output_shape [2, 4, 1, 2, 4] : memref<2x4x2x4xi16, #hivm.address_space<ub>> into memref<2x4x1x2x4xi16, #hivm.address_space<ub>>
    %alloc_107 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x2x4xi16, #hivm.address_space<ub>>
    %alloc_108 = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_108 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x4x2x4xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_103 : memref<2x4x2x4xi16, #hivm.address_space<ub>>) outs(%alloc_108 : memref<2x4x2x4xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_17_outlined_vf_0(%alloc_8, %collapse_shape_104, %expand_shape_106, %alloc_107, %alloc_108) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x4x2x2x4xi16, #hivm.address_space<ub>>, memref<2x4x1x2x4xi16, #hivm.address_space<ub>>, memref<2x4x2x2x4xi16, #hivm.address_space<ub>>, memref<2x4x2x4xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_109 = memref.expand_shape %alloc_107 [[0], [1, 2], [3], [4], [5, 6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<2x4x2x2x4xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_110 = memref.expand_shape %alloc_108 [[0], [1, 2], [3], [4, 5]] output_shape [2, 2, 2, 2, 2, 2] : memref<2x4x2x4xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_111 = memref.collapse_shape %alloc_7 [[0], [1, 2, 3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x8x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_111 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x8x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_112 = memref.collapse_shape %expand_shape_109 [[0], [1, 2, 3], [4], [5], [6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x8x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_113 = memref.collapse_shape %expand_shape_110 [[0], [1, 2, 3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x8x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_114 = memref.expand_shape %collapse_shape_113 [[0], [1, 2], [3], [4]] output_shape [2, 8, 1, 2, 2] : memref<2x8x2x2xi16, #hivm.address_space<ub>> into memref<2x8x1x2x2xi16, #hivm.address_space<ub>>
    %alloc_115 = memref.alloc() {alignment = 64 : i64} : memref<2x8x2x2x2xi16, #hivm.address_space<ub>>
    %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<2x8x2x2xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_116 {hivm.stride_align_dims = array<i32: 3>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x8x2x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_111 : memref<2x8x2x2xi16, #hivm.address_space<ub>>) outs(%alloc_116 : memref<2x8x2x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_18_outlined_vf_0(%alloc_8, %collapse_shape_112, %expand_shape_114, %alloc_115, %alloc_116) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x8x2x2x2xi16, #hivm.address_space<ub>>, memref<2x8x1x2x2xi16, #hivm.address_space<ub>>, memref<2x8x2x2x2xi16, #hivm.address_space<ub>>, memref<2x8x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_117 = memref.collapse_shape %alloc_116 [[0], [1, 2], [3]] : memref<2x8x2x2xi16, #hivm.address_space<ub>> into memref<2x16x2xi16, #hivm.address_space<ub>>
    %collapse_shape_118 = memref.collapse_shape %alloc_115 [[0], [1, 2], [3], [4]] : memref<2x8x2x2x2xi16, #hivm.address_space<ub>> into memref<2x16x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_119 = memref.expand_shape %collapse_shape_117 [[0], [1, 2], [3]] output_shape [2, 16, 1, 2] : memref<2x16x2xi16, #hivm.address_space<ub>> into memref<2x16x1x2xi16, #hivm.address_space<ub>>
    %alloc_120 = memref.alloc() {alignment = 64 : i64} : memref<2x16x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_19_outlined_vf_0(%alloc_8, %collapse_shape_118, %expand_shape_119, %alloc_120) {hivm.vector_function, no_inline} : (memref<2x2xi32, #hivm.address_space<ub>>, memref<2x16x2x2xi16, #hivm.address_space<ub>>, memref<2x16x1x2xi16, #hivm.address_space<ub>>, memref<2x16x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_121 = memref.collapse_shape %alloc_120 [[0], [1, 2], [3]] : memref<2x16x2x2xi16, #hivm.address_space<ub>> into memref<2x32x2xi16, #hivm.address_space<ub>>
    %alloc_122 = memref.alloc() {alignment = 64 : i64} : memref<2x16x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_19_outlined_vf_1(%alloc_120, %alloc_122) {hivm.vector_function, no_inline} : (memref<2x16x2x2xi16, #hivm.address_space<ub>>, memref<2x16x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_123 = memref.collapse_shape %alloc_122 [[0], [1, 2]] : memref<2x16x2xi16, #hivm.address_space<ub>> into memref<2x32xi16, #hivm.address_space<ub>>
    %collapse_shape_124 = memref.collapse_shape %alloc_7 [[0, 1, 2, 3, 4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<32x2xi16, #hivm.address_space<ub>>
    %expand_shape_125 = memref.expand_shape %collapse_shape_123 [[0], [1, 2]] output_shape [2, 32, 1] : memref<2x32xi16, #hivm.address_space<ub>> into memref<2x32x1xi16, #hivm.address_space<ub>>
    %alloc_126 = memref.alloc() {alignment = 64 : i64} : memref<2x32x2xi16, #hivm.address_space<ub>>
    %alloc_127 = memref.alloc() {alignment = 64 : i64} : memref<32x2xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_124 : memref<32x2xi16, #hivm.address_space<ub>>) outs(%alloc_127 : memref<32x2xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_20_outlined_vf_0(%expand_shape_125, %collapse_shape_121, %alloc_8, %alloc_126, %alloc_127) {hivm.vector_function, no_inline} : (memref<2x32x1xi16, #hivm.address_space<ub>>, memref<2x32x2xi16, #hivm.address_space<ub>>, memref<2x2xi32, #hivm.address_space<ub>>, memref<2x32x2xi16, #hivm.address_space<ub>>, memref<32x2xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_128 = memref.expand_shape %alloc_126 [[0], [1, 2, 3, 4, 5], [6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<2x32x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_129 = memref.expand_shape %alloc_127 [[0, 1, 2, 3, 4], [5]] output_shape [2, 2, 2, 2, 2, 2] : memref<32x2xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_130 = memref.collapse_shape %alloc_7 [[0], [1, 2, 3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x32xi16, #hivm.address_space<ub>>
    %collapse_shape_131 = memref.collapse_shape %expand_shape_128 [[0], [1], [2, 3, 4, 5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x2x32xi16, #hivm.address_space<ub>>
    %collapse_shape_132 = memref.collapse_shape %expand_shape_129 [[0], [1, 2, 3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<2x32xi16, #hivm.address_space<ub>>
    %expand_shape_133 = memref.expand_shape %collapse_shape_132 [[0, 1], [2]] output_shape [1, 2, 32] : memref<2x32xi16, #hivm.address_space<ub>> into memref<1x2x32xi16, #hivm.address_space<ub>>
    %alloc_134 = memref.alloc() {alignment = 64 : i64} : memref<2x2x32xi16, #hivm.address_space<ub>>
    %alloc_135 = memref.alloc() {alignment = 64 : i64} : memref<2x32xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_130 : memref<2x32xi16, #hivm.address_space<ub>>) outs(%alloc_135 : memref<2x32xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_21_outlined_vf_0(%alloc_3, %collapse_shape_131, %expand_shape_133, %alloc_134, %alloc_135) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<2x2x32xi16, #hivm.address_space<ub>>, memref<1x2x32xi16, #hivm.address_space<ub>>, memref<2x2x32xi16, #hivm.address_space<ub>>, memref<2x32xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_136 = memref.expand_shape %alloc_134 [[0], [1], [2, 3]] output_shape [2, 2, 2, 16] : memref<2x2x32xi16, #hivm.address_space<ub>> into memref<2x2x2x16xi16, #hivm.address_space<ub>>
    %expand_shape_137 = memref.expand_shape %alloc_135 [[0, 1], [2, 3]] output_shape [2, 1, 2, 16] : memref<2x32xi16, #hivm.address_space<ub>> into memref<2x1x2x16xi16, #hivm.address_space<ub>>
    %alloc_138 = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x16xi16, #hivm.address_space<ub>>
    %alloc_139 = memref.alloc() {alignment = 64 : i64} : memref<2x2x16xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_91 : memref<2x2x16xi16, #hivm.address_space<ub>>) outs(%alloc_139 : memref<2x2x16xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_22_outlined_vf_0(%alloc_3, %expand_shape_136, %expand_shape_137, %alloc_138, %alloc_139) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<2x2x2x16xi16, #hivm.address_space<ub>>, memref<2x1x2x16xi16, #hivm.address_space<ub>>, memref<2x2x2x16xi16, #hivm.address_space<ub>>, memref<2x2x16xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_140 = memref.expand_shape %alloc_138 [[0], [1], [2], [3, 4, 5, 6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<2x2x2x16xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_141 = memref.expand_shape %alloc_139 [[0], [1], [2, 3, 4, 5]] output_shape [2, 2, 2, 2, 2, 2] : memref<2x2x16xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_142 = memref.collapse_shape %alloc_7 [[0, 1], [2], [3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x8xi16, #hivm.address_space<ub>>
    %collapse_shape_143 = memref.collapse_shape %expand_shape_140 [[0, 1], [2], [3], [4, 5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x2x8xi16, #hivm.address_space<ub>>
    %collapse_shape_144 = memref.collapse_shape %expand_shape_141 [[0, 1], [2], [3, 4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<4x2x8xi16, #hivm.address_space<ub>>
    %expand_shape_145 = memref.expand_shape %collapse_shape_144 [[0, 1], [2], [3]] output_shape [4, 1, 2, 8] : memref<4x2x8xi16, #hivm.address_space<ub>> into memref<4x1x2x8xi16, #hivm.address_space<ub>>
    %alloc_146 = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x8xi16, #hivm.address_space<ub>>
    %alloc_147 = memref.alloc() {alignment = 64 : i64} : memref<4x2x8xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_142 : memref<4x2x8xi16, #hivm.address_space<ub>>) outs(%alloc_147 : memref<4x2x8xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_23_outlined_vf_0(%alloc_3, %collapse_shape_143, %expand_shape_145, %alloc_146, %alloc_147) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<4x2x2x8xi16, #hivm.address_space<ub>>, memref<4x1x2x8xi16, #hivm.address_space<ub>>, memref<4x2x2x8xi16, #hivm.address_space<ub>>, memref<4x2x8xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_148 = memref.expand_shape %alloc_146 [[0, 1], [2], [3], [4, 5, 6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<4x2x2x8xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_149 = memref.expand_shape %alloc_147 [[0, 1], [2], [3, 4, 5]] output_shape [2, 2, 2, 2, 2, 2] : memref<4x2x8xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_150 = memref.collapse_shape %alloc_7 [[0, 1, 2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %collapse_shape_150 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_151 = memref.collapse_shape %expand_shape_148 [[0, 1, 2], [3], [4], [5, 6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x2x4xi16, #hivm.address_space<ub>>
    %collapse_shape_152 = memref.collapse_shape %expand_shape_149 [[0, 1, 2], [3], [4, 5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<8x2x4xi16, #hivm.address_space<ub>>
    %expand_shape_153 = memref.expand_shape %collapse_shape_152 [[0, 1], [2], [3]] output_shape [8, 1, 2, 4] : memref<8x2x4xi16, #hivm.address_space<ub>> into memref<8x1x2x4xi16, #hivm.address_space<ub>>
    %alloc_154 = memref.alloc() {alignment = 64 : i64} : memref<8x2x2x4xi16, #hivm.address_space<ub>>
    %alloc_155 = memref.alloc() {alignment = 64 : i64} : memref<8x2x4xi16, #hivm.address_space<ub>>
    annotation.mark %alloc_155 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x2x4xi16, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_150 : memref<8x2x4xi16, #hivm.address_space<ub>>) outs(%alloc_155 : memref<8x2x4xi16, #hivm.address_space<ub>>)
    call @median_small_flat_kernel_fused_24_outlined_vf_0(%alloc_3, %collapse_shape_151, %expand_shape_153, %alloc_154, %alloc_155) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<8x2x2x4xi16, #hivm.address_space<ub>>, memref<8x1x2x4xi16, #hivm.address_space<ub>>, memref<8x2x2x4xi16, #hivm.address_space<ub>>, memref<8x2x4xi16, #hivm.address_space<ub>>) -> ()
    %expand_shape_156 = memref.expand_shape %alloc_154 [[0, 1, 2], [3], [4], [5, 6]] output_shape [2, 2, 2, 2, 2, 2, 2] : memref<8x2x2x4xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_157 = memref.expand_shape %alloc_155 [[0, 1, 2], [3], [4, 5]] output_shape [2, 2, 2, 2, 2, 2] : memref<8x2x4xi16, #hivm.address_space<ub>> into memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_158 = memref.collapse_shape %expand_shape_156 [[0, 1, 2, 3], [4], [5], [6]] : memref<2x2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<16x2x2x2xi16, #hivm.address_space<ub>>
    %collapse_shape_159 = memref.collapse_shape %expand_shape_157 [[0, 1, 2, 3], [4], [5]] : memref<2x2x2x2x2x2xi16, #hivm.address_space<ub>> into memref<16x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_160 = memref.expand_shape %collapse_shape_159 [[0, 1], [2], [3]] output_shape [16, 1, 2, 2] : memref<16x2x2xi16, #hivm.address_space<ub>> into memref<16x1x2x2xi16, #hivm.address_space<ub>>
    %alloc_161 = memref.alloc() {alignment = 64 : i64} : memref<16x2x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_25_outlined_vf_0(%alloc_3, %collapse_shape_158, %expand_shape_160, %alloc_161, %collapse_shape_15) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<16x2x2x2xi16, #hivm.address_space<ub>>, memref<16x1x2x2xi16, #hivm.address_space<ub>>, memref<16x2x2x2xi16, #hivm.address_space<ub>>, memref<16x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_162 = memref.collapse_shape %alloc_161 [[0, 1], [2], [3]] : memref<16x2x2x2xi16, #hivm.address_space<ub>> into memref<32x2x2xi16, #hivm.address_space<ub>>
    %expand_shape_163 = memref.expand_shape %collapse_shape_124 [[0, 1], [2]] output_shape [32, 1, 2] : memref<32x2xi16, #hivm.address_space<ub>> into memref<32x1x2xi16, #hivm.address_space<ub>>
    %alloc_164 = memref.alloc() {alignment = 64 : i64} : memref<32x2x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_26_outlined_vf_0(%alloc_3, %collapse_shape_162, %expand_shape_163, %alloc_164) {hivm.vector_function, no_inline} : (memref<2xi32, #hivm.address_space<ub>>, memref<32x2x2xi16, #hivm.address_space<ub>>, memref<32x1x2xi16, #hivm.address_space<ub>>, memref<32x2x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_165 = memref.collapse_shape %alloc_164 [[0, 1], [2]] : memref<32x2x2xi16, #hivm.address_space<ub>> into memref<64x2xi16, #hivm.address_space<ub>>
    %alloc_166 = memref.alloc() {alignment = 64 : i64} : memref<32x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_26_outlined_vf_1(%alloc_164, %alloc_166) {hivm.vector_function, no_inline} : (memref<32x2x2xi16, #hivm.address_space<ub>>, memref<32x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_167 = memref.collapse_shape %alloc_166 [[0, 1]] : memref<32x2xi16, #hivm.address_space<ub>> into memref<64xi16, #hivm.address_space<ub>>
    %expand_shape_168 = memref.expand_shape %collapse_shape_167 [[0, 1]] output_shape [64, 1] : memref<64xi16, #hivm.address_space<ub>> into memref<64x1xi16, #hivm.address_space<ub>>
    %alloc_169 = memref.alloc() {alignment = 64 : i64} : memref<64x2xi16, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_27_outlined_vf_0(%expand_shape_168, %collapse_shape_165, %alloc_3, %alloc_169) {hivm.vector_function, no_inline} : (memref<64x1xi16, #hivm.address_space<ub>>, memref<64x2xi16, #hivm.address_space<ub>>, memref<2xi32, #hivm.address_space<ub>>, memref<64x2xi16, #hivm.address_space<ub>>) -> ()
    %collapse_shape_170 = memref.collapse_shape %alloc_169 [[0, 1]] : memref<64x2xi16, #hivm.address_space<ub>> into memref<128xi16, #hivm.address_space<ub>>
    %subview_171 = memref.subview %collapse_shape_170[0] [44] [1] : memref<128xi16, #hivm.address_space<ub>> to memref<44xi16, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%alloc : memref<44xi16, #hivm.address_space<ub>>) outs(%subview_171 : memref<44xi16, strided<[1]>, #hivm.address_space<ub>>)
    %subview_172 = memref.subview %collapse_shape_170[0] [45] [1] : memref<128xi16, #hivm.address_space<ub>> to memref<45xi16, strided<[1]>, #hivm.address_space<ub>>
    %subview_173 = memref.subview %alloc_0[0] [45] [1] : memref<128xi16, #hivm.address_space<ub>> to memref<45xi16, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%subview_172 : memref<45xi16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_173 : memref<45xi16, strided<[1]>, #hivm.address_space<ub>>)
    %alloc_174 = memref.alloc() {alignment = 64 : i64} : memref<i32, #hivm.address_space<ub>>
    memref.store %c0_i32, %alloc_174[] : memref<i32, #hivm.address_space<ub>>
    call @median_small_flat_kernel_fused_28_outlined_vf_0(%alloc_0, %alloc_174) {hivm.vector_function, no_inline} : (memref<128xi16, #hivm.address_space<ub>>, memref<i32, #hivm.address_space<ub>>) -> ()
    %2 = memref.load %alloc_174[] : memref<i32, #hivm.address_space<ub>>
    %3 = arith.trunci %2 : i32 to i16
    %alloc_175 = memref.alloc() {alignment = 64 : i64} : memref<1xi16, #hivm.address_space<ub>>
    memref.store %3, %alloc_175[%c0] : memref<1xi16, #hivm.address_space<ub>>
    %reinterpret_cast_176 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi16, #hivm.address_space<gm>> to memref<1xi16, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%alloc_175 : memref<1xi16, #hivm.address_space<ub>>) outs(%reinterpret_cast_176 : memref<1xi16, strided<[1]>, #hivm.address_space<gm>>)
    hivm.hir.set_ctrl true at ctrl[60]
    return
  }
}
