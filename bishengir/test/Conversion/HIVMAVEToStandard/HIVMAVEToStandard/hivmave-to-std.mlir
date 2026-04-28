// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivmave-to-std %s --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @vadd_i64
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
func.func @vadd_i64(%arg0: memref<741xi64, #hivm.address_space<ub>>, %arg1: memref<741xi64, #hivm.address_space<ub>>, %arg2: memref<741xi64, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c741 = arith.constant 741 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c741 step %c64 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 741, 64)>(%arg3)
    %subview = memref.subview %arg0[%arg3] [%0] [1] : memref<741xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [%0] [1] : memref<741xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [%0] [1] : memref<741xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %0 : vector<64xi1>, index
    // CHECK: func.call @vload_NORM_int64_t_rank1
    // CHECK: func.call @vload_NORM_int64_t_rank1
    // CHECK: func.call @_mlir_ciface_vadd_int64_t
    // CHECK: func.call @masked_store_NORM_B64_int64_t_rank1
    %1 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    %2 = ave.hir.vload <NORM> %subview_0[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    %3 = ave.hir.vadd %1, %2, %res : vector<64xi64>, vector<64xi1>
    ave.hir.masked_store <NORM_B64> %subview_1[%c0], %res, %3 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
  }
  return
}
}

// -----

// CHECK-LABEL: func.func @vdiv_i64
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
  func.func @vdiv_i64(%arg0: memref<192xi64, #hivm.address_space<ub>>, %arg1: memref<192xi64, #hivm.address_space<ub>>, %arg2: memref<192xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c192 = arith.constant 192 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c192 step %c64 {
      %subview = memref.subview %arg0[%arg3] [64] [1] : memref<192xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3] [64] [1] : memref<192xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg3] [64] [1] : memref<192xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: func.call @vload_NORM_int64_t_rank1
      // CHECK: func.call @vload_NORM_int64_t_rank1
      %0 = ave.hir.vload <NORM> %subview[%c0] : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %1 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      %3 = ave.hir.vdiv %0, %1, %2 : vector<64xi64>, vector<64xi1>
      // CHECK: func.call @_mlir_ciface_vdiv_int64_t
      %4 = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: func.call @masked_store_NORM_B64_int64_t_rank1
      ave.hir.masked_store <NORM_B64> %subview_1[%c0], %4, %3 : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @vdiv_u64
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
  func.func @vdiv_u64(%arg0: memref<192xi64, #hivm.address_space<ub>>, %arg1: memref<192xi64, #hivm.address_space<ub>>, %arg2: memref<192xi64, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c192 = arith.constant 192 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c192 step %c64 {
      %subview = memref.subview %arg0[%arg3] [64] [1] : memref<192xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3] [64] [1] : memref<192xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg3] [64] [1] : memref<192xi64, #hivm.address_space<ub>> to memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      // CHECK: func.call @vload_NORM_int64_t_rank1
      // CHECK: func.call @vload_NORM_int64_t_rank1
      %0 = ave.hir.vload <NORM> %subview[%c0] : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %1 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      %3 = ave.hir.vdiv %0, %1, %2 {cast = #hivm.cast<cast_unsigned>} : vector<64xi64>, vector<64xi1>
      // CHECK: func.call @_mlir_ciface_vdiv_uint64_t
      %4 = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: func.call @masked_store_NORM_B64_int64_t_rank1
      ave.hir.masked_store <NORM_B64> %subview_1[%c0], %4, %3 : memref<64xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @reduction_test
#map = affine_map<(d0) -> (-d0 + 123, 64)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
  func.func @reduction_test(%arg0: memref<7x123xi64, strided<[124, 1]>, #hivm.address_space<ub>>, %arg1: memref<7xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = -1 : i32, hivm.vector_function} {
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c123 = arith.constant 123 : index
    scf.for %arg2 = %c0 to %c7 step %c1 {
      %subview = memref.subview %arg1[%arg2] [1] [1] : memref<7xi64, #hivm.address_space<ub>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %subview[0] [1] [1] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i64, strided<[], offset: ?>, #hivm.address_space<ub>>
      // CHECK: func.call @vload_BRC_B64_int64_t_rank0
      %0 = ave.hir.vload <BRC_B64> %subview_0[] : memref<i64, strided<[], offset: ?>, #hivm.address_space<ub>> into vector<1xi64>
      %1 = builtin.unrealized_conversion_cast %0 : vector<1xi64> to vector<64xi64>
      // CHECK: llvm.getelementptr
      // CHECK: llvm.load
      // CHECK: llvm.getelementptr
      // CHECK: llvm.load
      %2 = scf.for %arg3 = %c0 to %c123 step %c64 iter_args(%arg4 = %1) -> (vector<64xi64>) {
        %5 = affine.min #map(%arg3)
        %subview_1 = memref.subview %arg0[%arg2, %arg3] [1, %5] [1, 1] : memref<7x123xi64, strided<[124, 1]>, #hivm.address_space<ub>> to memref<1x?xi64, strided<[124, 1], offset: ?>, #hivm.address_space<ub>>
        %res, %new_true_shape = ave.hir.plt %5 : vector<64xi1>, index
        %subview_2 = memref.subview %subview_1[0, 0] [1, %5] [1, 1] : memref<1x?xi64, strided<[124, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, #map1, #hivm.address_space<ub>>
        // CHECK: func.call @vload_NORM_int64_t_rank1
        %6 = ave.hir.vload <NORM> %subview_2[%c0] : memref<?xi64, #map1, #hivm.address_space<ub>> into vector<64xi64>
        // CHECK: func.call @_mlir_ciface_vcadd_int64_t
        %7 = ave.hir.reduction <add>, %6, %res : vector<64xi64>, vector<64xi1> -> vector<64xi64>
        %8 = ave.hir.pge <ALL> : vector<64xi1>
        // CHECK: func.call @_mlir_ciface_vadd_int64_t
        %9 = ave.hir.vadd %arg4, %7, %8 : vector<64xi64>, vector<64xi1>
        scf.yield %9 : vector<64xi64>
      } {element_alignment_bit_width = 32 : i32}
      // CHECK: llvm.alloca
      // CHECK: llvm.getelementptr
      // CHECK: llvm.store
      // CHECK: llvm.getelementptr
      // CHECK: llvm.store
      %3 = builtin.unrealized_conversion_cast %2 : vector<64xi64> to vector<1xi64>
      %4 = ave.hir.pge <ALL> : vector<1xi1>
      // CHECK: func.call @masked_store_ONEPT_B64_int64_t_rank0
      ave.hir.masked_store <ONEPT_B64> %subview_0[], %4, %3 : memref<i64, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi64>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}

// -----

// CHECK-LABEL: func.func @triton_permute_2d_outlined_vf_0
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_permute_2d_outlined_vf_0(%arg0: memref<2x257xi64, strided<[260, 1]>, #hivm.address_space<ub>>, %arg1: memref<257x2xi64, strided<[4, 1]>, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c257 = arith.constant 257 : index
    %c1 = arith.constant 1 : index
    %c260_i64 = arith.constant 260 : i64
    %c0_i64 = arith.constant 0 : i64
    // CHECK: call @_mlir_ciface_vci_int64_t
    %0 = ave.hir.vci %c0_i64, <INCREASE> : i64, vector<64xi64>
    %1 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: call @_mlir_ciface_vmuls_int64_t
    %2 = ave.hir.vmuls %0, %c260_i64, %1 : vector<64xi64>, i64, vector<64xi1>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: call @_mlir_ciface_vdups_int64_t
    %4 = ave.hir.broadcast %c0_i64, %3 : i64, vector<64xi1> -> vector<64xi64>
    %5 = ave.hir.pge <VL2> : vector<64xi1>
    // CHECK: call @_mlir_ciface_vsel_int64_t
    %6 = ave.hir.vsel %5, %2, %4 : vector<64xi1>, vector<64xi64>
    scf.for %arg2 = %c0 to %c257 step %c1 {
      %subview = memref.subview %arg0[0, %arg2] [2, 1] [1, 1] : memref<2x257xi64, strided<[260, 1]>, #hivm.address_space<ub>> to memref<2x1xi64, strided<[260, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, 0] [1, 2] [1, 1] : memref<257x2xi64, strided<[4, 1]>, #hivm.address_space<ub>> to memref<1x2xi64, strided<[4, 1], offset: ?>, #hivm.address_space<ub>>
      %7 = ave.hir.pge <VL2> {mask_op_idx = 0 : i32} : vector<64xi1>
      // CHECK: func.call @vgather_int64_t
      %8 = ave.hir.vgather %subview[%c0, %c0] [%6], %7 : memref<2x1xi64, strided<[260, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi64>, vector<64xi1> into vector<64xi64>
      %subview_1 = memref.subview %subview_0[0, 0] [1, 2] [1, 1] : memref<1x2xi64, strided<[4, 1], offset: ?>, #hivm.address_space<ub>> to memref<2xi64, #map, #hivm.address_space<ub>>
      // CHECK: func.call @masked_store_NORM_B64_int64_t_rank1
      ave.hir.masked_store <NORM_B64> %subview_1[%c0], %7, %8 : memref<2xi64, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}


// -----


// CHECK-LABEL: func.func @vmod_i64
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
func.func @vmod_i64(%arg0: memref<741xi64, #hivm.address_space<ub>>, %arg1: memref<741xi64, #hivm.address_space<ub>>, %arg2: memref<741xi64, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i64 = arith.constant 0 : i64
  %c64 = arith.constant 64 : index
  %c741 = arith.constant 741 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c741 step %c64 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 741, 64)>(%arg3)
    %subview = memref.subview %arg0[%arg3] [%0] [1] : memref<741xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [%0] [1] : memref<741xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [%0] [1] : memref<741xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %0 : vector<64xi1>, index
    // CHECK: func.call @vload_NORM_int64_t_rank1
    // CHECK: func.call @vload_NORM_int64_t_rank1
    // CHECK: func.call @_mlir_ciface_vmod_int64_t
    // CHECK: func.call @masked_store_NORM_B64_int64_t_rank1
    %1 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    %2 = ave.hir.vload <NORM> %subview_0[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    %3 = ave.hir.vmod %1, %2, %res : vector<64xi64>, vector<64xi1>
    ave.hir.masked_store <NORM_B64> %subview_1[%c0], %res, %3 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
  }
  return
}
}


// -----

// CHECK-LABEL: func.func @triton_cmp_1D_outlined_vf_0
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_cmp_1D_outlined_vf_0(%arg0: vector<64xi64>, %arg1: vector<64xi64>, %pred: vector<64xi1>) -> (vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>) {
    // CHECK: call @_mlir_ciface_vcmp_ule_uint64_t
    %0 = ave.hir.vcmp <ULE> %arg0, %arg1, %pred : vector<64xi64>, vector<64xi1> -> vector<64xi1>
    // CHECK: call @_mlir_ciface_vcmp_ult_uint64_t
    %1 = ave.hir.vcmp <ULT> %arg0, %arg1, %pred : vector<64xi64>, vector<64xi1> -> vector<64xi1>
    // CHECK: call @_mlir_ciface_vcmp_uge_uint64_t
    %2 = ave.hir.vcmp <UGE> %arg0, %arg1, %pred : vector<64xi64>, vector<64xi1> -> vector<64xi1>
    // CHECK: call @_mlir_ciface_vcmp_ugt_uint64_t
    %3 = ave.hir.vcmp <UGT> %arg0, %arg1, %pred : vector<64xi64>, vector<64xi1> -> vector<64xi1>
    return %0, %1, %2, %3: vector<64xi1>, vector<64xi1>, vector<64xi1>, vector<64xi1>
  }
}

// -----

// CHECK-LABEL: func.func @triton_sum_2D_dim0_outlined_vf_1
#map = affine_map<(d0) -> (-d0 + 151, 64)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_sum_2D_dim0_outlined_vf_1(%arg0: memref<13x151xi64, #hivm.address_space<ub>>, %arg1: memref<151xi64, #hivm.address_space<ub>>, %arg2: memref<512xi8, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c13 = arith.constant 13 : index
    %c1 = arith.constant 1 : index
    %c151 = arith.constant 151 : index
    %c64 = arith.constant 64 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %c0_i64, %0 : i64, vector<64xi1> -> vector<64xi64>
    scf.for %arg3 = %c0 to %c151 step %c64 {
      %2 = affine.min #map(%arg3)
      %subview = memref.subview %arg1[%arg3] [%2] [1] : memref<151xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %view = memref.view %arg2[%c0][%2] : memref<512xi8, #hivm.address_space<ub>> to memref<1x?xi64, #hivm.address_space<ub>>
      %3 = ave.hir.pltm %arg3, %c151 : vector<64xi1>
      %subview_0 = memref.subview %view[0, 0] [1, %2] [1, 1] : memref<1x?xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      ave.hir.masked_store <NORM_B64> %subview_0[%c0], %3, %1 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
      %subview_1 = memref.subview %view[0, 0] [1, %2] [1, 1] : memref<1x?xi64, #hivm.address_space<ub>> to memref<1x?xi64, strided<[?, 1]>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_1[0, 0] [1, %2] [1, 1] : memref<1x?xi64, strided<[?, 1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      %4 = ave.hir.vload <NORM> %subview_2[%c0] : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> into vector<64xi64>
      %5 = scf.for %arg4 = %c0 to %c13 step %c1 iter_args(%arg5 = %4) -> (vector<64xi64>) {
        %subview_3 = memref.subview %arg0[%arg4, %arg3] [1, %2] [1, 1] : memref<13x151xi64, #hivm.address_space<ub>> to memref<1x?xi64, strided<[151, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_4 = memref.subview %subview_3[0, 0] [1, %2] [1, 1] : memref<1x?xi64, strided<[151, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi64, #map1, #hivm.address_space<ub>>
        // CHECK: func.call @vload_NORM_int64_t_unalign_rank1(%{{.*}}, %{{.*}}, %{{.*}})
        %10 = ave.hir.vload <NORM> %subview_4[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<?xi64, #map1, #hivm.address_space<ub>> into vector<64xi64>
        %11 = ave.hir.pge <ALL> : vector<64xi1>
        %12 = ave.hir.vadd %10, %arg5, %11 : vector<64xi64>, vector<64xi1>
        scf.yield %12 : vector<64xi64>
      } {element_alignment_bit_width = 32 : i32}
      ave.hir.masked_store <NORM_B64> %subview_2[%c0], %3, %5 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
      %6 = ave.hir.vload <NORM> %subview_0[%c0] : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> into vector<64xi64>
      %7 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
      %8 = ave.hir.vadd %7, %6, %3 : vector<64xi64>, vector<64xi1>
      %9 = ave.hir.vsel %3, %8, %6 : vector<64xi1>, vector<64xi64>
      ave.hir.masked_store <NORM_B64> %subview[%c0], %3, %9 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}


// -----

// CHECK-LABEL: func.func @triton_unk_fused_embedding_mul_sigmoid_sum_0_outlined_vf_2
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_unk_fused_embedding_mul_sigmoid_sum_0_outlined_vf_2(%arg0: memref<4x39xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c39 = arith.constant 39 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = ave.hir.pge <ALL> : vector<64xi1>
    %1 = ave.hir.broadcast %c0_i64, %0 : i64, vector<64xi1> -> vector<64xi64>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %subview = memref.subview %arg0[%arg1, 0] [1, 39] [1, 1] : memref<4x39xi64, #hivm.address_space<ub>> to memref<1x39xi64, strided<[39, 1], offset: ?>, #hivm.address_space<ub>>
      %res, %new_true_shape = ave.hir.plt %c39 {mask_op_idx = 0 : i32} : vector<64xi1>, index
      %subview_0 = memref.subview %subview[0, 0] [1, 39] [1, 1] : memref<1x39xi64, strided<[39, 1], offset: ?>, #hivm.address_space<ub>> to memref<39xi64, #map, #hivm.address_space<ub>>
      // CHECK: %[[ARG1:.*]] = arith.constant 39 : i32
      // func.call @masked_store_NORM_B64_int64_t_rank1(%{{.*}}, %{{.*}}, %{{.*}}, %[[ARG1:.*]])
      ave.hir.masked_store <NORM_B64> %subview_0[%c0], %res, %1 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<39xi64, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}

// -----

// CHECK-LABEL: func.func @triton_cast_1d_outlined_vf_0
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_cast_1d_outlined_vf_0(%arg0: memref<13xi8, #hivm.address_space<ub>>, %arg1: memref<13xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c13 = arith.constant 13 : index
    %res, %new_true_shape = ave.hir.plt %c13 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<13xi8, #hivm.address_space<ub>> into vector<64xi8>
    %1 = ave.hir.vextui %0, %res {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<64xi1>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: llvm.alloca
    // CHECK: call @_mlir_ciface_cast_uint32_t_to_int64_t
    %3 = ave.hir.vextui %1, %2 {part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi64>, vector<64xi1>
    ave.hir.masked_store <NORM_B64> %arg1[%c0], %res, %3 : memref<13xi64, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    return
  }
  func.func private @load_gm_to_ubuf_1d_int8_t(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @triton_cast_1d(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<13xi8, strided<[1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<13xi8, #hivm.address_space<ub>>
    %cast = memref.cast %reinterpret_cast : memref<13xi8, strided<[1]>, #hivm.address_space<gm>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_0 = memref.cast %0 : memref<13xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_int8_t(%cast, %cast_0, %c0_i32, %c0_i8, %c0) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) -> ()
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %1 = hivm.hir.pointer_cast(%c32_i64) : memref<13xi64, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @triton_cast_1d_outlined_vf_0(%0, %1) {hivm.vector_function} : (memref<13xi8, #hivm.address_space<ub>>, memref<13xi64, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [13], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<13xi64, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %cast_2 = memref.cast %1 : memref<13xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_3 = memref.cast %reinterpret_cast_1 : memref<13xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
    call @store_ubuf_to_gm_1d_int64_t(%cast_2, %cast_3, %c0_i32) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}


// -----


// CHECK-LABEL: func.func @triton_cast1d_float_to_uint64_t
// CHECK: llvm.alloca
// CHECK: func.call @_mlir_ciface_cast_float_to_uint64_t
#map = affine_map<(d0) -> (-d0 + 300, 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_cast1d_float_to_uint64_t(%arg0: memref<300xf32, #hivm.address_space<ub>>, %arg1: memref<300xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c64 = arith.constant 64 : index
    %c300 = arith.constant 300 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c300 step %c64 {
      %0 = affine.min #map(%arg2)
      %subview = memref.subview %arg0[%arg2] [%0] [1] : memref<300xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2] [%0] [1] : memref<300xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %1 = ave.hir.pltm %arg2, %c300 : vector<64xi1>
      %2 = ave.hir.vload <NORM> %subview[%c0] : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %3 = ave.hir.vfptoui %2, <trunc>, false, <part_even>, %1 : vector<64xf32>, vector<64xi1>, vector<64xi64>
      ave.hir.masked_store <NORM_B64> %subview_0[%c0], %1, %3 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}

// -----

// CHECK-LABEL: func.func @test_mask_all
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_mask_all(%arg0: memref<105x3xi64, #hivm.address_space<ub>>, %arg1: memref<3x105xi64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3_i64 = arith.constant 3 : i64
    %c41 = arith.constant 41 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = ave.hir.vci %c0_i64, <INCREASE> : i64, vector<64xi64>
    %1 = ave.hir.pge <ALL> : vector<64xi1>
    %2 = ave.hir.vmuls %0, %c3_i64, %1 : vector<64xi64>, i64, vector<64xi1>
    %3 = ave.hir.pge <ALL> : vector<64xi1>
    %4 = ave.hir.broadcast %c0_i64, %3 : i64, vector<64xi1> -> vector<64xi64>
    %res, %new_true_shape = ave.hir.plt %c64 : vector<64xi1>, index
    %5 = ave.hir.vsel %res, %2, %4 : vector<64xi1>, vector<64xi64>
    %6 = ave.hir.vci %c0_i64, <INCREASE> : i64, vector<64xi64>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vmuls %6, %c3_i64, %7 : vector<64xi64>, i64, vector<64xi1>
    scf.for %arg2 = %c0 to %c3 step %c1 {
      %subview = memref.subview %arg0[0, %arg2] [64, 1] [1, 1] : memref<105x3xi64, #hivm.address_space<ub>> to memref<64x1xi64, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, 0] [1, 64] [1, 1] : memref<3x105xi64, #hivm.address_space<ub>> to memref<1x64xi64, strided<[105, 1], offset: ?>, #hivm.address_space<ub>>
      %9 = ave.hir.pge <ALL> : vector<64xi1>
      %10 = ave.hir.vgather %subview[%c0, %c0] [%8], %9 : memref<64x1xi64, strided<[3, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi64>, vector<64xi1> into vector<64xi64>
      %subview_1 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xi64, strided<[105, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xi64, #map, #hivm.address_space<ub>>
      %11 = ave.hir.pge <ALL> : vector<64xi1>
      // CHECK: %[[ARG1:.*]] = arith.constant 64 : i32
      // CHECK: func.call @masked_store_NORM_B64_int64_t_unalign_rank1(%{{.*}}, %{{.*}}, %{{.*}}, %[[ARG1:.*]])
      ave.hir.masked_store <NORM_B64> %subview_1[%c0], %11, %10 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<64xi64, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi64>
    } {element_alignment_bit_width = 32 : i32}
    return
  }
}

// -----

// CHECK-LABEL: func.func @vdiv_i16
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
  func.func @vdiv_i16(%arg0: memref<192xi16, #hivm.address_space<ub>>, %arg1: memref<192xi16, #hivm.address_space<ub>>, %arg2: memref<192xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c192 = arith.constant 192 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c192 step %c64 {
      %subview = memref.subview %arg0[%arg3] [64] [1] : memref<192xi16, #hivm.address_space<ub>> to memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3] [64] [1] : memref<192xi16, #hivm.address_space<ub>> to memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg3] [64] [1] : memref<192xi16, #hivm.address_space<ub>> to memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = ave.hir.vload <NORM> %subview[%c0] : memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
      %1 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi16>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      %3 = ave.hir.vdiv %0, %1, %2 : vector<64xi16>, vector<64xi1>
      // CHECK: func.call @_mlir_ciface_vdiv_int16_t
      %4 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B64> %subview_1[%c0], %4, %3 : memref<64xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @vdiv_i32
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>}{
  func.func @vdiv_i32(%arg0: memref<192xi32, #hivm.address_space<ub>>, %arg1: memref<192xi32, #hivm.address_space<ub>>, %arg2: memref<192xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c192 = arith.constant 192 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c192 step %c64 {
      %subview = memref.subview %arg0[%arg3] [64] [1] : memref<192xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3] [64] [1] : memref<192xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg3] [64] [1] : memref<192xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = ave.hir.vload <NORM> %subview[%c0] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
      %1 = ave.hir.vload <NORM> %subview_0[%c0] : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
      %2 = ave.hir.pge <ALL> : vector<64xi1>
      %3 = ave.hir.vdiv %0, %1, %2 : vector<64xi32>, vector<64xi1>
      // CHECK: func.call @_mlir_ciface_vdiv_int32_t
      %4 = ave.hir.pge <ALL> : vector<64xi1>
      ave.hir.masked_store <NORM_B64> %subview_1[%c0], %4, %3 : memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @triton_divfhp
func.func @triton_divfhp(%arg0: memref<1600xf32, #hivm.address_space<ub>>, %arg1: memref<1600xf32, #hivm.address_space<ub>>, %arg2: memref<1600xf32, #hivm.address_space<ub>>, %arg3: memref<1600xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1600 = arith.constant 1600 : index
  %c64 = arith.constant 64 : index
  scf.for %arg4 = %c0 to %c1600 step %c64 {
    %subview = memref.subview %arg0[%arg4] [64] [1] : memref<1600xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg4] [64] [1] : memref<1600xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg2[%arg4] [64] [1] : memref<1600xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res_3 = ave.hir.vload <NORM> %subview_1[%c0] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %10 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: func.call @_mlir_ciface_vdivfhp_float
    %11 = ave.hir.vdivfhp %res_3, %res, %10 : vector<64xf32>, vector<64xi1>
    %34 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %subview_2[%c0], %34, %11 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  } {element_alignment_bit_width = 32 : i32}
  return
}

// -----

// CHECK-LABEL: func.func @triton_mod_i16
func.func @triton_mod_i16(%arg0: memref<256xi16, #hivm.address_space<ub>>, %arg1: memref<256xi16, #hivm.address_space<ub>>, %arg2: memref<256xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 16 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c256 step %c128 {
    %subview = memref.subview %arg0[%arg3] [128] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [128] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [128] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
    %res_2 = ave.hir.vload <NORM> %subview_0[%c0] : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
    %0 = ave.hir.pge <ALL> : vector<128xi1>
    // CHECK: func.call @_mlir_ciface_vmod_int16_t
    %1 = ave.hir.vmod %res, %res_2, %0 : vector<128xi16>, vector<128xi1>
    %2 = ave.hir.pge <ALL> : vector<128xi1>
    ave.hir.masked_store <NORM_B16> %subview_1[%c0], %2, %1 : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  } {element_alignment_bit_width = 16 : i32}
  return
}

// -----

// CHECK-LABEL: func.func @triton_mod_u16
func.func @triton_mod_u16(%arg0: memref<256xi16, #hivm.address_space<ub>>, %arg1: memref<256xi16, #hivm.address_space<ub>>, %arg2: memref<256xi16, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 16 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c256 step %c128 {
    %subview = memref.subview %arg0[%arg3] [128] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [128] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [128] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
    %res_2 = ave.hir.vload <NORM> %subview_0[%c0] : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xi16>
    %0 = ave.hir.pge <ALL> : vector<128xi1>
    // CHECK: func.call @_mlir_ciface_vmodui_int16_t
    %1 = ave.hir.vmodui %res, %res_2, %0 : vector<128xi16>, vector<128xi1>
    %2 = ave.hir.pge <ALL> : vector<128xi1>
    ave.hir.masked_store <NORM_B16> %subview_1[%c0], %2, %1 : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
  } {element_alignment_bit_width = 16 : i32}
  return
}

// -----

// CHECK-LABEL: func.func @reduction_max_to_vcmax_vmax_int64
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @reduction_max_to_vcmax_vmax_int64(%arg0: memref<3xi64, #hivm.address_space<ub>>, %arg1: memref<i64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %0 = ave.hir.pge <VL3> : vector<64xi1>
    %res = ave.hir.vload <NORM> %arg0[%c0] : memref<3xi64, #hivm.address_space<ub>> into vector<64xi64>
    %res_0 = ave.hir.vload <BRC_B64> %arg1[] : memref<i64, #hivm.address_space<ub>> into vector<1xi64>
    %1 = builtin.unrealized_conversion_cast %res_0 : vector<1xi64> to vector<i64>
    %2 = builtin.unrealized_conversion_cast %1 : vector<i64> to i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to vector<1xi64>
    %4 = builtin.unrealized_conversion_cast %3 : vector<1xi64> to vector<64xi64>
    %5 = ave.hir.reduction <max>, %res, %0 : vector<64xi64>, vector<64xi1> -> vector<64xi64>
    // CHECK-NOT: ave.hir.reduction
    // CHECK: call @_mlir_ciface_vcmax_int64_t
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    %7 = ave.hir.vsmax %4, %5, %6 : vector<64xi64>, vector<64xi1>
    // CHECK-NOT: ave.hir.vsmax
    // CHECK: call @_mlir_ciface_vmax_int64_t
    %8 = builtin.unrealized_conversion_cast %7 : vector<64xi64> to vector<1xi64>
    %9 = builtin.unrealized_conversion_cast %8 : vector<1xi64> to i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to vector<i64>
    %11 = builtin.unrealized_conversion_cast %10 : vector<i64> to vector<1xi64>
    %12 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B64> %arg1[], %12, %11 : memref<i64, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi64>
    return
  }
}

// -----

// CHECK-LABEL: func.func @reduction_umax_to_vcmax_vmax_uint64
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @reduction_umax_to_vcmax_vmax_uint64(%arg0: memref<3xi64, #hivm.address_space<ub>>, %arg1: memref<i64, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %0 = ave.hir.pge <VL3> : vector<64xi1>
    %res = ave.hir.vload <NORM> %arg0[%c0] : memref<3xi64, #hivm.address_space<ub>> into vector<64xi64>
    %res_0 = ave.hir.vload <BRC_B64> %arg1[] : memref<i64, #hivm.address_space<ub>> into vector<1xi64>
    %1 = builtin.unrealized_conversion_cast %res_0 : vector<1xi64> to vector<i64>
    %2 = builtin.unrealized_conversion_cast %1 : vector<i64> to i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to vector<1xi64>
    %4 = builtin.unrealized_conversion_cast %3 : vector<1xi64> to vector<64xi64>
    %5 = ave.hir.reduction <umax>, %res, %0 : vector<64xi64>, vector<64xi1> -> vector<64xi64>
    // CHECK-NOT: ave.hir.reduction
    // CHECK: call @_mlir_ciface_vcmax_uint64_t
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    %7 = ave.hir.vumax %4, %5, %6 : vector<64xi64>, vector<64xi1>
    // CHECK-NOT: ave.hir.vumax
    // CHECK: call @_mlir_ciface_vmax_uint64_t
    %8 = builtin.unrealized_conversion_cast %7 : vector<64xi64> to vector<1xi64>
    %9 = builtin.unrealized_conversion_cast %8 : vector<1xi64> to i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to vector<i64>
    %11 = builtin.unrealized_conversion_cast %10 : vector<i64> to vector<1xi64>
    %12 = ave.hir.pge <ALL> : vector<1xi1>
    ave.hir.masked_store <ONEPT_B64> %arg1[], %12, %11 : memref<i64, #hivm.address_space<ub>>, vector<1xi1>, vector<1xi64>
    return
  }
}
