// REQUIRES: asserts
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory --debug-only="vf-inplace-reuse" -split-input-file -verify-diagnostics 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=disable-vf-reachable-check=true -split-input-file -verify-diagnostics | FileCheck %s -check-prefix=CHECK-NO-REACHABLE-CHECK

// -----

func.func @read_once_and_write_once_0(
    %arg0: memref<64xf32, #hivm.address_space<ub>>,
    %arg1: memref<64xf32, #hivm.address_space<ub>>,
    %arg2: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.693147182> : vector<64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %1 = vector.transfer_read %arg1[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = math.log %1 : vector<64xf32>
  %3 = arith.divf %2, %cst : vector<64xf32>
  %4 = arith.addf %0, %3 : vector<64xf32>
  vector.transfer_write %4, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_read_once_and_write_once_0
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[CONST1:.*]] = arith.constant 256 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
func.func @plan_memory_vf_read_once_and_write_once_0() {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  call @read_once_and_write_once_0(%alloc, %alloc_0, %alloc_1) {hivm.vector_function} :
    (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func @plan_memory_vf_best_inplace_pair_outlined_vf_1(
    %arg0: memref<31744xf32, #hivm.address_space<ub>>,
    %arg1: memref<31744xf8E4M3FN, #hivm.address_space<ub>>,
    %arg2: memref<31744xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c31744 = arith.constant 31744 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c31744 step %c64 {
    %subview = memref.subview %arg0[%arg3] [64] [1] : memref<31744xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg2[%arg3] [64] [1] : memref<31744xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
    %1 = math.log %0 : vector<64xf32>
    vector.transfer_write %1, %subview_0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg3] [64] [1] : memref<31744xf8E4M3FN, #hivm.address_space<ub>> to memref<64xf8E4M3FN, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = arith.truncf %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<si2si>} : vector<64xf32> to vector<64xf8E4M3FN>
    vector.transfer_write %2, %subview_1[%c0] {in_bounds = [true]} : vector<64xf8E4M3FN>, memref<64xf8E4M3FN, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_best_inplace_pair(
// CHECK-DAG: %[[F32_OFFSET:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[F8_OFFSET:.*]] = arith.constant 126976 : i64
// CHECK: hivm.hir.pointer_cast(%[[F32_OFFSET]]) : memref<31744xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.pointer_cast(%[[F8_OFFSET]]) : memref<31744xf8E4M3FN, #hivm.address_space<ub>>
// CHECK: hivm.hir.pointer_cast(%[[F32_OFFSET]]) : memref<31744xf32, #hivm.address_space<ub>>
func.func @plan_memory_vf_best_inplace_pair(
    %arg3: memref<?xf8E4M3FN, #hivm.address_space<gm>>) 
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<31744xf32, #hivm.address_space<ub>>
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " x0: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc_0 : memref<31744xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<31744xf8E4M3FN, #hivm.address_space<ub>>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<31744xf32, #hivm.address_space<ub>>
  call @plan_memory_vf_best_inplace_pair_outlined_vf_1(%alloc_0, %alloc_1, %alloc_2) {hivm.vector_function, no_inline} : (memref<31744xf32, #hivm.address_space<ub>>, memref<31744xf8E4M3FN, #hivm.address_space<ub>>, memref<31744xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " ret: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc_2 : memref<31744xf32, #hivm.address_space<ub>>
  %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [31744], strides: [1] : memref<?xf8E4M3FN, #hivm.address_space<gm>> to memref<31744xf8E4M3FN, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.store ins(%alloc_1 : memref<31744xf8E4M3FN, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<31744xf8E4M3FN, strided<[1]>, #hivm.address_space<gm>>)
  return
}

// -----

func.func @read_once_and_write_once_subview_0(
  %arg0: memref<64xf32, #hivm.address_space<ub>>,
  %arg1: memref<64x64xf32, #hivm.address_space<ub>>,
  %arg2: memref<64x64xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %1 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %2 = arith.divf %0, %1 : vector<1x64xf32>
    %3 = arith.truncf %2 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
    vector.transfer_write %3, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_read_once_and_write_once_subview_0
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[CONST1:.*]] = arith.constant 256 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
func.func @plan_memory_vf_read_once_and_write_once_subview_0() {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<64x64xf16, #hivm.address_space<ub>>
  call @read_once_and_write_once_subview_0(%alloc, %alloc_0, %alloc_1) {hivm.vector_function} :
    (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func @read_once_and_write_once_widening_0(
  %arg0: memref<32xf16, #hivm.address_space<ub>>,
  %arg1: memref<8xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c8 step %c1 {
    %subview = memref.subview %arg0[%arg2] [1] [1] : memref<32xf16, #hivm.address_space<ub>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [1] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf16>
    %1 = arith.extf %0 : vector<1xf16> to vector<1xf32>
    vector.transfer_write %1, %subview_0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_no_inplace_reuse_widening_0
// CHECK-DAG: %[[CONST1:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]]) : memref<32xf16, #hivm.address_space<ub>>
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]]) : memref<8xf32, #hivm.address_space<ub>>
func.func @plan_memory_vf_no_inplace_reuse_widening_0() {
  %alloc = memref.alloc() : memref<32xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  call @read_once_and_write_once_widening_0(%alloc, %alloc_0) {hivm.vector_function} :
    (memref<32xf16, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func @write_only_once_0(%arg0: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg1 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg1, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    vector.transfer_write %cst, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_write_only_once_0
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
func.func @plan_memory_vf_write_only_once_0() {
  %alloc = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
  call @write_only_once_0(%alloc) {hivm.vector_function} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
  return
}

// -----

// CHECK-DEBUG-LABEL: func.func @plan_memory_vf_read_write_diff_blocks_0
// CHECK-DEBUG: %[[alloc:.*]] = memref.alloc()
// CHECK-DEBUG: %[[alloc0:.*]] = memref.alloc()
// CHECK-DEBUG: %[[alloc1:.*]] = memref.alloc()
// CHECK-DEBUG: %[[alloc2:.*]] = memref.alloc()
// CHECK-DEBUG: %[[alloc3:.*]] = memref.alloc()
// CHECK-DEBUG: vf call: func.call @read_write_diff_blocks_0(
// CHECK-DEBUG-SAME: %[[alloc]], %[[alloc0]], %[[alloc1]], %[[alloc2]], %[[alloc3]]
// CHECK-DEBUG: inplace reusable values for operand: %[[alloc3]]
// CHECK-DEBUG: --- %[[alloc]]
// CHECK-DEBUG: --- %[[alloc0]]
func.func @read_write_diff_blocks_0(
    %arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>,
    %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64x16xf32, #hivm.address_space<ub>>,
    %arg4: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.72134751> : vector<64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 16] : vector<1x64xi1>
  scf.for %arg5 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg3[%arg5, 0] [1, 16] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg5] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %6 = vector.transfer_read %subview[%c0, %c0], %cst_0, %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %7 = vector.transfer_read %subview_1[%c0], %cst_0 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
    %8 = vector.mask %0 { vector.multi_reduction <add>, %6, %7 [1] : vector<1x64xf32> to vector<1xf32> } : vector<1x64xi1> -> vector<1xf32>
    vector.transfer_write %8, %subview_1[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  %1 = vector.transfer_read %arg0[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = vector.transfer_read %arg2[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %4 = arith.mulf %1, %2 : vector<64xf32>
  %5 = arith.addf %4, %3 : vector<64xf32>
  vector.transfer_write %5, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_read_write_diff_blocks_0
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 768 : i64
// CHECK-DAG: %[[CONST1:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[CONST2:.*]] = arith.constant 256 : i64
// CHECK-DAG: %[[CONST3:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST3]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST2]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST3]])
func.func @plan_memory_vf_read_write_diff_blocks_0() {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_2 = memref.alloc() : memref<64x16xf32, #hivm.address_space<ub>>
  %alloc_3 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  call @read_write_diff_blocks_0(%alloc, %alloc_0, %alloc_1, %alloc_2, %alloc_3) {hivm.vector_function} :
       (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>, memref<64x16xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func @transpose_read_write_0(%arg0: memref<256x256xi8, #hivm.address_space<ub>>, %arg1: memref<256x256xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c256 step %c1 {
    scf.for %arg3 = %c0 to %c256 step %c128 {
      %subview = memref.subview %arg0[%arg3, %arg2] [128, 1] [1, 1] : memref<256x256xi8, #hivm.address_space<ub>> to memref<128x1xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [1, 128] [1, 1] : memref<256x256xi8, #hivm.address_space<ub>> to memref<1x128xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<128x1xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xi8>
      vector.transfer_write %0, %subview_0[%c0, %c0] {in_bounds = [true, true]} : vector<1x128xi8>, memref<1x128xi8, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// CHECK-LABEL: func.func @plan_memory_vf_not_inplace_reuse_transpose_read_write_0
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 65536 : i64
// CHECK-DAG: %[[CONST1:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
func.func @plan_memory_vf_not_inplace_reuse_transpose_read_write_0() {
  %alloc = memref.alloc() : memref<256x256xi8, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<256x256xi8, #hivm.address_space<ub>>
  call @transpose_read_write_0(%alloc, %alloc_0) {hivm.vector_function} :
       (memref<256x256xi8, #hivm.address_space<ub>>, memref<256x256xi8, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func @simple_reusable_vf(
    %arg0: memref<64xf32, #hivm.address_space<ub>>,
    %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.693147182> : vector<64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %1 = math.log %0 : vector<64xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// test inplace-reuse when gen/kill value reach store/load through subviews or inplace-reuse vf operands not in the loop block
// CHECK-LABEL: func.func @not_inplace_reuse_when_reach_load_and_store_0
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
func.func @not_inplace_reuse_when_reach_load_and_store_0(
  %arg0: memref<64xf32, #hivm.address_space<gm>>,
  %arg1: memref<64xf32, #hivm.address_space<gm>>
) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc_0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %subview_0 = memref.subview %alloc_0[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>)
                outs(%subview_0 : memref<64xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index
  %alloc_1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %alloc_1[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  call @simple_reusable_vf(%subview_0, %subview_1) {hivm.vector_function} :
       (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  %alloc_2 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %subview_2 = memref.subview %alloc_2[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  call @simple_reusable_vf(%subview_1, %subview_2) {hivm.vector_function} :
       (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.store ins(%subview_2 : memref<64xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<64xf32, #hivm.address_space<gm>>)
  return
}

// this vf is not reusable because %arg1 is read and write at the same time
func.func @simple_not_reusable_vf(
    %arg0: memref<64xf32, #hivm.address_space<ub>>,
    %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.693147182> : vector<64xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %1 = vector.transfer_read %arg1[%c0], %cst_0 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = arith.mulf %0, %1 : vector<64xf32>
  vector.transfer_write %2, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// test can inplace-reuse because kill value can not reach store directly, which is blocked by another vf function
// that is not inplace reusable
// CHECK-DAG: %[[CONST0:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[CONST1:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST1]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
func.func @can_inplace_reuse_when_not_reach_load_or_store_0(
  %arg0: memref<64xf32, #hivm.address_space<gm>>,
  %arg1: memref<64xf32, #hivm.address_space<gm>>
) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc_0 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %subview_0 = memref.subview %alloc_0[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>)
                outs(%subview_0 : memref<64xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index
  %alloc_1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %subview_1 = memref.subview %alloc_1[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  call @simple_reusable_vf(%subview_0, %subview_1) {hivm.vector_function} :
       (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  %alloc_2 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %subview_2 = memref.subview %alloc_2[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  call @simple_not_reusable_vf(%subview_1, %subview_2) {hivm.vector_function} :
       (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.store ins(%subview_2 : memref<64xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<64xf32, #hivm.address_space<gm>>)
  return
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_950z">} {
  func.func @vf_a(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  
  func.func @test_tightly_coupled_buffer_in_AIC(%arg0: memref<64xf32, #hivm.address_space<gm>>,
                              %arg1: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64xf32, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc_2 : memref<64xf32, #hivm.address_space<cbuf>>)
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %alloc_2: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %alloc_2 : memref<64xf32, #hivm.address_space<cbuf>>
    return
  }

  func.func @test_tightly_coupled_buffer_in_AIV(%arg0: memref<64xf32, #hivm.address_space<gm>>,
                              %arg1: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}  {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c64 step %c64 {
      // CHECK: hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<64xf32, #hivm.address_space<ub>>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64xf32, #hivm.address_space<ub>>)
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_a(%alloc, %alloc_1) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      // CHECK-NOT: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64xf32, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%alloc_1 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_2 : memref<64xf32, #hivm.address_space<cbuf>>) {ub_to_l1}
    }
    return
  }
}

// -----

// Two kill buffers cannot both be reused by gen; gen reuses kill_1, store gets a new slot.
// CHECK-LABEL: func.func @kernel_two_kill
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[C0]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[C256]])
// CHECK-DAG: hivm.hir.pointer_cast(%[[C0]])
// CHECK-DAG: hivm.hir.pointer_cast(%{{.*}}512
// CHECK-NOT: pointer_cast(%{{.*}}768

module {
  func.func @vf_a(%kill_1: memref<64xf32, #hivm.address_space<ub>>, %kill_2: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill_1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %kill_2[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = arith.mulf %0, %1 : vector<64xf32>
    vector.transfer_write %2, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @vf_b(%kill: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @kernel_two_kill(%gm: memref<64xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c64 {
      %kill_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      %kill_2 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%kill_1 : memref<64xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%kill_2 : memref<64xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
      %gen = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_a(%kill_1, %kill_2, %gen) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      %store = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_b(%gen, %store) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.store ins(%store : memref<64xf32, #hivm.address_space<ub>>) outs(%gm : memref<64xf32, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----

// Single kill can be reused into gen; store uses a separate slot.
// CHECK-LABEL: func.func @kernel_one_kill
// CHECK-DAG: hivm.hir.pointer_cast(%{{.*}}0
// CHECK-DAG: hivm.hir.pointer_cast(%{{.*}}0
// CHECK-DAG: hivm.hir.pointer_cast(%{{.*}}256

module {
  func.func @vf_a(%kill_1: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill_1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @vf_b(%kill: memref<64xf32, #hivm.address_space<ub>>, %gen: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }

  func.func @kernel_one_kill(%gm: memref<64xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c64 {
      %kill_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%kill_1 : memref<64xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
      %gen = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_a(%kill_1, %gen) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      %store = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      func.call @vf_b(%gen, %store) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.store ins(%store : memref<64xf32, #hivm.address_space<ub>>) outs(%gm : memref<64xf32, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----

module {
  func.func @vf_a(%kill_1: memref<30720xf32, #hivm.address_space<ub>>, %gen: memref<30720xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill_1[%c0], %cst {in_bounds = [true]} : memref<30720xf32, #hivm.address_space<ub>>, vector<30720xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<30720xf32>, memref<30720xf32, #hivm.address_space<ub>>
    return
  }

  func.func @vf_b(%kill: memref<30720xf32, #hivm.address_space<ub>>, %gen: memref<30720xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %kill[%c0], %cst {in_bounds = [true]} : memref<30720xf32, #hivm.address_space<ub>>, vector<30720xf32>
    vector.transfer_write %0, %gen[%c0] {in_bounds = [true]} : vector<30720xf32>, memref<30720xf32, #hivm.address_space<ub>>
    return
  }
  // CHECK-LABEL: func.func @test_no_reachable_check
  // CHECK-NO-REACHABLE-CHECK-LABEL: func.func @test_no_reachable_check
  func.func @test_no_reachable_check(%gm: memref<30720xf32, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    // CHECK-DAG: %{{.*}} = arith.constant 0 : i64
    // CHECK-DAG: %{{.*}} = arith.constant 122880 : i64
    // CHECK-NO-REACHABLE-CHECK-NOT: %{{.*}} = arith.constant 122880 : i64
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    scf.for %iv = %c0 to %c256 step %c64 {
      %kill_1 = memref.alloc() : memref<30720xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%gm : memref<30720xf32, #hivm.address_space<gm>>) outs(%kill_1 : memref<30720xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
      %gen = memref.alloc() : memref<30720xf32, #hivm.address_space<ub>>
      func.call @vf_a(%kill_1, %gen) {hivm.vector_function, no_inline} : (memref<30720xf32, #hivm.address_space<ub>>, memref<30720xf32, #hivm.address_space<ub>>) -> ()
      %store = memref.alloc() : memref<30720xf32, #hivm.address_space<ub>>
      func.call @vf_b(%gen, %store) {hivm.vector_function, no_inline} : (memref<30720xf32, #hivm.address_space<ub>>, memref<30720xf32, #hivm.address_space<ub>>) -> ()
      hivm.hir.store ins(%store : memref<30720xf32, #hivm.address_space<ub>>) outs(%gm : memref<30720xf32, #hivm.address_space<gm>>)
    }
    return
  }
}