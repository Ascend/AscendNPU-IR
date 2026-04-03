// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory --debug-only="vf-inplace-reuse" -split-input-file -verify-diagnostics 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=disable-tightly-coupled-buffer-reuse=true -split-input-file -verify-diagnostics 2>&1 | FileCheck %s -check-prefix=CHECK-NOREUSE
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
// CHECK-DAG: hivm.hir.pointer_cast(%[[CONST0]])
func.func @plan_memory_vf_read_once_and_write_once_0() {
  %alloc = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  call @read_once_and_write_once_0(%alloc, %alloc_0, %alloc_1) {hivm.vector_function} :
    (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
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
  func.func @test_tightly_coupled_buffer_in_AIC(%arg0: memref<16x16xf16, #hivm.address_space<gm>>,
                               %arg1: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK: %[[ARG1:.*]] = hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<16x16xf32, #hivm.address_space<cbuf>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%arg1 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16x16xf32, #hivm.address_space<cbuf>>)
    // CHECK: %[[ARG2:.*]] = hivm.hir.pointer_cast(%[[CONST1:.*]]) : memref<16x16xf16, #hivm.address_space<cbuf>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: %[[ARG3:.*]] = hivm.hir.pointer_cast(%[[CONST2:.*]]) : memref<16x16xf32, #hivm.address_space<cc>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cc>>
    hivm.hir.load ins(%arg1 : memref<16x16xf32, #hivm.address_space<gm>>) outs(%alloc_3 : memref<16x16xf32, #hivm.address_space<cc>>)
    // CHECK: %[[ARG4:.*]] = hivm.hir.pointer_cast(%[[CONST3:.*]]) : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {enable_nz2nd, l0c_to_ub} ins(%alloc_3 : memref<16x16xf32, #hivm.address_space<cc>>) outs(%alloc_4 : memref<16x16xf32, #hivm.address_space<ub>>)
    return
  }
  func.func @test_tightly_coupled_buffer_in_AIV(%arg0: memref<16x16xf16, #hivm.address_space<gm>>,
                               %arg1: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}  {
    // CHECK: %[[ARG5:.*]] = hivm.hir.pointer_cast(%[[CONST4:.*]]) : memref<16x16xf16, #hivm.address_space<ub>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16x16xf16, #hivm.address_space<ub>>)
    // CHECK: %[[ARG6:.*]] = hivm.hir.pointer_cast(%[[CONST1]]) : memref<16x16xf16, #hivm.address_space<cbuf>>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x16xf16, #hivm.address_space<cbuf>>
    hivm.hir.copy ins(%alloc : memref<16x16xf16, #hivm.address_space<ub>>) outs(%alloc_2 : memref<16x16xf16, #hivm.address_space<cbuf>>) {ub_to_l1}
    // CHECK: %[[ARG7:.*]] = hivm.hir.pointer_cast(%[[CONST3]]) : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf32, #hivm.address_space<ub>>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_950z">} {
  func.func @test_tightly_coupled_buffer_in_AIC(%arg0: memref<16x16xf16, #hivm.address_space<gm>>,
                               %arg1: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cc>>
    hivm.hir.load ins(%arg1 : memref<16x16xf16, #hivm.address_space<gm>>) outs(%alloc_3 : memref<16x16xf16, #hivm.address_space<cc>>)
    // CHECK-NOREUSE: {{.*}} = hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<16x16xf16, #hivm.address_space<ub>>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<ub>>
    annotation.mark %alloc_4 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf16, #hivm.address_space<ub>>
    hivm.hir.fixpipe {enable_nz2nd, l0c_to_ub} ins(%alloc_3 : memref<16x16xf16, #hivm.address_space<cc>>) outs(%alloc_4 : memref<16x16xf16, #hivm.address_space<ub>>)
    return
  }
  func.func @test_unique_memory_for_CV_tightly_coupled_buffer(%arg0: i32, %arg1: memref<16x16xf16, #hivm.address_space<gm>>,
                               %arg2: memref<16x16xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}  {
    // CHECK-NOREUSE: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]]) : memref<16x16xf16, #hivm.address_space<ub>>
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<16x16xf16, #hivm.address_space<ub>>
    %0 = arith.cmpi eq, %arg0, %c1_i32 : i32
    // CHECK-NOREUSE: {{.*}} = scf.if {{.*}} -> (memref<16x16xf16, #hivm.address_space<ub>>) {
    %1 = scf.if %0 -> (memref<16x16xf16, #hivm.address_space<ub>>) {
      // CHECK-NOREUSE-NOT: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]]) : memref<16x16xf16, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16x16xf16, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16x16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc_1 : memref<16x16xf16, #hivm.address_space<ub>>
    } else {
      scf.yield %alloc : memref<16x16xf16, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%1 : memref<16x16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_unique_memory_if_reuse(%arg0: i32, %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>, %arg2: memref<16x16x16xf16, #hivm.address_space<gm>>, %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>) {
    %c1_i32 = arith.constant 1 : i32
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {mem_unique} : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST1:.*]]) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    %0 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %1 = scf.if %0 -> (memref<16x16x16xf16, #hivm.address_space<ub>>) {
      // CHECK-NOT: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]]) : memref<16x16x16xf16, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16x16x16xf16, #hivm.address_space<ub>>, memref<16x16x16xf16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc_1 : memref<16x16x16xf16, #hivm.address_space<ub>>
    } else {
      scf.yield %alloc : memref<16x16x16xf16, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%1 : memref<16x16x16xf16, #hivm.address_space<ub>>) outs(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

// expected-error@below {{ub overflow, requires 2097152 bits while 2031616 bits available!}}
func.func @test_unique_memory_time_reuse(%arg0: memref<16x32x128xf16, #hivm.address_space<gm>>, %arg1: memref<16x32x128xf16, #hivm.address_space<gm>>) {
  %alloc = memref.alloc() : memref<16x32x128xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {mem_unique} : memref<16x32x128xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<16x32x128xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16x32x128xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%alloc : memref<16x32x128xf16, #hivm.address_space<ub>>) outs(%arg0 : memref<16x32x128xf16, #hivm.address_space<gm>>)
  %alloc_0 = memref.alloc() : memref<16x32x128xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg1 : memref<16x32x128xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16x32x128xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%alloc_0 : memref<16x32x128xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16x32x128xf16, #hivm.address_space<gm>>)
  return
}

// -----

func.func @test_unique_memory_with_multi_buffer(%arg0: memref<16x32x64xf16, #hivm.address_space<gm>>, %arg1: memref<16x32x64xf16, #hivm.address_space<gm>>) {
  // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0:.*]], %[[CONST1:.*]]) : memref<16x32x64xf16, #hivm.address_space<ub>>
  %alloc = memref.alloc() : memref<16x32x64xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {mem_unique, hivm.multi_buffer = 2 : i32} : memref<16x32x64xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<16x32x64xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16x32x64xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%alloc : memref<16x32x64xf16, #hivm.address_space<ub>>) outs(%arg0 : memref<16x32x64xf16, #hivm.address_space<gm>>)
  // CHECK-DAG_NOT: {{.*}} = hivm.hir.pointer_cast(%[[CONST1]]) : memref<16x32x64xf16, #hivm.address_space<ub>>
  // CHECK-DAG_NOT: {{.*}} = hivm.hir.pointer_cast(%[[CONST2]]) : memref<16x32x64xf16, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<16x32x64xf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg1 : memref<16x32x64xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16x32x64xf16, #hivm.address_space<ub>>)
  hivm.hir.store ins(%alloc_0 : memref<16x32x64xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16x32x64xf16, #hivm.address_space<gm>>)
  return
}

// -----

// expected-error@below {{ub overflow, requires 1966080 bits while 1769472 bits available!}}
func.func @invalid_alloc_for_mix(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>} {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<61440xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<61440xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<61440xf32, #hivm.address_space<gm>>) outs(%alloc : memref<61440xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg1 : memref<61440xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<61440xf32, #hivm.address_space<ub>>)
  %0 = memref.load %alloc[%c0] : memref<61440xf32, #hivm.address_space<ub>>
  %1 = memref.load %alloc_0[%c0] : memref<61440xf32, #hivm.address_space<ub>>
  %2 = arith.mulf %0, %1 : f32
  return
}

// -----

// CHECK-NOT: memref.alloc()
func.func @valid_alloc_for_simd(%arg0: memref<61440xf32, #hivm.address_space<gm>>, %arg1: memref<61440xf32, #hivm.address_space<gm>>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<61440xf32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<61440xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<61440xf32, #hivm.address_space<gm>>) outs(%alloc : memref<61440xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg1 : memref<61440xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<61440xf32, #hivm.address_space<ub>>)
  %0 = memref.load %alloc[%c0] : memref<61440xf32, #hivm.address_space<ub>>
  %1 = memref.load %alloc_0[%c0] : memref<61440xf32, #hivm.address_space<ub>>
  %2 = arith.mulf %0, %1 : f32
  return
}

// -----

// CHECK-LABEL: func.func @vfreuse_store_subview_collapseshape
// CHECK-DAG: %[[C41088:.*]] = arith.constant 41088 : i64
// CHECK-DAG: %[[C40960:.*]] = arith.constant 40960 : i64
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[C0]]) : memref<64x320xbf16, #hivm.address_space<ub>>
// CHECK-DAG: hivm.hir.pointer_cast(%[[C40960]]) : memref<64xbf16, #hivm.address_space<ub>>
// CHECK-DAG: hivm.hir.pointer_cast(%[[C41088]]) : memref<64x320xbf16, #hivm.address_space<ub>>
func.func @vfreuse_store_subview_collapseshape_outlined_vf_2(%arg0: memref<64xbf16, #hivm.address_space<ub>>, %arg1: memref<64x320xbf16, #hivm.address_space<ub>>, %arg2: memref<64x320xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<1.000000e+00> : vector<1x64xf32>
  %cst_0 = arith.constant dense<-1.000000e+00> : vector<1x64xf32>
  %cst_1 = arith.constant 0.000000e+00 : bf16
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c320 = arith.constant 320 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
  %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xbf16, #hivm.address_space<ub>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg4 = %c0 to %c320 step %c64 {
      %subview_2 = memref.subview %arg1[%arg3, %arg4] [1, 64] [1, 1] : memref<64x320xbf16, #hivm.address_space<ub>> to memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %arg2[%arg3, %arg4] [1, 64] [1, 1] : memref<64x320xbf16, #hivm.address_space<ub>> to memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xbf16>
      %1 = vector.transfer_read %subview[%c0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xbf16>
      %2 = arith.extf %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xbf16> to vector<1x64xf32>
      %3 = arith.mulf %2, %cst_0 : vector<1x64xf32>
      %4 = math.exp %3 : vector<1x64xf32>
      %5 = arith.addf %4, %cst : vector<1x64xf32>
      %6 = arith.divf %cst, %5 : vector<1x64xf32>
      %7 = arith.extf %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xbf16> to vector<1x64xf32>
      %8 = arith.mulf %7, %6 : vector<1x64xf32>
      %9 = arith.truncf %8 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xbf16>
      vector.transfer_write %9, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xbf16>, memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}
func.func @vfreuse_store_subview_collapseshape(%arg2: memref<?xbf16, #hivm.address_space<gm>>, %18: index) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c59_i32 = arith.constant 59 : i32
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  scf.for %arg9 = %c0_i32 to %c59_i32 step %c1_i32  : i32 {
    %14 = arith.index_cast %arg9 : i32 to index
    %alloc = memref.alloc() : memref<64x320xbf16, #hivm.address_space<ub>>
    %alloc_3 = memref.alloc() : memref<64xbf16, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64x320xbf16, #hivm.address_space<ub>>
    %subview_0 = memref.subview %alloc[0, 0] [%18, 320] [1, 1] : memref<64x320xbf16, #hivm.address_space<ub>> to memref<?x320xbf16, strided<[320, 1]>, #hivm.address_space<ub>>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1]] : memref<?x320xbf16, strided<[320, 1]>, #hivm.address_space<ub>> into memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 320], strides: [320, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x320xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%18, 320] [1, 1] : memref<64x320xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x320xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<?x320xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%collapse_shape : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_1 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index eviction_policy = <EvictFirst>
    func.call @vfreuse_store_subview_collapseshape_outlined_vf_2(%alloc_3, %alloc, %alloc_6) {hivm.vector_function, no_inline} : (memref<64xbf16, #hivm.address_space<ub>>, memref<64x320xbf16, #hivm.address_space<ub>>, memref<64x320xbf16, #hivm.address_space<ub>>) -> ()
    %subview_7 = memref.subview %alloc_6[0, 0] [%18, 320] [1, 1] : memref<64x320xbf16, #hivm.address_space<ub>> to memref<?x320xbf16, strided<[320, 1]>, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1]] : memref<?x320xbf16, strided<[320, 1]>, #hivm.address_space<ub>> into memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.store ins(%collapse_shape_8 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  }
  return
}

// -----

// VF: read arg0, arg1; write arg2. Arg2 can inplace-reuse arg0 or arg1 from callee perspective.
func.func @no_inplace_reuse_vf_args_same_alloc_vf(
    %arg0: memref<64xbf16, #hivm.address_space<ub>>,
    %arg1: memref<64x320xbf16, #hivm.address_space<ub>>,
    %arg2: memref<64x320xbf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst_1 = arith.constant 0.000000e+00 : bf16
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c320 = arith.constant 320 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xbf16, #hivm.address_space<ub>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg4 = %c0 to %c320 step %c64 {
      %subview_2 = memref.subview %arg1[%arg3, %arg4] [1, 64] [1, 1] : memref<64x320xbf16, #hivm.address_space<ub>> to memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %arg2[%arg3, %arg4] [1, 64] [1, 1] : memref<64x320xbf16, #hivm.address_space<ub>> to memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xbf16>
      %1 = vector.transfer_read %subview[%c0], %cst_1 {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xbf16>
      %2 = arith.extf %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xbf16> to vector<1x64xf32>
      %3 = math.exp %2 : vector<1x64xf32>
      %4 = arith.extf %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xbf16> to vector<1x64xf32>
      %5 = arith.mulf %4, %3 : vector<1x64xf32>
      %6 = arith.truncf %5 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xbf16>
      vector.transfer_write %6, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xbf16>, memref<1x64xbf16, strided<[320, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// Test: no inplace reuse when arg1 and arg2 are the same alloc.
// vf.call(%a, %b, %b) - reusing arg2 with arg0 would make %a,%b share memory while %b is
// also used as arg1. Skip to avoid unsound VF optimizations.
// CHECK-LABEL: func.func @plan_memory_no_inplace_reuse_vf_args_same_alloc
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C1:.*]] = arith.constant 128 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[C0]]) : memref<64xbf16, #hivm.address_space<ub>>
// CHECK-DAG: hivm.hir.pointer_cast(%[[C1]]) : memref<64x320xbf16, #hivm.address_space<ub>>
func.func @plan_memory_no_inplace_reuse_vf_args_same_alloc() {
  %alloc_3 = memref.alloc() : memref<64xbf16, #hivm.address_space<ub>>
  %alloc = memref.alloc() : memref<64x320xbf16, #hivm.address_space<ub>>
  call @no_inplace_reuse_vf_args_same_alloc_vf(%alloc_3, %alloc, %alloc) {hivm.vector_function} :
    (memref<64xbf16, #hivm.address_space<ub>>, memref<64x320xbf16, #hivm.address_space<ub>>, memref<64x320xbf16, #hivm.address_space<ub>>) -> ()
  return
}

// Test: no inplace reuse when arg1 and arg2 trace to same alloc via reshape.
// %c = collapse_shape(expand_shape %alloc_b) - %b and %c share alloc.
// vf.call(%a, %b, %c): reusing arg2 with arg0 would make %a,%b,%c share memory. Skip.
// CHECK-LABEL: func.func @plan_memory_no_inplace_reuse_vf_args_reshape_share_alloc
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C1:.*]] = arith.constant 128 : i64
// CHECK-DAG: hivm.hir.pointer_cast(%[[C0]]) : memref<64xbf16, #hivm.address_space<ub>>
// CHECK-DAG: hivm.hir.pointer_cast(%[[C1]]) : memref<64x320xbf16, #hivm.address_space<ub>>
func.func @plan_memory_no_inplace_reuse_vf_args_reshape_share_alloc() {
  %alloc_3 = memref.alloc() : memref<64xbf16, #hivm.address_space<ub>>
  %alloc_b = memref.alloc() : memref<64x320xbf16, #hivm.address_space<ub>>
  %expanded = memref.expand_shape %alloc_b [[0], [1, 2]] output_shape [64, 320, 1] : memref<64x320xbf16, #hivm.address_space<ub>> into memref<64x320x1xbf16, #hivm.address_space<ub>>
  %collapsed = memref.collapse_shape %expanded [[0], [1, 2]] : memref<64x320x1xbf16, #hivm.address_space<ub>> into memref<64x320xbf16, #hivm.address_space<ub>>
  call @no_inplace_reuse_vf_args_same_alloc_vf(%alloc_3, %alloc_b, %collapsed) {hivm.vector_function} :
    (memref<64xbf16, #hivm.address_space<ub>>, memref<64x320xbf16, #hivm.address_space<ub>>, memref<64x320xbf16, #hivm.address_space<ub>>) -> ()
  return
}
