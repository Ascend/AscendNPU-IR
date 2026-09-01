// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -hivm-vf-operand-substitution -split-input-file | FileCheck %s

func.func @vf_reuse_one_output_two_sources(
    %arg0: memref<64xf32, #hivm.address_space<ub>>,
    %arg1: memref<64xf32, #hivm.address_space<ub>>,
    %arg2: memref<64xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} :
      memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} :
      memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = arith.addf %0, %1 : vector<64xf32>
  vector.transfer_write %2, %arg2[%c0] {in_bounds = [true]} :
      vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// Each call has two candidates for the same destination index. Only the first
// source may donate its buffer, and candidate-index tracking must be reset for
// the next call.
// CHECK-LABEL: func.func @deduplicate_overlapping_indices_per_call(
// CHECK: %[[SRC0:[a-zA-Z0-9_]+]] = arith.select
// CHECK: %[[OTHER0:[a-zA-Z0-9_]+]] = arith.select
// CHECK-NOT: memref.alloc()
// CHECK: call @vf_reuse_one_output_two_sources(%[[SRC0]], %[[OTHER0]], %[[SRC0]])
// CHECK: %[[SRC1:[a-zA-Z0-9_]+]] = arith.select
// CHECK: %[[OTHER1:[a-zA-Z0-9_]+]] = arith.select
// CHECK-NOT: memref.alloc()
// CHECK: call @vf_reuse_one_output_two_sources(%[[SRC1]], %[[OTHER1]], %[[SRC1]])
func.func @deduplicate_overlapping_indices_per_call(%cond: i1) {
  %left0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %right0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %src0 = arith.select %cond, %left0, %right0 :
      memref<64xf32, #hivm.address_space<ub>>
  %left1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %right1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %other0 = arith.select %cond, %left1, %right1 :
      memref<64xf32, #hivm.address_space<ub>>
  %dst0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_one_output_two_sources(%src0, %other0, %dst0)
      {hivm.vector_function, no_inline} :
      (memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>) -> ()

  %left2 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %right2 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %src1 = arith.select %cond, %left2, %right2 :
      memref<64xf32, #hivm.address_space<ub>>
  %left3 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %right3 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %other1 = arith.select %cond, %left3, %right3 :
      memref<64xf32, #hivm.address_space<ub>>
  %dst1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_one_output_two_sources(%src1, %other1, %dst1)
      {hivm.vector_function, no_inline} :
      (memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>) -> ()
  return
}

// -----

func.func @vf_reuse_one_source_two_outputs(
    %arg0: memref<64xf32, #hivm.address_space<ub>>,
    %arg1: memref<64xf32, #hivm.address_space<ub>>,
    %arg2: memref<64xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} :
      memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  vector.transfer_write %0, %arg1[%c0] {in_bounds = [true]} :
      vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  vector.transfer_write %0, %arg2[%c0] {in_bounds = [true]} :
      vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

// A source buffer can be donated to only one output. Keep one of the two
// distinct destination allocs instead of substituting both with the source.
// CHECK-LABEL: func.func @deduplicate_source_index(
// CHECK: arith.select
// CHECK: memref.alloc()
// CHECK-NOT: memref.alloc()
// CHECK: call @vf_reuse_one_source_two_outputs
func.func @deduplicate_source_index(%cond: i1) {
  %left = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %right = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %src = arith.select %cond, %left, %right :
      memref<64xf32, #hivm.address_space<ub>>
  %dst0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %dst1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_one_source_two_outputs(%src, %dst0, %dst1)
      {hivm.vector_function, no_inline} :
      (memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>) -> ()
  return
}

// The same destination alloc may occur at multiple operand indices. Replacing
// the first candidate rewrites every use of the alloc and erases it; later
// candidates must not access that erased operation.
// CHECK-LABEL: func.func @deduplicate_destination_alloc(
// CHECK: %[[SRC:[a-zA-Z0-9_]+]] = arith.select
// CHECK-NOT: memref.alloc()
// CHECK: call @vf_reuse_one_source_two_outputs(%[[SRC]], %[[SRC]], %[[SRC]])
func.func @deduplicate_destination_alloc(%cond: i1) {
  %left = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %right = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %src = arith.select %cond, %left, %right :
      memref<64xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  func.call @vf_reuse_one_source_two_outputs(%src, %dst, %dst)
      {hivm.vector_function, no_inline} :
      (memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>,
       memref<64xf32, #hivm.address_space<ub>>) -> ()
  return
}
