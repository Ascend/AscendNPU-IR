// RUN: bishengir-opt %s -hivm-map-forall-to-blocks -split-input-file -allow-unregistered-dialect | FileCheck %s

// -----
func.func @trivial_test() {
  // CHECK: %[[BLOCK_X:.*]] = hivm.hir.get_block_idx -> i64
  // CHECK: %[[CAST_X:.*]] = arith.index_cast %[[BLOCK_X]] : i64 to index
  // CHECK: "some_use"(
  // CHECK-NOT: scf.forall (
  scf.forall (%arg0) in (100) {
    "some_use"(%arg0) : (index) -> ()
  } {mapping = [#hivm.block]}
  return
}

// -----
func.func @sub_block_test() {
  // CHECK: "some_use"({{.*}}, {{.*}})
  // CHECK-NOT: scf.forall (
  scf.forall (%arg0, %arg1) in (2, 2) {
    "some_use"(%arg0, %arg1) : (index, index) -> ()
  } {mapping = [#hivm.block, #hivm.sub_block<x>]}
  return
}

// -----
func.func @sub_block_only_test() {
  // CHECK: "some_use"({{.*}})
  // CHECK-NOT: scf.forall (
  scf.forall (%arg0) in (2) {
    "some_use"(%arg0) : (index) -> ()
  } {mapping = [#hivm.sub_block<x>]}
  return
}

// -----
#ceildiv = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
module {
  func.func @test_fuse_loop_for_parallel_axis_d_1(%arg: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@test_fuse_loop_for_parallel_axis_d_tiling_function>, hacc.block_dim = 48 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, transform.target_tag = "test_fuse_loop_for_parallel_axis_d_1_payload"} {
    %c49152 = arith.constant 49152 : index
    %c1 = arith.constant 1 : index
    %c3072 = arith.constant 3072 : index
    %c0 = arith.constant 0 : index
    %0 = "parllel_block_dim"() : () -> index
    %1 = "reduce_block_dim"() : () -> index
    %2 = arith.index_cast %arg : i64 to index
    %ub = affine.apply #ceildiv(%c3072)[%0]
    %iub = affine.apply #ceildiv(%c49152)[%1]
    // CHECK: %[[IDXI64:[a-zA-Z0-9]+]] = hivm.hir.get_block_idx
    // CHECK: %[[IDX:[a-zA-Z0-9]+]] = arith.index_cast %[[IDXI64]]
    // CHECK: affine.apply
    // CHECK-SAME: %[[IDX]]
    // CHECK: affine.apply
    // CHECK-SAME: %[[IDX]]
    // CHECK-NOT: scf.forall
    scf.forall (%arg7) = (%c0) to (%ub) step (%c1) {
      %3 = "some_use"(%arg7) : (index) -> index
      scf.for %arg8 = %c0 to %3 step %c1 {
        scf.forall (%arg9) = (%c0) to (%iub) step (%c1) {
          "some_use"(%arg9) : (index)->()
        } {mapping = [#hivm.block]}
        "some_use"(%arg8) : (index)->()
      }
    } {mapping = [#hivm.block]}
    return
  }
}
