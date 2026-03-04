// RUN: bishengir-opt %s -map-for-to-forall -allow-unregistered-dialect --split-input-file | FileCheck %s

#map = affine_map<()[s0, s1] -> (-s0 + 3072, s1)>
#map1 = affine_map<()[s0, s1] -> (-s0 + 49152, s1)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 49152, s0)>
module {
  func.func @test_fuse_loop_for_parallel_axis_d_1(%arg2: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
                                                  %arg5: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> memref<3072xf32> 
                                                  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.block_dim = 48 : i64} {
    %c49152 = arith.constant 49152 : index
    %c1 = arith.constant 1 : index
    %c3072 = arith.constant 3072 : index
    %c0 = arith.constant 0 : index
    %0 = "parllel_block_dim"() : () -> index
    %1 = "reduce_block_dim"() : () -> index
    // CHECK: scf.forall
    scf.for %arg7 = %c0 to %c3072 step %0 {
      %2 = affine.min #map()[%arg7, %0]
      // CHECK: scf.for
      scf.for %arg8 = %c0 to %2 step %c1 {
        %4 = arith.index_cast %arg5 : i64 to index
        // CHECK: scf.forall
        scf.for %arg9 = %c0 to %c49152 step %1 {
          %5 = affine.min #map1()[%arg9, %1]
          // CHECK: scf.for
          scf.for %arg10 = %c0 to %5 step %4 {
            %6 = arith.addi %arg9, %arg10 : index
            %7 = affine.min #map2(%6)[%4]
            "dummy_op"(%arg10, %arg9, %arg8) : (index, index, index) -> ()
          }
        // CHECK: mapping = [#hivm.block
        } {map_for_to_forall}
      }
    // CHECK: mapping = [#hivm.block
    } {map_for_to_forall}
    return %arg2 : memref<3072xf32>
  }
}

// -----

#map = affine_map<()[s0, s1] -> (-s0 + 3072, s1)>
#map1 = affine_map<()[s0, s1] -> (-s0 + 49152, s1)>
#map2 = affine_map<(d0)[s0] -> (-d0 + 49152, s0)>
module {
  func.func @test_forward_mapping(%arg2: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
                                                  %arg5: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) -> memref<3072xf32> 
                                                  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.block_dim = 48 : i64} {
    %c49152 = arith.constant 49152 : index
    %c1 = arith.constant 1 : index
    %c3072 = arith.constant 3072 : index
    %c0 = arith.constant 0 : index
    %0 = "parllel_block_dim"() : () -> index
    %1 = "reduce_block_dim"() : () -> index
    // CHECK: scf.forall
    scf.for %arg7 = %c0 to %c3072 step %0 {
      %2 = affine.min #map()[%arg7, %0]
      // CHECK: scf.for
      scf.for %arg8 = %c0 to %2 step %c1 {
        %4 = arith.index_cast %arg5 : i64 to index
        // CHECK: scf.forall
        scf.for %arg9 = %c0 to %c49152 step %1 {
          %5 = affine.min #map1()[%arg9, %1]
          // CHECK: scf.for
          scf.for %arg10 = %c0 to %5 step %4 {
            %6 = arith.addi %arg9, %arg10 : index
            %7 = affine.min #map2(%6)[%4]
            "dummy_op"(%arg10, %arg9, %arg8) : (index, index, index) -> ()
          }
        // CHECK: mapping = [#hivm.block<linear_dim = 1
        } {map_for_to_forall, mapping = [#hivm.block<linear_dim = 1>]}
      }
    // CHECK: mapping = [#hivm.block<linear_dim = 0
    } {map_for_to_forall, mapping = [#hivm.block<linear_dim = 0>]}
    return %arg2 : memref<3072xf32>
  }
}

// -----
func.func @test_for_to_forall_with_sub_block(%arg1: index, %arg2: index, %arg3: index) {
  // CHECK: scf.forall
  // CHECK: #hivm.sub_block<x>
  scf.for %arg4 = %arg1 to %arg2 step %arg3 {
    %1 = "offset_compute"(%arg4, %arg1, %arg2) : (index, index, index) -> index
    %2 = "size_compute"(%arg4, %arg1, %arg2) : (index, index, index) -> index
  } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  return
}