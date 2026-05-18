// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

#map = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>

module {
  // CHECK-LABEL: tt.func @affine_apply_to_arith
  // CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
  // CHECK: %[[IDX0:.*]] = arith.index_cast %[[ARG0]] : i32 to index
  // CHECK: %[[IDX1:.*]] = arith.index_cast %[[ARG1]] : i32 to index
  // CHECK: %[[IDX2:.*]] = arith.index_cast %[[ARG2]] : i32 to index
  // CHECK: %[[MUL:.*]] = arith.muli %[[IDX1]], %[[IDX2]] : index
  // CHECK: %[[ADD:.*]] = arith.addi %[[IDX0]], %[[MUL]] : index
  // CHECK: arith.index_cast %[[ADD]] : index to i32
  // CHECK-NOT: affine.apply
  func.func @affine_apply_to_arith(%arg0: i32, %arg1: i32, %arg2: i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %idx0 = arith.index_cast %arg0 : i32 to index
    %idx1 = arith.index_cast %arg1 : i32 to index
    %idx2 = arith.index_cast %arg2 : i32 to index
    %offset = affine.apply #map()[%idx0, %idx1, %idx2]
    %offset_i32 = arith.index_cast %offset : index to i32
    return
  }
}
