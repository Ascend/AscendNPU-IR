// RUN: bishengir-opt -hivm-insert-fixpipe %s -split-input-file | FileCheck %s

// Verify fixpipe is moved into the scf.if branch that yields the loop result.
//
// CHECK-LABEL: func.func @move_fixpipe_into_scf_if_branch
// CHECK: scf.for
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK: scf.if
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK: } else {
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK-NOT: hivm.hir.fixpipe
func.func @move_fixpipe_into_scf_if_branch() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %false = arith.constant false
  %cst = arith.constant 0.000000e+00 : f32
  %init_a = tensor.empty() : tensor<16x16xf16>
  %init_b = tensor.empty() : tensor<16x16xf16>
  %init_c = tensor.empty() : tensor<16x16xf32>
  %fallback = tensor.empty() : tensor<16x16xf32>
  %for_res = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init_c) -> (tensor<16x16xf32>) : i32 {
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%init_a, %init_b, %true, %c16, %c16, %c16
            : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
        outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  %if_res = scf.if %false -> (tensor<16x16xf32>) {
    %vbrc = hivm.hir.vbrc ins(%cst : f32) outs(%fallback : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %vbrc : tensor<16x16xf32>
  } else {
    scf.yield %for_res : tensor<16x16xf32>
  }
  return %if_res : tensor<16x16xf32>
}

// -----

// Verify a fixpipe already nested in an outer scf.if is moved into the inner
// scf.if branch that yields it (the checker must not reject "already inside
// scf.if").
//
//   scf.if {
//     %fix = fixpipe
//     scf.if { ... } else { yield %fix }
//   }
//
// CHECK-LABEL: func.func @move_fixpipe_into_nested_scf_if_branch
// CHECK: %[[FOR:.*]] = scf.for
// CHECK: scf.if
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.if
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK: } else {
// CHECK: %[[FIX:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[FOR]]
// CHECK-NEXT: scf.yield %[[FIX]] : tensor<16x16xf32>
// CHECK-NOT: hivm.hir.fixpipe
func.func @move_fixpipe_into_nested_scf_if_branch() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %cst = arith.constant 0.000000e+00 : f32
  %init_c = tensor.empty() : tensor<16x16xf32>
  %fallback = tensor.empty() : tensor<16x16xf32>
  %for_res = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init_c) -> (tensor<16x16xf32>) : i32 {
    scf.yield %acc : tensor<16x16xf32>
  }
  %if_res = scf.if %false -> (tensor<16x16xf32>) {
    %fix_out = tensor.empty() : tensor<16x16xf32>
    %fix = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
        ins(%for_res : tensor<16x16xf32>) outs(%fix_out : tensor<16x16xf32>)
        -> tensor<16x16xf32>
    %inner = scf.if %false -> (tensor<16x16xf32>) {
      %vbrc = hivm.hir.vbrc ins(%cst : f32) outs(%fallback : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %vbrc : tensor<16x16xf32>
    } else {
      scf.yield %fix : tensor<16x16xf32>
    }
    scf.yield %inner : tensor<16x16xf32>
  } else {
    scf.yield %fallback : tensor<16x16xf32>
  }
  return %if_res : tensor<16x16xf32>
}

// -----

// Verify iter-arg fixpipe stays inside scf.for when tagged do_not_move_out_of_scffor.
//
// CHECK-LABEL: func.func @keep_fixpipe_inside_scf_for_with_attr
// CHECK: scf.for
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true}
// CHECK: scf.yield %{{.*}} : tensor<16x16xf32>
// CHECK-NOT: do_not_move_out_of_scffor = true
func.func @keep_fixpipe_inside_scf_for_with_attr() -> tensor<16x16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %init_a = tensor.empty() : tensor<16x16xf16>
  %init_b = tensor.empty() : tensor<16x16xf16>
  %init_c = tensor.empty() : tensor<16x16xf32>
  %for_res = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init_c) -> (tensor<16x16xf32>) : i32 {
    %empty = tensor.empty() : tensor<16x16xf32>
    %mmad = hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C}
        ins(%acc, %init_b, %true, %c16, %c16, %c16
            : tensor<16x16xf32>, tensor<16x16xf16>, i1, index, index, index)
        outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mmad : tensor<16x16xf32>
  }
  return %for_res : tensor<16x16xf32>
}
