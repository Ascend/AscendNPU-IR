// RUN: bishengir-opt %s -test-hivm-dimension-analyzer -split-input-file | FileCheck %s
// The larger column candidate misses the mandatory row UB. Allow the
// smaller row candidate even though its L1 copies remain untiled.
// CHECK: Tiling dim for hivm.hir.store {{.*}} is 0
// CHECK: Tiling dim for hivm.hir.store {{.*}} is 0
// CHECK-LABEL: func.func @partial_stores_cover_ub(
func.func @partial_stores_cover_ub(%matrix: tensor<16x32xf32>, %out: memref<16x32xf32>, %out_rows: memref<16xf32>) {
  %rows = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %cols = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  annotation.mark %rows {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
  %r = bufferization.to_tensor %rows restrict writable : memref<16xf32, #hivm.address_space<ub>>
  %c = bufferization.to_tensor %cols restrict writable : memref<32xf32, #hivm.address_space<ub>>
  %re = tensor.expand_shape %r [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  %ce = tensor.expand_shape %c [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %a = hivm.hir.vadd ins(%matrix, %re : tensor<16x32xf32>, tensor<16x1xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [1] -> tensor<16x32xf32>
  %b = hivm.hir.vadd ins(%a, %ce : tensor<16x32xf32>, tensor<1x32xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [0] -> tensor<16x32xf32>
  %l1 = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.store ins(%b : tensor<16x32xf32>) outs(%out : memref<16x32xf32>)
  hivm.hir.store ins(%r : tensor<16xf32>) outs(%out_rows : memref<16xf32>)
  return
}
// -----
// Candidate counts include multiple operations sharing one UB parent.
// The analyzer chooses the larger column candidate here; the bind pass is
// responsible for rejecting a result that leaves a required UB untiled.
// CHECK: Tiling dim for hivm.hir.store {{.*}} is 1
// CHECK: Tiling dim for hivm.hir.store {{.*}} is -1
// CHECK-LABEL: func.func @operation_count_with_conflicting_ubs(
func.func @operation_count_with_conflicting_ubs(%matrix: tensor<16x32xf32>, %out: memref<16x32xf32>, %out_rows: memref<16xf32>) {
  %rows = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %cols = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  annotation.mark %rows {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
  annotation.mark %cols {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<32xf32, #hivm.address_space<ub>>
  %r = bufferization.to_tensor %rows restrict writable : memref<16xf32, #hivm.address_space<ub>>
  %c = bufferization.to_tensor %cols restrict writable : memref<32xf32, #hivm.address_space<ub>>
  %re = tensor.expand_shape %r [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  %ce = tensor.expand_shape %c [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %a = hivm.hir.vadd ins(%matrix, %re : tensor<16x32xf32>, tensor<16x1xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [1] -> tensor<16x32xf32>
  %b = hivm.hir.vadd ins(%a, %ce : tensor<16x32xf32>, tensor<1x32xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [0] -> tensor<16x32xf32>
  %l1 = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.store ins(%b : tensor<16x32xf32>) outs(%out : memref<16x32xf32>)
  hivm.hir.store ins(%r : tensor<16xf32>) outs(%out_rows : memref<16xf32>)
  return
}
// -----
// No UB TCB: retain the original all-store coverage policy.
// CHECK: Tiling dim for hivm.hir.store {{.*}} is -1
// CHECK: Tiling dim for hivm.hir.store {{.*}} is -1
// CHECK-LABEL: func.func @no_ub_keeps_policy(
func.func @no_ub_keeps_policy(%matrix: tensor<16x32xf32>, %out: memref<16x32xf32>, %out_rows: memref<16xf32>) {
  %rows = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %cols = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  %r = bufferization.to_tensor %rows restrict writable : memref<16xf32, #hivm.address_space<ub>>
  %c = bufferization.to_tensor %cols restrict writable : memref<32xf32, #hivm.address_space<ub>>
  %re = tensor.expand_shape %r [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  %ce = tensor.expand_shape %c [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %a = hivm.hir.vadd ins(%matrix, %re : tensor<16x32xf32>, tensor<16x1xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [1] -> tensor<16x32xf32>
  %b = hivm.hir.vadd ins(%a, %ce : tensor<16x32xf32>, tensor<1x32xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [0] -> tensor<16x32xf32>
  %l1 = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.store ins(%b : tensor<16x32xf32>) outs(%out : memref<16x32xf32>)
  hivm.hir.store ins(%r : tensor<16xf32>) outs(%out_rows : memref<16xf32>)
  return
}
// -----
// A legacy selection is not replaced by the fallback. The later tiling
// pass still checks whether the actual UB buffers have been tiled.
// CHECK: Tiling dim for hivm.hir.store {{.*}} is 1
// CHECK-LABEL: func.func @existing_selection_is_preserved(
func.func @existing_selection_is_preserved(%matrix: tensor<16x32xf32>, %out: memref<16x32xf32>, %out_rows: memref<16xf32>) {
  %rows = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %cols = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  annotation.mark %rows {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
  %r = bufferization.to_tensor %rows restrict writable : memref<16xf32, #hivm.address_space<ub>>
  %c = bufferization.to_tensor %cols restrict writable : memref<32xf32, #hivm.address_space<ub>>
  %re = tensor.expand_shape %r [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  %ce = tensor.expand_shape %c [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %a = hivm.hir.vadd ins(%matrix, %re : tensor<16x32xf32>, tensor<16x1xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [1] -> tensor<16x32xf32>
  %b = hivm.hir.vadd ins(%a, %ce : tensor<16x32xf32>, tensor<1x32xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [0] -> tensor<16x32xf32>
  %l1 = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.store ins(%b : tensor<16x32xf32>) outs(%out : memref<16x32xf32>)
  return
}
// -----
// Both candidates cover the mandatory UB. Prefer the candidate covering
// more stores/copies.
// CHECK: Tiling dim for hivm.hir.store {{.*}} is 1
// CHECK: Tiling dim for hivm.hir.store {{.*}} is -1
// CHECK-LABEL: func.func @largest_covering_candidate(
func.func @largest_covering_candidate(%unused: tensor<16x32xf32>, %out: memref<16x32xf32>, %out_rows: memref<16xf32>) {
  %rows = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %cols = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  %matrix_ub = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
  annotation.mark %matrix_ub {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x32xf32, #hivm.address_space<ub>>
  %matrix = bufferization.to_tensor %matrix_ub restrict writable : memref<16x32xf32, #hivm.address_space<ub>>
  %r = bufferization.to_tensor %rows restrict writable : memref<16xf32, #hivm.address_space<ub>>
  %c = bufferization.to_tensor %cols restrict writable : memref<32xf32, #hivm.address_space<ub>>
  %re = tensor.expand_shape %r [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
  %ce = tensor.expand_shape %c [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
  %a = hivm.hir.vadd ins(%matrix, %re : tensor<16x32xf32>, tensor<16x1xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [1] -> tensor<16x32xf32>
  %b = hivm.hir.vadd ins(%a, %ce : tensor<16x32xf32>, tensor<1x32xf32>) outs(%matrix : tensor<16x32xf32>) broadcast = [0] -> tensor<16x32xf32>
  %l1 = memref.alloc() : memref<32xf32, #hivm.address_space<cbuf>>
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.copy ins(%c : tensor<32xf32>) outs(%l1 : memref<32xf32, #hivm.address_space<cbuf>>)
  hivm.hir.store ins(%b : tensor<16x32xf32>) outs(%out : memref<16x32xf32>)
  hivm.hir.store ins(%r : tensor<16xf32>) outs(%out_rows : memref<16xf32>)
  return
}
