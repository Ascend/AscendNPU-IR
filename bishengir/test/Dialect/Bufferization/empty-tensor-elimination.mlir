// RUN: bishengir-opt %s -empty-tensor-elimination="bufferize-function-boundaries=true allow-return-allocs-from-loops=true allow-unknown-ops=true analysis-heuristic=top-down check-parallel-regions=true" -split-input-file | FileCheck %s --implicit-check-not=tensor.empty

// Match the One-Shot analysis options used by bufferizationPipelineBase.

// A restricted, writable memref destination replaces the producer's empty
// tensor with a to_tensor view of the destination buffer.
// CHECK-LABEL: func.func @materialize_in_destination(
// CHECK-SAME: %[[SRC:.*]]: tensor<8xf32>, %[[DST:.*]]: memref<8xf32>
// CHECK: %[[INIT:.*]] = bufferization.to_tensor %[[DST]] restrict writable : memref<8xf32>
// CHECK: %[[RESULT:.*]] = hivm.hir.vadd ins(%[[SRC]], %[[SRC]] : tensor<8xf32>, tensor<8xf32>) outs(%[[INIT]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: bufferization.materialize_in_destination %[[RESULT]] in writable %[[DST]] : (tensor<8xf32>, memref<8xf32>) -> ()
// CHECK: return
func.func @materialize_in_destination(%src: tensor<8xf32>, %dst: memref<8xf32>) {
  %empty = tensor.empty() : tensor<8xf32>
  %result = hivm.hir.vadd ins(%src, %src : tensor<8xf32>, tensor<8xf32>) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  bufferization.materialize_in_destination %result in restrict writable %dst : (tensor<8xf32>, memref<8xf32>) -> ()
  return
}

// -----

// Reuse the insertion destination's subset as the producer's init tensor,
// preserving the original extraction used as input and the insertion offsets.
// CHECK-LABEL: func.func @extract_insert_slice(
// CHECK-SAME: %[[SRC:.*]]: tensor<8xf32>, %[[DST:.*]]: tensor<16xf32>
// CHECK: %[[INPUT:.*]] = tensor.extract_slice %[[DST]][4] [8] [1] : tensor<16xf32> to tensor<8xf32>
// CHECK: %[[INIT:.*]] = tensor.extract_slice %[[DST]][4] [8] [1] : tensor<16xf32> to tensor<8xf32>
// CHECK: %[[RESULT:.*]] = hivm.hir.vadd ins(%[[SRC]], %[[INPUT]] : tensor<8xf32>, tensor<8xf32>) outs(%[[INIT]] : tensor<8xf32>) -> tensor<8xf32>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[RESULT]] into %[[DST]][4] [8] [1] : tensor<8xf32> into tensor<16xf32>
// CHECK: return %[[INSERT]] : tensor<16xf32>
func.func @extract_insert_slice(%src: tensor<8xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> {
  %input = tensor.extract_slice %dst[4] [8] [1] : tensor<16xf32> to tensor<8xf32>
  %empty = tensor.empty() : tensor<8xf32>
  %result = hivm.hir.vadd ins(%src, %input : tensor<8xf32>, tensor<8xf32>) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %insert = tensor.insert_slice %result into %dst[4] [8] [1] : tensor<8xf32> into tensor<16xf32>
  return %insert : tensor<16xf32>
}

// -----

// Reduced from kernel_da_bwd_q_u: fused_9 has two destination arguments,
// nested row/column loops, vector transfers, and two returned tensors.
// Its caller passes separate empty tensors. Internal insert_slice operations
// only update these VF arguments; they do not provide another destination
// for the caller's empty tensors. Both allocations must remain.
// CHECK-LABEL: func.func @fused_two_output_vf(
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK-SAME: iter_args({{.*}}) -> (tensor<32x128xf32>, tensor<32x128xf32>)
// CHECK: scf.for
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1 : tensor<32x128xf32>, tensor<32x128xf32>
func.func @fused_two_output_vf(%arg0: f32, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>, %arg3: tensor<32x128xf32>, %arg4: tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %0 = vector.broadcast %arg0 : f32 to vector<1x64xf32>
  %1:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
    %2:2 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
      %extracted_slice = tensor.extract_slice %arg9[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %3 = vector.transfer_write %0, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg9[%arg5, %arg8] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<32x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %extracted_slice_2 = tensor.extract_slice %arg10[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %4 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %6 = arith.mulf %5, %0 : vector<1x64xf32>
      %7 = arith.mulf %4, %0 : vector<1x64xf32>
      %8 = arith.addf %7, %6 : vector<1x64xf32>
      %9 = vector.transfer_write %8, %extracted_slice_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice_3 = tensor.insert_slice %9 into %arg10[%arg5, %arg8] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<32x128xf32>
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<32x128xf32>, tensor<32x128xf32>
    }
    scf.yield %2#0, %2#1 : tensor<32x128xf32>, tensor<32x128xf32>
  }
  return %1#0, %1#1 : tensor<32x128xf32>, tensor<32x128xf32>
}

// CHECK-LABEL: func.func @vf_preserve_empty_without_destination(
// CHECK-SAME: %[[SCALE:.*]]: f32, %[[A:.*]]: memref<32x128xf32>, %[[B:.*]]: memref<32x128xf32>
// CHECK: %[[AT:.*]] = bufferization.to_tensor %[[A]] restrict writable : memref<32x128xf32>
// CHECK: %[[BT:.*]] = bufferization.to_tensor %[[B]] restrict writable : memref<32x128xf32>
// CHECK: %[[E0:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK: %[[E1:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK: %[[R:.*]]:2 = call @fused_two_output_vf(%[[SCALE]], %[[AT]], %[[BT]], %[[E0]], %[[E1]]) {hivm.vector_function, no_inline}
// CHECK: return %[[R]]#0, %[[R]]#1 : tensor<32x128xf32>, tensor<32x128xf32>
func.func @vf_preserve_empty_without_destination(%scale: f32, %a: memref<32x128xf32>, %b: memref<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
  %at = bufferization.to_tensor %a restrict writable : memref<32x128xf32>
  %bt = bufferization.to_tensor %b restrict writable : memref<32x128xf32>
  %e0 = tensor.empty() : tensor<32x128xf32>
  %e1 = tensor.empty() : tensor<32x128xf32>
  %r:2 = func.call @fused_two_output_vf(%scale, %at, %bt, %e0, %e1) {hivm.vector_function, no_inline} : (f32, tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>)
  return %r#0, %r#1 : tensor<32x128xf32>, tensor<32x128xf32>
}

// -----

// The same real VF becomes eligible when its caller explicitly materializes
// both results in restricted, writable buffers. Both destination arguments
// must be replaced independently, preserving result-to-argument mapping.
// CHECK-LABEL: func.func @fused_two_output_vf(
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK-SAME: iter_args({{.*}}) -> (tensor<32x128xf32>, tensor<32x128xf32>)
// CHECK: scf.for
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1 : tensor<32x128xf32>, tensor<32x128xf32>
func.func @fused_two_output_vf(%arg0: f32, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>, %arg3: tensor<32x128xf32>, %arg4: tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %0 = vector.broadcast %arg0 : f32 to vector<1x64xf32>
  %1:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
    %2:2 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
      %extracted_slice = tensor.extract_slice %arg9[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %3 = vector.transfer_write %0, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg9[%arg5, %arg8] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<32x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %extracted_slice_2 = tensor.extract_slice %arg10[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %4 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %6 = arith.mulf %5, %0 : vector<1x64xf32>
      %7 = arith.mulf %4, %0 : vector<1x64xf32>
      %8 = arith.addf %7, %6 : vector<1x64xf32>
      %9 = vector.transfer_write %8, %extracted_slice_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice_3 = tensor.insert_slice %9 into %arg10[%arg5, %arg8] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<32x128xf32>
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<32x128xf32>, tensor<32x128xf32>
    }
    scf.yield %2#0, %2#1 : tensor<32x128xf32>, tensor<32x128xf32>
  }
  return %1#0, %1#1 : tensor<32x128xf32>, tensor<32x128xf32>
}

// CHECK-LABEL: func.func @vf_eliminate_empty_with_destination(
// CHECK-SAME: %[[SCALE:.*]]: f32, %[[A:.*]]: tensor<32x128xf32>, %[[B:.*]]: tensor<32x128xf32>, %[[D0:.*]]: memref<32x128xf32>, %[[D1:.*]]: memref<32x128xf32>
// CHECK-DAG: %[[T0:.*]] = bufferization.to_tensor %[[D0]] restrict writable : memref<32x128xf32>
// CHECK-DAG: %[[T1:.*]] = bufferization.to_tensor %[[D1]] restrict writable : memref<32x128xf32>
// CHECK: %[[R:.*]]:2 = call @fused_two_output_vf(%[[SCALE]], %[[A]], %[[B]], %[[T0]], %[[T1]]) {hivm.vector_function, no_inline}
// CHECK: bufferization.materialize_in_destination %[[R]]#0 in writable %[[D0]]
// CHECK: bufferization.materialize_in_destination %[[R]]#1 in writable %[[D1]]
// CHECK: return
func.func @vf_eliminate_empty_with_destination(%scale: f32, %a: tensor<32x128xf32>, %b: tensor<32x128xf32>, %d0: memref<32x128xf32>, %d1: memref<32x128xf32>) {
  %e0 = tensor.empty() : tensor<32x128xf32>
  %e1 = tensor.empty() : tensor<32x128xf32>
  %r:2 = func.call @fused_two_output_vf(%scale, %a, %b, %e0, %e1) {hivm.vector_function, no_inline} : (f32, tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>)
  bufferization.materialize_in_destination %r#0 in restrict writable %d0 : (tensor<32x128xf32>, memref<32x128xf32>) -> ()
  bufferization.materialize_in_destination %r#1 in restrict writable %d1 : (tensor<32x128xf32>, memref<32x128xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @fused_two_output_vf(
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK-SAME: iter_args({{.*}}) -> (tensor<32x128xf32>, tensor<32x128xf32>)
// CHECK: scf.for
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1 : tensor<32x128xf32>, tensor<32x128xf32>
func.func @fused_two_output_vf(%arg0: f32, %arg1: tensor<32x128xf32>, %arg2: tensor<32x128xf32>, %arg3: tensor<32x128xf32>, %arg4: tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %0 = vector.broadcast %arg0 : f32 to vector<1x64xf32>
  %1:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
    %2:2 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<32x128xf32>, tensor<32x128xf32>) {
      %extracted_slice = tensor.extract_slice %arg9[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %3 = vector.transfer_write %0, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg9[%arg5, %arg8] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<32x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %extracted_slice_2 = tensor.extract_slice %arg10[%arg5, %arg8] [1, 64] [1, 1] : tensor<32x128xf32> to tensor<1x64xf32>
      %4 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %6 = arith.mulf %5, %0 : vector<1x64xf32>
      %7 = arith.mulf %4, %0 : vector<1x64xf32>
      %8 = arith.addf %7, %6 : vector<1x64xf32>
      %9 = vector.transfer_write %8, %extracted_slice_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice_3 = tensor.insert_slice %9 into %arg10[%arg5, %arg8] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<32x128xf32>
      scf.yield %inserted_slice, %inserted_slice_3 : tensor<32x128xf32>, tensor<32x128xf32>
    }
    scf.yield %2#0, %2#1 : tensor<32x128xf32>, tensor<32x128xf32>
  }
  return %1#0, %1#1 : tensor<32x128xf32>, tensor<32x128xf32>
}

// Extract input tiles from larger tensors and insert the two VF results
// into distinct destination subsets. Empty elimination must pass matching
// destination slices to the VF without swapping the two results or offsets.
// CHECK-LABEL: func.func @vf_eliminate_empty_with_slices(
// CHECK-SAME: %[[SCALE:.*]]: f32, %[[A:.*]]: tensor<64x256xf32>, %[[B:.*]]: tensor<64x256xf32>, %[[D0:.*]]: tensor<64x256xf32>, %[[D1:.*]]: tensor<64x256xf32>
// CHECK: %[[A_TILE:.*]] = tensor.extract_slice %[[A]][0, 64] [32, 128] [1, 1] : tensor<64x256xf32> to tensor<32x128xf32>
// CHECK: %[[B_TILE:.*]] = tensor.extract_slice %[[B]][16, 0] [32, 128] [1, 1] : tensor<64x256xf32> to tensor<32x128xf32>
// CHECK-DAG: %[[T0:.*]] = tensor.extract_slice %[[D0]][0, 64] [32, 128] [1, 1] : tensor<64x256xf32> to tensor<32x128xf32>
// CHECK-DAG: %[[T1:.*]] = tensor.extract_slice %[[D1]][16, 128] [32, 128] [1, 1] : tensor<64x256xf32> to tensor<32x128xf32>
// CHECK: %[[R:.*]]:2 = call @fused_two_output_vf(%[[SCALE]], %[[A_TILE]], %[[B_TILE]], %[[T0]], %[[T1]]) {hivm.vector_function, no_inline}
// CHECK: %[[OUT0:.*]] = tensor.insert_slice %[[R]]#0 into %[[D0]][0, 64] [32, 128] [1, 1] : tensor<32x128xf32> into tensor<64x256xf32>
// CHECK: %[[OUT1:.*]] = tensor.insert_slice %[[R]]#1 into %[[D1]][16, 128] [32, 128] [1, 1] : tensor<32x128xf32> into tensor<64x256xf32>
// CHECK: return %[[OUT0]], %[[OUT1]] : tensor<64x256xf32>, tensor<64x256xf32>
func.func @vf_eliminate_empty_with_slices(%scale: f32, %a: tensor<64x256xf32>, %b: tensor<64x256xf32>, %d0: tensor<64x256xf32>, %d1: tensor<64x256xf32>) -> (tensor<64x256xf32>, tensor<64x256xf32>) {
  %a_tile = tensor.extract_slice %a[0, 64] [32, 128] [1, 1] : tensor<64x256xf32> to tensor<32x128xf32>
  %b_tile = tensor.extract_slice %b[16, 0] [32, 128] [1, 1] : tensor<64x256xf32> to tensor<32x128xf32>
  %e0 = tensor.empty() : tensor<32x128xf32>
  %e1 = tensor.empty() : tensor<32x128xf32>
  %r:2 = func.call @fused_two_output_vf(%scale, %a_tile, %b_tile, %e0, %e1) {hivm.vector_function, no_inline} : (f32, tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> (tensor<32x128xf32>, tensor<32x128xf32>)
  %out0 = tensor.insert_slice %r#0 into %d0[0, 64] [32, 128] [1, 1] : tensor<32x128xf32> into tensor<64x256xf32>
  %out1 = tensor.insert_slice %r#1 into %d1[16, 128] [32, 128] [1, 1] : tensor<32x128xf32> into tensor<64x256xf32>
  return %out0, %out1 : tensor<64x256xf32>, tensor<64x256xf32>
}
