// RUN: bishengir-opt -intra-stage-scheduling -split-input-file %s | FileCheck %s

// Reduced from intra-stage-scheduling-kernel.mlir. A vector scope prepares an
// L1 buffer for the next cube scope and also computes values that stay on the
// vector core. The cross-core copy and everything it depends on move into one
// `vf-scope` group at the top of the body, so the cube waits for that group
// instead of the whole scope.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @hoist_cube_dependency() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %alloc = memref.alloc() : memref<128x128xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
  %1 = scf.for %arg0 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg1 = %0) -> (tensor<128x128xf32>)  : i32 {
    %2:2 = scope.scope : () -> (tensor<128x128xbf16>, tensor<128x128xf32>) {
      %3:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
        // Stays behind: the cube scope reads nothing that depends on it.
        %alloc_0 = memref.alloc() : memref<128xf32>
        %6 = bufferization.to_tensor %alloc_0 restrict writable : memref<128xf32>
        %7 = tensor.empty() : tensor<128xf32>
        %8 = hivm.hir.vexp ins(%6 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %expanded = tensor.expand_shape %8 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %9 = tensor.empty() : tensor<128x128xf32>
        %10 = hivm.hir.vmul ins(%arg1, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%9 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        // Moves: the copy filling the L1 buffer, and its dependencies.
        %11 = tensor.empty() : tensor<128x128xbf16>
        %12 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%arg1 : tensor<128x128xf32>) outs(%11 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %expanded_1 = tensor.expand_shape %12 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
        %13 = tensor.empty() : tensor<8x128x16xbf16>
        %14 = hivm.hir.vtranspose ins(%expanded_1 : tensor<128x8x16xbf16>) outs(%13 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
        %expanded_2 = tensor.expand_shape %14 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
        %alloc_3 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %15 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_2 : tensor<8x8x16x16xbf16>) outs(%alloc_3 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        scope.return %10, %12, %15 : tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %4 = scope.scope : () -> tensor<64x128xf32> {
        %alloc_0 = memref.alloc() : memref<8x4x16x16xbf16>
        %6 = bufferization.to_tensor %alloc_0 restrict writable : memref<8x4x16x16xbf16>
        %7 = tensor.empty() : tensor<8x4x16x16xf32>
        %8 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%6, %3#2, %true, %c64, %c128, %c128 : tensor<8x4x16x16xbf16>, tensor<8x8x16x16xbf16>, i1, index, index, index) outs(%7 : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_1 : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %9 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%8 : tensor<8x4x16x16xf32>) outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>)
        scope.return %9 : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      // Reads the vector result that stayed behind, so not every result of the
      // vector scope feeds the cube.
      %5 = scope.scope : () -> tensor<128x128xf32> {
        %6 = tensor.empty() : tensor<64x128xf32>
        %7 = hivm.hir.vexp ins(%4 : tensor<64x128xf32>) outs(%6 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %8 = tensor.empty() : tensor<128x128xf32>
        %9 = hivm.hir.vexp ins(%3#0 : tensor<128x128xf32>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scope.return %9 : tensor<128x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %3#1, %5 : tensor<128x128xbf16>, tensor<128x128xf32>
    } {cv_merged_scope}
    // Returns nothing, so there is no dependency to hand to a cube scope.
    scope.scope : () -> () {
      hivm.hir.set_ctrl true at ctrl[60]
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %2#1 : tensor<128x128xf32>
  } {hivm.cv_pipelined_loop}
  return
}
}

// CHECK-LABEL: func.func @hoist_cube_dependency
// CHECK:         scope.scope : () -> (tensor<128x128xbf16>, tensor<128x128xf32>) {
// CHECK-NEXT:      %[[VF:.*]]:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
// CHECK-NEXT:        %[[GROUP:.*]]:2 = scope.scope : () -> (tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
// CHECK-NEXT:          %[[EMPTY0:.*]] = tensor.empty() : tensor<128x128xbf16>
// CHECK-NEXT:          %[[CAST:.*]] = hivm.hir.vcast {{.*}} outs(%[[EMPTY0]] : tensor<128x128xbf16>) -> tensor<128x128xbf16>
// CHECK-NEXT:          %[[EXP0:.*]] = tensor.expand_shape %[[CAST]] {{.*}} into tensor<128x8x16xbf16>
// CHECK-NEXT:          %[[EMPTY1:.*]] = tensor.empty() : tensor<8x128x16xbf16>
// CHECK-NEXT:          %[[TRANS:.*]] = hivm.hir.vtranspose ins(%[[EXP0]] {{.*}}) outs(%[[EMPTY1]] {{.*}}
// CHECK-NEXT:          %[[EXP1:.*]] = tensor.expand_shape %[[TRANS]] {{.*}} into tensor<8x8x16x16xbf16>
// CHECK-NEXT:          %[[ALLOC:.*]] = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
// CHECK-NEXT:          %[[L1:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK-NEXT:          hivm.hir.copy ins(%[[EXP1]] : tensor<8x8x16x16xbf16>) outs(%[[ALLOC]] :
// CHECK-NEXT:          scope.return %[[CAST]], %[[L1]] : tensor<128x128xbf16>, tensor<8x8x16x16xbf16>
// CHECK-NEXT:        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, "vf-scope"}

// What is left of the vector scope runs after the group, and the values the
// group produced reach the terminator through its results.
// CHECK-NEXT:        %[[KEPT:.*]] = memref.alloc() : memref<128xf32>
// CHECK:             %[[VMUL:.*]] = hivm.hir.vmul {{.*}} broadcast = [1] -> tensor<128x128xf32>
// CHECK-NEXT:        scope.return %[[VMUL]], %[[GROUP]]#0, %[[GROUP]]#1 : tensor<128x128xf32>, tensor<128x128xbf16>, tensor<8x8x16x16xbf16>
// CHECK-NEXT:      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}

// The cube scope still reads the L1 tensor through the vector scope result.
// CHECK:           hivm.hir.mmadL1 {{.*}}, %[[VF]]#2,

// CHECK:         scope.scope : () -> () {
// CHECK-NEXT:      hivm.hir.set_ctrl

// -----

// Every result of the vector scope feeds the next cube scope, so nothing is
// left to run after the group and the scope is untouched.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @all_results_feed_cube() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %alloc = memref.alloc() : memref<128x128xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
  %1 = scf.for %arg0 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg1 = %0) -> (tensor<128x128xf32>)  : i32 {
    %2 = scope.scope : () -> tensor<128x128xf32> {
      %3:2 = scope.scope : () -> (tensor<8x8x16x16xbf16>, tensor<8x4x16x16xbf16>) {
        %6 = tensor.empty() : tensor<128x128xbf16>
        %7 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%arg1 : tensor<128x128xf32>) outs(%6 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %expanded = tensor.expand_shape %7 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
        %8 = tensor.empty() : tensor<8x128x16xbf16>
        %9 = hivm.hir.vtranspose ins(%expanded : tensor<128x8x16xbf16>) outs(%8 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
        %expanded_0 = tensor.expand_shape %9 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
        %alloc_1 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %10 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_0 : tensor<8x8x16x16xbf16>) outs(%alloc_1 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %alloc_2 = memref.alloc() : memref<8x4x16x16xbf16, #hivm.address_space<cbuf>>
        %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<8x4x16x16xbf16, #hivm.address_space<cbuf>>
        scope.return %10, %11 : tensor<8x8x16x16xbf16>, tensor<8x4x16x16xbf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %4 = scope.scope : () -> tensor<64x128xf32> {
        %6 = tensor.empty() : tensor<8x4x16x16xf32>
        %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%3#1, %3#0, %true, %c64, %c128, %c128 : tensor<8x4x16x16xbf16>, tensor<8x8x16x16xbf16>, i1, index, index, index) outs(%6 : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_1 : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %8 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<8x4x16x16xf32>) outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>)
        scope.return %8 : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %5 = scope.scope : () -> tensor<128x128xf32> {
        %6 = tensor.empty() : tensor<64x128xf32>
        %7 = hivm.hir.vexp ins(%4 : tensor<64x128xf32>) outs(%6 : tensor<64x128xf32>) -> tensor<64x128xf32>
        %8 = tensor.empty() : tensor<128x128xf32>
        %9 = hivm.hir.vexp ins(%arg1 : tensor<128x128xf32>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scope.return %9 : tensor<128x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %5 : tensor<128x128xf32>
    } {cv_merged_scope}
    scf.yield %2 : tensor<128x128xf32>
  } {hivm.cv_pipelined_loop}
  return
}
}

// CHECK-LABEL: func.func @all_results_feed_cube
// CHECK:         scope.scope : () -> (tensor<8x8x16x16xbf16>, tensor<8x4x16x16xbf16>) {
// CHECK-NEXT:      tensor.empty() : tensor<128x128xbf16>

// -----

// The vector scope keeps a result on the vector core, but every one of its
// operations feeds the cube scope anyway, so there is nothing to hoist.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @whole_body_feeds_cube() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %alloc = memref.alloc() : memref<128x128xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
  %1 = scf.for %arg0 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg1 = %0) -> (tensor<128x128xf32>)  : i32 {
    %2 = scope.scope : () -> tensor<128x128xf32> {
      %3:2 = scope.scope : () -> (tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
        %6 = tensor.empty() : tensor<128x128xbf16>
        %7 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%arg1 : tensor<128x128xf32>) outs(%6 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
        %expanded = tensor.expand_shape %7 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
        %8 = tensor.empty() : tensor<8x128x16xbf16>
        %9 = hivm.hir.vtranspose ins(%expanded : tensor<128x8x16xbf16>) outs(%8 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
        %expanded_0 = tensor.expand_shape %9 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
        %alloc_1 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        %10 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_0 : tensor<8x8x16x16xbf16>) outs(%alloc_1 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        scope.return %7, %10 : tensor<128x128xbf16>, tensor<8x8x16x16xbf16>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      %4 = scope.scope : () -> tensor<64x128xf32> {
        %alloc_1 = memref.alloc() : memref<8x4x16x16xbf16>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x4x16x16xbf16>
        %7 = tensor.empty() : tensor<8x4x16x16xf32>
        %8 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%6, %3#1, %true, %c64, %c128, %c128 : tensor<8x4x16x16xbf16>, tensor<8x8x16x16xbf16>, i1, index, index, index) outs(%7 : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        %alloc_2 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_2 : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %9 = bufferization.to_tensor %memspacecast restrict writable : memref<64x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%8 : tensor<8x4x16x16xf32>) outs(%alloc_2 : memref<64x128xf32, #hivm.address_space<ub>>)
        scope.return %9 : tensor<64x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
      %5 = scope.scope : () -> tensor<128x128xf32> {
        %6 = tensor.empty() : tensor<128x128xf32>
        %7 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%3#0 : tensor<128x128xbf16>) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %8 = tensor.empty() : tensor<64x128xf32>
        %9 = hivm.hir.vexp ins(%4 : tensor<64x128xf32>) outs(%8 : tensor<64x128xf32>) -> tensor<64x128xf32>
        scope.return %7 : tensor<128x128xf32>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
      scope.return %5 : tensor<128x128xf32>
    } {cv_merged_scope}
    scf.yield %2 : tensor<128x128xf32>
  } {hivm.cv_pipelined_loop}
  return
}
}

// CHECK-LABEL: func.func @whole_body_feeds_cube
// CHECK:         scope.scope : () -> (tensor<128x128xbf16>, tensor<8x8x16x16xbf16>) {
// CHECK-NEXT:      tensor.empty() : tensor<128x128xbf16>

// -----

// No cube scope follows the vector scope in its block, so the stage has no
// cube dependency to hoist for.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @no_following_cube_scope() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %alloc = memref.alloc() : memref<128x128xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf32>
  %1 = scf.for %arg0 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg1 = %0) -> (tensor<128x128xf32>)  : i32 {
    %2:2 = scope.scope : () -> (tensor<128x128xf32>, tensor<8x8x16x16xbf16>) {
      %4 = tensor.empty() : tensor<128x128xf32>
      %5 = hivm.hir.vexp ins(%arg1 : tensor<128x128xf32>) outs(%4 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %6 = tensor.empty() : tensor<128x128xbf16>
      %7 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%arg1 : tensor<128x128xf32>) outs(%6 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
      %expanded = tensor.expand_shape %7 [[0], [1, 2]] output_shape [128, 8, 16] : tensor<128x128xbf16> into tensor<128x8x16xbf16>
      %8 = tensor.empty() : tensor<8x128x16xbf16>
      %9 = hivm.hir.vtranspose ins(%expanded : tensor<128x8x16xbf16>) outs(%8 : tensor<8x128x16xbf16>) permutation = [1, 0, 2] -> tensor<8x128x16xbf16>
      %expanded_0 = tensor.expand_shape %9 [[0], [1, 2], [3]] output_shape [8, 8, 16, 16] : tensor<8x128x16xbf16> into tensor<8x8x16x16xbf16>
      %alloc_1 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
      %10 = bufferization.to_tensor %alloc_1 restrict writable : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%expanded_0 : tensor<8x8x16x16xbf16>) outs(%alloc_1 : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %5, %10 : tensor<128x128xf32>, tensor<8x8x16x16xbf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    %3 = scope.scope : () -> tensor<128x128xf32> {
      %4 = tensor.empty() : tensor<128x128xf32>
      %5 = hivm.hir.vexp ins(%2#0 : tensor<128x128xf32>) outs(%4 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scope.return %5 : tensor<128x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scf.yield %3 : tensor<128x128xf32>
  } {hivm.cv_pipelined_loop}
  return
}
}

// CHECK-LABEL: func.func @no_following_cube_scope
// CHECK:         scope.scope : () -> (tensor<128x128xf32>, tensor<8x8x16x16xbf16>) {
// CHECK-NEXT:      tensor.empty() : tensor<128x128xf32>
