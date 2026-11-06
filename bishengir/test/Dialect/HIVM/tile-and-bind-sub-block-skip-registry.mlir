// RUN: bishengir-opt %s -hivm-bind-sub-block -split-input-file | FileCheck %s

// Temporary name-based CV1:2 skip: SWA / GQA paged-decode kernels stay on
// sub-block 0 instead of mix-AIV tiling.

// CHECK-LABEL:   func.func @_swa_paged_decode_kernel_mix_aiv(
// CHECK:           hivm.hir.vln
// CHECK:           hivm.hir.get_sub_block_idx
// CHECK:           scf.if
// CHECK:             hivm.hir.store
// CHECK:           } {limit_sub_block_id0}
// CHECK-NOT:       #hivm.sub_block<x>
#map = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @_swa_paged_decode_kernel_mix_aiv(%arg0: tensor<512xf32>, %arg1: memref<512xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = tensor.empty() : tensor<64xf32>
    %1 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %arg0) -> (tensor<512xf32>) {
      %2 = affine.apply #map()[%arg2]
      %extracted_slice = tensor.extract_slice %arg3[%2] [64] [1] : tensor<512xf32> to tensor<64xf32>
      %3 = hivm.hir.vln ins(%extracted_slice : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %subview = memref.subview %arg1[%2] [64] [1] : memref<512xf32> to memref<64xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%3 : tensor<64xf32>) outs(%subview : memref<64xf32, strided<[1], offset: ?>>)
      scf.yield %arg0 : tensor<512xf32>
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @paged_decode_kernel_mix_aiv(
// CHECK:           } {limit_sub_block_id0}
// CHECK-NOT:       #hivm.sub_block<x>
#map = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func @paged_decode_kernel_mix_aiv(%arg0: tensor<512xf32>, %arg1: memref<512xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = tensor.empty() : tensor<64xf32>
    %1 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %arg0) -> (tensor<512xf32>) {
      %2 = affine.apply #map()[%arg2]
      %extracted_slice = tensor.extract_slice %arg3[%2] [64] [1] : tensor<512xf32> to tensor<64xf32>
      %3 = hivm.hir.vln ins(%extracted_slice : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %subview = memref.subview %arg1[%2] [64] [1] : memref<512xf32> to memref<64xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%3 : tensor<64xf32>) outs(%subview : memref<64xf32, strided<[1], offset: ?>>)
      scf.yield %arg0 : tensor<512xf32>
    }
    return
  }
}
