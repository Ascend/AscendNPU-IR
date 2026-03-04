// RUN: bishengir-opt --vf-fusion="fusion-mode=n-most-op enable-outline-cf=false enable-outline-memref=true enable-outline-arith=true" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @kernel_fused_0
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: vector.constant_mask
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: vector.transfer_write
// CHECK: tensor.insert_slice
// CHECK-LABEL: func.func @kernel
// CHECK: scf.for
// CHECK-NEXT: func.call @kernel_fused_0
// CHECK: scf.yield

// CHECK-LABEL: func.func private @test_not_affect_affinity_fused_0
// CHECK: bufferization.to_tensor
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.mmadL1
// CHECK-LABEL: func.func private @test_not_affect_affinity_fused_1
// CHECK: memref.memory_space_cast
// CHECK: hivm.hir.fixpipe
// CHECK: bufferization.to_tensor

// CHECK-LABEL: func.func private @test_not_affect_affinity_fused_2
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.copy
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.mmadL1
// CHECK-LABEL: func.func @test_not_affect_affinity

// CHECK: call @test_not_affect_affinity_fused_0
// CHECK: call @test_not_affect_affinity_fused_1
// CHECK: call @kernel
// CHECK: call @test_not_affect_affinity_fused_2
module {
  func.func @kernel(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {hivm.vector_function} {
    %cst = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg1) -> (tensor<16x16xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg2, 0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<1x16xf32>
      %1 = vector.constant_mask [1, 16] : vector<1x64xi1>
      %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst_0, %1 {in_bounds = [true, true]} : tensor<1x16xf32>, vector<1x64xf32>
      %3 = arith.mulf %2, %cst : vector<1x64xf32>
      %4 = vector.transfer_write %3, %extracted_slice_1[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x16xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg3[%arg2, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
    }
    return %0 : tensor<16x16xf32>
  }
  func.func @test_not_affect_affinity() -> tensor<16x16xf32> {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<cbuf>>
    %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %memspacecast_1 = memref.memory_space_cast %alloc_0 : memref<16x16xf32, #hivm.address_space<cbuf>> to memref<16x16xf32>
    %alloc_2 = memref.alloc() : memref<16x16xf16>
    %0 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x16xf16>
    %alloc_3 = memref.alloc() : memref<16x16xf16>
    %1 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf16>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = hivm.hir.mmadL1 {b_transpose} ins(%1, %0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<16x16xf32>) outs(%alloc : memref<16x16xf32, #hivm.address_space<ub>>)
    %4 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = call @kernel(%4, %5) {hivm.vector_function} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %7 = bufferization.to_tensor %memspacecast_1 restrict writable : memref<16x16xf32>
    %8 = hivm.hir.copy ins(%6 : tensor<16x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_4 = memref.alloc() : memref<16x16xf16>
    %9 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x16xf16>
    %10 = hivm.hir.mmadL1 {a_transpose} ins(%9, %8, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf32>, i1, index, index, index) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %10 : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: func.func private @tensor_empty_with_dim_fused_0(
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.elemwise_binary

// CHECK-LABEL: func.func @tensor_empty_with_dim(
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: call @tensor_empty_with_dim_fused_0
module {
  func.func @tensor_empty_with_dim(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}

// -----

module {
  func.func @nested_regions(%arg1: memref<128x128xf16, strided<[128, 1], offset: ?>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
    %c32_i32 = arith.constant 32 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<128x128xf16>
    scf.for %arg4 = %arg3 to %c8192_i32 step %c32_i32  : i32 {
      memref.copy %arg1, %alloc_0 : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
      scope.scope : () -> () {
        %2 = bufferization.to_tensor %alloc_0 restrict writable : memref<128x128xf16>
        %3 = linalg.matmul {input_precison = "ieee"} ins(%2, %2 : tensor<128x128xf16>, tensor<128x128xf16>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%3 : tensor<128x128xf32>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        scope.return
      } {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>}
    }
    return
  }
}

// -----

module {
  func.func @limited_n_most_op(%arg0: tensor<16x4x4x16xf16>, %arg1: tensor<16x4x4x16xf16>, %arg2: f16, %arg3: tensor<16x4x4x16xf32>) -> tensor<64x64xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<16x4x4x16xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg2 : tensor<16x4x4x16xf16>, f16) outs(%arg1 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<16x4x4x16xf16>) outs(%arg3 : tensor<16x4x4x16xf32>) -> tensor<16x4x4x16xf32>
    %3 = hfusion.bitcast ins(%2 : tensor<16x4x4x16xf32>) outs(%0 : tensor<16x4x4x16xi32>) -> tensor<16x4x4x16xi32>
    %collapsed = tensor.collapse_shape %3 [[0, 1], [2, 3]] : tensor<16x4x4x16xi32> into tensor<64x64xi32>
    return %collapsed : tensor<64x64xi32>
  }
}

// -----

module {
  func.func @nested_for(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<16x4x4x16xf16>, %arg8: f16) -> tensor<64x64xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = tensor.empty() : tensor<16x4x4x16xf32>
    %2 = linalg.fill ins(%arg0 : i32) outs(%0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %3 = tensor.empty() : tensor<64x64xi32>
    %alloc = memref.alloc() : memref<16x4x4x16xf16>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<16x4x4x16xf16>
    %5 = tensor.empty() : tensor<16x4x4x16xi32>
    %6 = scf.for %arg9 = %arg1 to %arg2 step %arg3 iter_args(%arg10 = %3) -> (tensor<64x64xi32>)  : i32 {
      %7 = scf.for %arg11 = %arg4 to %arg5 step %arg6 iter_args(%arg12 = %2) -> (tensor<64x64xi32>)  : i32 {
        %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %4 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %arg8 : tensor<16x4x4x16xf16>, f16) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<16x4x4x16xf16>) outs(%1 : tensor<16x4x4x16xf32>) -> tensor<16x4x4x16xf32>
        %11 = hfusion.bitcast ins(%10 : tensor<16x4x4x16xf32>) outs(%5 : tensor<16x4x4x16xi32>) -> tensor<16x4x4x16xi32>
        %collapsed = tensor.collapse_shape %11 [[0, 1], [2, 3]] : tensor<16x4x4x16xi32> into tensor<64x64xi32>
        scf.yield %collapsed : tensor<64x64xi32>
      }
      scf.yield %7 : tensor<64x64xi32>
    }
    return %6 : tensor<64x64xi32>
  }
}

// -----

module {
  func.func @nested_for_with_anchor_outside(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<16x4x4x16xf16>, %arg8: f16) -> tensor<64x64xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = tensor.empty() : tensor<16x4x4x16xf32>
    %2 = tensor.empty() : tensor<64x64xi32>
    %3 = tensor.empty() : tensor<16x4x4x16xi32>
    %alloc = memref.alloc() : memref<16x4x4x16xf16>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<16x4x4x16xf16>
    %5 = linalg.fill ins(%arg0 : i32) outs(%0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %6 = scf.for %arg9 = %arg1 to %arg2 step %arg3 iter_args(%arg10 = %2) -> (tensor<64x64xi32>)  : i32 {
      %7 = scf.for %arg11 = %arg4 to %arg5 step %arg6 iter_args(%arg12 = %5) -> (tensor<64x64xi32>)  : i32 {
        %8 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%4 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %arg8 : tensor<16x4x4x16xf16>, f16) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<16x4x4x16xf16>) outs(%1 : tensor<16x4x4x16xf32>) -> tensor<16x4x4x16xf32>
        %11 = hfusion.bitcast ins(%10 : tensor<16x4x4x16xf32>) outs(%3 : tensor<16x4x4x16xi32>) -> tensor<16x4x4x16xi32>
        %collapsed = tensor.collapse_shape %11 [[0, 1], [2, 3]] : tensor<16x4x4x16xi32> into tensor<64x64xi32>
        scf.yield %collapsed : tensor<64x64xi32>
      }
      scf.yield %7 : tensor<64x64xi32>
    }
    return %6 : tensor<64x64xi32>
  }
}

// -----

module {
  func.func @nested_for_operation_in_between(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<16x4x4x16xf16>, %arg8: tensor<16x4x4x16xf16>) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<16x4x4x16xf16>
    %1 = tensor.empty() : tensor<16x4x4x16xf16>
    %alloc = memref.alloc() : memref<16x4x4x16xf16>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x4x4x16xf16>
    %3 = linalg.fill ins(%arg0 : i32) outs(%0 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
    %4:2 = scf.for %arg9 = %arg1 to %arg2 step %arg3 iter_args(%arg10 = %1, %arg11 = %0) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>)  : i32 {
      %5:2 = scf.for %arg12 = %arg4 to %arg5 step %arg6 iter_args(%arg13 = %3, %arg14 = %1) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>)  : i32 {
        %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%2 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %7 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg8 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        scf.yield %7, %6 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
      }
      scf.yield %5#1, %5#0 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
    }
    return %4#0, %4#1 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
  }
}

// -----

module {
  func.func @nested_for_operation_in_between(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<16x4x4x16xf16>, %arg8: tensor<16x4x4x16xf16>) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<16x4x4x16xf16>
    %1 = tensor.empty() : tensor<16x4x4x16xf16>
    %2:2 = scf.for %arg9 = %arg1 to %arg2 step %arg3 iter_args(%arg10 = %1, %arg11 = %0) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>)  : i32 {
      %3 = linalg.fill ins(%arg0 : i32) outs(%0 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
      %4:2 = scf.for %arg12 = %arg4 to %arg5 step %arg6 iter_args(%arg13 = %3, %arg14 = %1) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>)  : i32 {
        %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        scf.yield %6, %5 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
      }
      scf.yield %4#1, %4#0 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
    }
    return %2#0, %2#1 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
  }
}

// -----

module {
  func.func @block_arg_anchor(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<16x4x4x16xf16>) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<16x4x4x16xf16>
    %1 = tensor.empty() : tensor<16x4x4x16xf16>
    %2:2 = scf.for %arg8 = %arg1 to %arg2 step %arg3 iter_args(%arg9 = %1, %arg10 = %0) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>)  : i32 {
      %3 = linalg.fill ins(%arg0 : i32) outs(%arg10 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
      %4:2 = scf.for %arg11 = %arg4 to %arg5 step %arg6 iter_args(%arg12 = %3, %arg13 = %1) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>)  : i32 {
        %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg12 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%arg12 : tensor<16x4x4x16xf16>) outs(%arg7 : tensor<16x4x4x16xf16>) -> tensor<16x4x4x16xf16>
        scf.yield %5, %6 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
      }
      scf.yield %4#0, %4#1 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
    }
    return %2#0, %2#1 : tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>
  }
}

// -----

// CHECK-LABEL: func.func private @outline_cf_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary
// CHECK-LABEL: func.func private @outline_cf_fused_1(
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary
// CHECK-LABEL: func.func @outline_cf(
// CHECK: scf.for
// CHECK: scf.if
// CHECK-LABEL: func.call @outline_cf_fused_0(
// CHECK: scf.for
// CHECK-LABEL: func.call @outline_cf_fused_1(
module {
  func.func @outline_cf(%arg0: tensor<64x64xi32>, %arg1: tensor<16x4x4x16xf32>, %arg2: i32, %arg3: tensor<64x64xi32>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: f16) -> tensor<64x64xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = linalg.fill ins(%arg9 : i32) outs(%arg0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %1 = tensor.empty() : tensor<64x64xi32>
    %2 = scf.for %arg11 = %arg2 to %arg7 step %arg8 iter_args(%arg12 = %1) -> (tensor<64x64xi32>)  : i32 {
      %3 = arith.cmpi slt, %arg11, %arg6 : i32
      %4 = scf.if %3 -> (tensor<64x64xi32>) {
        %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %0 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%arg3 : tensor<64x64xi32>) -> tensor<64x64xi32>
        %6 = tensor.empty() : tensor<64x64xi32>
        %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %5 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%6 : tensor<64x64xi32>) -> tensor<64x64xi32>
        scf.yield %7 : tensor<64x64xi32>
      } else {
        %5 = scf.for %arg13 = %arg4 to %arg6 step %arg5 iter_args(%arg14 = %0) -> (tensor<64x64xi32>)  : i32 {
          %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg14, %0 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%arg3 : tensor<64x64xi32>) -> tensor<64x64xi32>
          %7 = tensor.empty() : tensor<64x64xi32>
          %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %6 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%7 : tensor<64x64xi32>) -> tensor<64x64xi32>
          scf.yield %8 : tensor<64x64xi32>
        }
        scf.yield %5 : tensor<64x64xi32>
      }
      scf.yield %4 : tensor<64x64xi32>
    }
    return %2 : tensor<64x64xi32>
  }
}
