// RUN: bishengir-opt --vf-fusion="fusion-mode=all-op enable-outline-cf=false enable-outline-memref=true enable-outline-arith=true" --split-input-file %s | FileCheck %s
// RUN: bishengir-opt --vf-fusion="fusion-mode=all-op enable-outline-cf=false enable-outline-memref=false enable-outline-arith=false" --split-input-file %s | FileCheck %s --check-prefix=OPTS-OFF

// CHECK-LABEL: func.func private @no_reshape_in_the_middle_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: hfusion.bitcast
// CHECK: tensor.collapse
// CHECK-LABEL: func.func @no_reshape_in_the_middle(
// CHECK: call @no_reshape_in_the_middle_fused_0

// OPTS-OFF-LABEL: func.func private @no_reshape_in_the_middle_fused_0(
// OPTS-OFF: linalg.elemwise_binary
// OPTS-OFF: hfusion.cast
// OPTS-OFF: hfusion.bitcast
// OPTS-OFF: tensor.collapse
// OPTS-OFF-LABEL: func.func @no_reshape_in_the_middle(
// OPTS-OFF: call @no_reshape_in_the_middle_fused_0
module {
  func.func @no_reshape_in_the_middle(%arg0: tensor<3x2xf16>, %arg1: tensor<3x2xf16>, %arg2: f16, %arg3: tensor<3x2xf32>) -> tensor<6xi32> attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = tensor.empty() : tensor<3x2xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg2 : tensor<3x2xf16>, f16) outs(%arg1 : tensor<3x2xf16>) -> tensor<3x2xf16>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<3x2xf16>) outs(%arg3 : tensor<3x2xf32>) -> tensor<3x2xf32>
    %3 = hfusion.bitcast ins(%2 : tensor<3x2xf32>) outs(%0 : tensor<3x2xi32>) -> tensor<3x2xi32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<3x2xi32> into tensor<6xi32>
    %4 = tensor.empty() : tensor<6xi32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%collapsed, %collapsed : tensor<6xi32>, tensor<6xi32>) outs(%4 : tensor<6xi32>) -> tensor<6xi32>
    return %5 : tensor<6xi32>
  }
}

// -----

// CHECK-LABEL: func.func private @outline_memref_fused_0(
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.store
// CHECK-LABEL: func.func @outline_memref(
// CHECK-NOT: hivm.hir.vadd
// CHECK: call @outline_memref_fused_0
// CHECK-NOT: hivm.hir.vadd

// OPTS-OFF-LABEL: func.func @outline_memref(
// OPTS-OFF: hivm.hir.vadd
// OPTS-OFF: hivm.hir.vadd
// OPTS-OFF: hivm.hir.store 
func.func @outline_memref(%valueA: memref<16xf16>,
                          %valueB: memref<16xf16>,
                          %valueC: memref<16xf16>)
                          attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %ubC = memref.alloc() : memref<16xf16>
  %ubD = memref.alloc() : memref<16xf16>
  hivm.hir.vadd ins(%valueA, %valueB: memref<16xf16>, memref<16xf16>) outs(%ubC: memref<16xf16>)
  hivm.hir.vadd ins(%ubC, %ubC: memref<16xf16>, memref<16xf16>) outs(%ubD: memref<16xf16>)
  hivm.hir.store ins(%ubD : memref<16xf16>) outs(%valueC : memref<16xf16>)
  return
}

// -----

// CHECK-LABEL: func.func private @outline_arith_fused_0(
// CHECK: arith.muli
// CHECK: arith.index_cast
// CHECK-LABEL: func.func @outline_arith(
// CHECK: arith.constant
// CHECK-NOT: arith.muli
// CHECK: call @outline_arith_fused_0
// CHECK-NOT: arith.muli

// OPTS-OFF-LABEL: func.func @outline_arith(
// OPTS-OFF: arith.constant
// OPTS-OFF: arith.muli
// OPTS-OFF: arith.index_cast 
module {
  func.func @outline_arith(%arg6: i32) -> index attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.muli %arg6, %c256_i32 : i32
    %2 = arith.index_cast %0 : i32 to index
    return %2 : index
  }
}

// -----

// CHECK-LABEL: func.func private @nested_regions_fused_0(
// CHECK: bufferization.to_tensor
// CHECK: linalg.matmul
// CHECK: hivm.hir.fixpipe
// CHECK-LABEL: func.func @nested_regions(
// CHECK: scf.for
// CHECK: memref.copy
// CHECK: scope.scope
// CHECK: func.call @nested_regions_fused_0
// CHECK-NEXT: scope.return 

// OPTS-OFF-LABEL: func.func @nested_regions(
// OPTS-OFF: scf.for
// OPTS-OFF: memref.copy
// OPTS-OFF: scope.scope
// OPTS-OFF: bufferization.to_tensor
// OPTS-OFF: linalg.matmul
// OPTS-OFF: hivm.hir.fixpipe
// OPTS-OFF-NEXT: scope.return 
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
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

// CHECK-LABEL: func.func private @nested_for_fused_0(
// CHECK: linalg.elemwise_binary
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: hfusion.bitcast
// CHECK: tensor.collapse_shape
// CHECK-LABEL: func.func @nested_for(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: func.call @nested_for_fused_0
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: return 

// OPTS-OFF-LABEL: func.func private @nested_for_fused_0(
// OPTS-OFF: linalg.elemwise_binary
// OPTS-OFF: linalg.elemwise_binary
// OPTS-OFF: hfusion.cast
// OPTS-OFF: hfusion.bitcast
// OPTS-OFF: tensor.collapse_shape
// OPTS-OFF-LABEL: func.func @nested_for(
// OPTS-OFF: scf.for
// OPTS-OFF: scf.for
// OPTS-OFF: func.call @nested_for_fused_0
// OPTS-OFF: scf.yield
// OPTS-OFF: scf.yield
// OPTS-OFF: return 
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

// CHECK-LABEL: func.func private @nested_for_with_anchor_outside_fused_0(
// CHECK: linalg.elemwise_unary
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: hfusion.bitcast
// CHECK: tensor.collapse_shape
// CHECK-LABEL: func.func @nested_for_with_anchor_outside(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: func.call @nested_for_with_anchor_outside_fused_0
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: return 

// OPTS-OFF-LABEL: func.func private @nested_for_with_anchor_outside_fused_0(
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: linalg.elemwise_binary
// OPTS-OFF: hfusion.cast
// OPTS-OFF: hfusion.bitcast
// OPTS-OFF: tensor.collapse_shape
// OPTS-OFF-LABEL: func.func @nested_for_with_anchor_outside(
// OPTS-OFF: scf.for
// OPTS-OFF: scf.for
// OPTS-OFF: func.call @nested_for_with_anchor_outside_fused_0
// OPTS-OFF: scf.yield
// OPTS-OFF: scf.yield
// OPTS-OFF: return 
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

// CHECK-LABEL: func.func @nested_for_with_outside_use(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.elemwise_unary
// CHECK: linalg.elemwise_unary
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: return 

// OPTS-OFF-LABEL: func.func @nested_for_with_outside_use(
// OPTS-OFF: scf.for
// OPTS-OFF: scf.for
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: scf.yield
// OPTS-OFF: scf.yield
// OPTS-OFF: return 
module {
  func.func @nested_for_with_outside_use(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: tensor<16x4x4x16xf16>, %arg8: tensor<16x4x4x16xf16>) -> (tensor<16x4x4x16xf16>, tensor<16x4x4x16xf16>) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
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

// CHECK-LABEL: func.func @nested_for_operation_in_between(
// CHECK: scf.for
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: linalg.elemwise_unary
// CHECK: linalg.elemwise_unary
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: return 

// OPTS-OFF-LABEL: func.func @nested_for_operation_in_between(
// OPTS-OFF: scf.for
// OPTS-OFF: linalg.fill
// OPTS-OFF: scf.for
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: scf.yield
// OPTS-OFF: scf.yield
// OPTS-OFF: return 
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

// CHECK-LABEL: func.func @block_arg_anchor(
// CHECK: scf.for
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: linalg.elemwise_unary
// CHECK: linalg.elemwise_unary
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: return 

// OPTS-OFF-LABEL: func.func @block_arg_anchor(
// OPTS-OFF: scf.for
// OPTS-OFF: linalg.fill
// OPTS-OFF: scf.for
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: linalg.elemwise_unary
// OPTS-OFF: scf.yield
// OPTS-OFF: scf.yield
// OPTS-OFF: return 
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
