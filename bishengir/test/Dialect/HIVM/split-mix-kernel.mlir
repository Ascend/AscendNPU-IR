// RUN: bishengir-opt %s -hivm-split-mix-kernel -split-input-file -verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: add(
  func.func private @add(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, 
                         %arg2: tensor<64x64xf16> {hacc.arg_type = #hacc.arg_type<output>}) -> tensor<64x64xf16> 
  attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}

  // CHECK-LABEL: mul_add_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: mul_add_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // CHECK: annotation.mark
  func.func @mul_add(%arg0: tensor<64x64xf16>,
                      %arg1: tensor<64x64xf16>,
                      %arg2: tensor<64x64xf16>,
                      %arg3: tensor<64x64xf16>,
                      %arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
                      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %0 = func.call @add(%arg0, %arg1, %arg2) : (tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    %1 = hivm.hir.matmul ins(%0, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

module {
  // CHECK-LABEL: mul_add_with_collapse_shape_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK: %[[T0:.*]] = hivm.hir.matmul
  // CHECK: annotation.mark %[[T0:.*]]
  // CHECK-LABEL: mul_add_with_collapse_shape_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @mul_add_with_collapse_shape(%arg0: tensor<64x64xf16>,
                      %arg1: tensor<64x64xf16>,
                      %arg2: tensor<64x64xf16>,
                      %arg3: tensor<4096xf16>,
                      %arg4: tensor<4096xf16>) -> tensor<4096xf16>
                      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.matmul ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg2: tensor<64x64xf16>) -> tensor<64x64xf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<64x64xf16> into tensor<4096xf16>
    %2 = hivm.hir.vadd ins(%collapsed, %arg3 : tensor<4096xf16>, tensor<4096xf16>) outs(%arg4 : tensor<4096xf16>) -> tensor<4096xf16>
    return %2 : tensor<4096xf16>
  }
}

// -----

module {
  // CHECK-LABEL: mixed_matmul_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: mixed_matmul_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @mixed_matmul(%arg0: tensor<64x64xf16>,
                      %arg1: tensor<64x64xf16>,
                      %arg2: tensor<64x64xf16>,
                      %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
                      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.mix_matmul ins(%arg0, %arg2 : tensor<64x64xf16>, tensor<64x64xf16>)
                         post_vector_func_ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
                         outs(%arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func private @mixed_matmul
  // CHECK-SAME: attributes
  // CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
  // CHECK-SAME: hacc.mix_entry
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>

  // CHECK-LABEL: mixed_matmul_mix_aic({{.*}} hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: mixed_matmul_mix_aiv({{.*}} hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @mixed_matmul(%arg0: tensor<64x64xf16>,
                          %arg1: tensor<64x64xf16>,
                          %arg2: tensor<64x64xf16>,
                          %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.mix_matmul ins(%arg0, %arg2 : tensor<64x64xf16>, tensor<64x64xf16>)
                         post_vector_func_ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
                         outs(%arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }

  func.func @host_caller(%arg0: tensor<64x64xf16>,
                         %arg1: tensor<64x64xf16>,
                         %arg2: tensor<64x64xf16>,
                         %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %1 = func.call @mixed_matmul(%arg0, %arg1, %arg2, %arg3) : (tensor<64x64xf16>,tensor<64x64xf16>,tensor<64x64xf16>,tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

module {
  // expected-error@below {{Currently, MIX kernels can only be called by host functions!}}
  func.func private @mix_function() -> tensor<16xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>}

  func.func @device_caller() -> tensor<16xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = func.call @mix_function() : () -> tensor<16xf16>
    return %1 : tensor<16xf16>
  }
}


// -----

// CHECK-LABEL: @test_callee_arg_with_inconsistent_order_mix_aic({{.*}}: i64, {{.*}}: tensor<128x256xf32>, 
// CHECK-SAME: {{.*}}: tensor<256xf32>, {{.*}}: tensor<768x256xf32>, %[[arg4:.*]]: tensor<128xf32>, %[[arg5:.*]]: tensor<128x1xf32>, 
// CHECK-SAME: {{.*}}: tensor<128x768xf32>, {{.*}}: tensor<128x256xf32>)
// CHECK: return %[[arg4]], %[[arg5]], {{.*}} : tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>
module {
  func.func @callee_arg_with_inconsistent_order(
    %arg0: tensor<128xf32> {hacc.arg_type = #hacc.arg_type<output>}, 
    %arg1: tensor<128x1xf32> {hacc.arg_type = #hacc.arg_type<output>}, 
    %arg2: tensor<128x256xf32>, 
    %arg3: tensor<256xf32>, 
    %arg4: tensor<128x256xf32> {hacc.arg_type = #hacc.arg_type<output>}) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>) attributes {hacc.always_inline, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "", hacc.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      return %arg0, %arg1, %arg4 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>
  }
  func.func @test_callee_arg_with_inconsistent_order(
    %arg0: i64, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<768x256xf32>, %arg4: tensor<128xf32>, 
    %arg5: tensor<128x1xf32>, %arg6: tensor<128x768xf32>, %arg7: tensor<128x256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
      %0:3 = call @callee_arg_with_inconsistent_order(%arg4, %arg5, %arg1, %arg2, %arg7) : 
        (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>, tensor<256xf32>, tensor<128x256xf32>) 
        -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>)
      %1 = hivm.hir.mix_matmul ins(%0#2, %arg3 : tensor<128x256xf32>, tensor<768x256xf32>) outs(%arg6 : tensor<128x768xf32>) b_transpose -> tensor<128x768xf32>
        return %0#0, %0#1, %1 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: _attn_fwd_mix_aic(
  // CHECK: memref.alloc()
  // CHECK: annotation.mark {{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
  // CHECK: bufferization.to_tensor
  // CHECK: hivm.hir.mmadL1
  // CHECK: %[[VAL_1:.*]]  = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[VAL_1]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.fixpipe {enable_nz2nd} ins({{.*}} : tensor<16x16xf32>) outs(%[[VAL_1:.*]] : memref<16x16xf32, #hivm.address_space<ub>>)
  // CHECK-LABEL: _attn_fwd_mix_aiv(
  // CHECK: annotation.mark {{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf16, #hivm.address_space<cbuf>>
  // CHECK: memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK: annotation.mark {{.*}} {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf32, #hivm.address_space<ub>>
  func.func @_attn_fwd(%arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg1: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: f32, %arg5: i32, %arg6: i32, %arg7: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %c16_i32 = arith.constant 16 : i32
    %true = arith.constant true
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg7, %arg6 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg5 : i32
    %7 = arith.muli %6, %c16_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.muli %8, %c16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf16, strided<[16, 1]>>) outs(%alloc : memref<16x16xf16>)
    %alloc_2 = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1], offset: ?>>) outs(%alloc_2 : memref<16x16xf32>)
    %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x16xf32>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16>
    %12 = tensor.empty() : tensor<16x16xf16>
    %13 = hivm.hir.vcast ins(%10 : tensor<16x16xf32>) outs(%12 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf16, #hivm.address_space<cbuf>>
    %14 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x16xf16, #hivm.address_space<cbuf>>
    %15 = hivm.hir.copy ins(%13 : tensor<16x16xf16>) outs(%14 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %16 = tensor.empty() : tensor<16x16xf32>
    %17 = hivm.hir.mmadL1 ins(%15, %11, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%16 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %alloc_3 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.fixpipe {enable_nz2nd} ins(%17 : tensor<16x16xf32>) outs(%alloc_3 : memref<16x16xf32, #hivm.address_space<ub>>)
    %18 = bufferization.to_tensor %alloc_3 restrict writable : memref<16x16xf32, #hivm.address_space<ub>>
    %19 = tensor.empty() : tensor<16x16xf16>
    %20 = hivm.hir.vcast ins(%18 : tensor<16x16xf32>) outs(%19 : tensor<16x16xf16>) -> tensor<16x16xf16>
    hivm.hir.store ins(%20 : tensor<16x16xf16>) outs(%reinterpret_cast_1 : memref<16x16xf16, strided<[16, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: scope_vector_result_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.hir.vadd
// CHECK: hivm.hir.matmul
// CHECK-LABEL: scope_vector_result_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// CHECK: scope.scope
// CHECK: hivm.hir.vadd
// CHECK-NOT: hivm.hir.matmul

module {
  func.func @scope_vector_result(
      %arg0: tensor<64x64xf16>,
      %arg1: tensor<64x64xf16>,
      %arg2: tensor<64x64xf16>,
      %arg3: tensor<64x64xf16>,
      %arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // VECTOR-type ScopeOp with one tensor result.
    // The scope result is consumed by the CUBE matmul below.
    %0 = scope.scope : () -> tensor<64x64xf16> {
      %vadd = hivm.hir.vadd
          ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
          outs(%arg2 : tensor<64x64xf16>) -> tensor<64x64xf16>
      scope.return %vadd : tensor<64x64xf16>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    %1 = hivm.hir.matmul
        ins(%0, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%arg4 : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

// CHECK-LABEL: scope_cube_internal_outs_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// CHECK: scope.scope
// CHECK: hivm.hir.matmul
// CHECK-NOT: hivm.hir.vadd
// CHECK-LABEL: scope_cube_internal_outs_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// CHECK: tensor.empty() : tensor<64x64xf16>
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.hir.matmul
// CHECK: hivm.hir.vadd

module {
func.func @scope_cube_internal_outs(
    %arg0: tensor<64x64xf16>,
    %arg1: tensor<64x64xf16>,
    %arg2: tensor<64x64xf16>,
    %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CUBE-type ScopeOp with result.
  // The outs operand of matmul is tensor.empty() created *inside* the scope,
  // so isDefinedOutside will return false → stub is created.
  %0 = scope.scope : () -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %mm = hivm.hir.matmul
        ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    scope.return %mm : tensor<64x64xf16>
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  %1 = hivm.hir.vadd
      ins(%0, %arg2 : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%arg3 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %1 : tensor<64x64xf16>
}
}

// -----

// CHECK-LABEL: scope_cube_external_outs_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// CHECK: scope.scope
// CHECK: hivm.hir.matmul
// CHECK-NOT: hivm.hir.vadd
// CHECK-LABEL: scope_cube_external_outs_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.hir.matmul
// CHECK: hivm.hir.vadd

module {
func.func @scope_cube_external_outs(
    %arg0: tensor<64x64xf16>,
    %arg1: tensor<64x64xf16>,
    %arg2: tensor<64x64xf16>,
    %arg3: tensor<64x64xf16>,
    %arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CUBE-type ScopeOp with result.
  // The outs operand of matmul is %arg2, a function argument defined
  // *outside* the scope, so isDefinedOutside returns true and the scope
  // result is replaced directly by %arg2 without creating a stub.
  %0 = scope.scope : () -> tensor<64x64xf16> {
    %mm = hivm.hir.matmul
        ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%arg2 : tensor<64x64xf16>) -> tensor<64x64xf16>
    scope.return %mm : tensor<64x64xf16>
  } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
  %1 = hivm.hir.vadd
      ins(%0, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%arg4 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %1 : tensor<64x64xf16>
}
}

// -----

// CHECK-LABEL: scope_loop_core_type_override_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// scope is CUBE (via loop_core_type), so it is RETAINED in the AIC kernel.
// The VECTOR vadd is erased; its result is replaced by its outs (%arg4).
// CHECK: scope.scope
// CHECK: hivm.hir.matmul
// CHECK-NOT: hivm.hir.vadd
// CHECK: return %arg4 : tensor<64x64xf16>
// CHECK-LABEL: scope_loop_core_type_override_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// scope is CUBE (via loop_core_type), so it is ERASED from the AIV kernel.
// The scope result is replaced by its matmul's outs (%arg2).
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.hir.matmul
// CHECK: hivm.hir.vadd
// CHECK: ins(%arg2, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>)

module {
func.func @scope_loop_core_type_override(
    %arg0: tensor<64x64xf16>,
    %arg1: tensor<64x64xf16>,
    %arg2: tensor<64x64xf16>,
    %arg3: tensor<64x64xf16>,
    %arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // ScopeOp has no hivm.tcore_type. loop_core_type should drive decision.
  // This exercises SplitMixKernel.cpp:482-486 where loop_core_type
  // overrides the scope op core type decision.
  %0 = scope.scope : () -> tensor<64x64xf16> {
    %mm = hivm.hir.matmul
        ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%arg2 : tensor<64x64xf16>) -> tensor<64x64xf16>
    scope.return %mm : tensor<64x64xf16>
  } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}

  %1 = hivm.hir.vadd
      ins(%0, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%arg4 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %1 : tensor<64x64xf16>
}
}

// -----

// CHECK-LABEL: scope_loop_core_type_vector_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// CHECK-NOT: scope.scope
// CHECK-NOT: hivm.hir.vadd
// CHECK: hivm.hir.matmul
// CHECK-LABEL: scope_loop_core_type_vector_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// CHECK: scope.scope
// CHECK: hivm.hir.vadd
// CHECK-NOT: hivm.hir.matmul

module {
func.func @scope_loop_core_type_vector(
    %arg0: tensor<64x64xf16>,
    %arg1: tensor<64x64xf16>,
    %arg2: tensor<64x64xf16>,
    %arg3: tensor<64x64xf16>,
    %arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // VECTOR ScopeOp with result, identified via hivm.loop_core_type (no tcore_type).
  // This exercises the kPipelinedLoopCoreTypeAttrName override path for VECTOR.
  %0 = scope.scope : () -> tensor<64x64xf16> {
    %add = hivm.hir.vadd
        ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%arg2 : tensor<64x64xf16>) -> tensor<64x64xf16>
    scope.return %add : tensor<64x64xf16>
  } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  %1 = hivm.hir.matmul
      ins(%0, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%arg4 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %1 : tensor<64x64xf16>
}
}

// -----

// CHECK-LABEL: test_extract_i1_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// CHECK: hivm.hir.mmadL1
// CHECK-LABEL: test_extract_i1_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// CHECK: hivm.hir.vcmp
// CHECK-NOT: hivm.hir.mmadL1
module {
  func.func @test_extract_i1(%arg0: memref<1xi16>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) -> tensor<16x16xf32> attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c0_i16 = arith.constant 0 : i16
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<1xi1>
    annotation.mark %0 {logical_block_num} : tensor<1xi1>
    %1 = bufferization.to_tensor %arg0 restrict writable : memref<1xi16>
    %2 = hivm.hir.vcmp ins(%1, %c0_i16 : tensor<1xi16>, i16) outs(%0 : tensor<1xi1>) compare_mode = <ne> -> tensor<1xi1>
    %extracted = tensor.extract %2[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xi1>
    %3 = tensor.empty() : tensor<1xi8>
    %4 = hivm.hir.vcast ins(%2 : tensor<1xi1>) outs(%3 : tensor<1xi8>) -> tensor<1xi8>
    %5 = memref_ext.alloc_workspace() : memref<1xi8>
    %6 = bufferization.to_tensor %5 restrict writable : memref<1xi8>
    %7 = hivm.hir.store ins(%4 : tensor<1xi8>) outs(%6 : tensor<1xi8>) -> tensor<1xi8>
    annotation.mark %7 {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : tensor<1xi8>
    %extracted_0 = tensor.extract %7[%c0] {"DuplicateTensorExtractForCube::newExtractLabel" = 1 : i32, "DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xi8>
    %8 = arith.trunci %extracted_0 : i8 to i1
    annotation.mark %extracted {"DuplicateTensorExtractForCube::replacementLabel" = 1 : i32} keys = [] values = [%8 : i1] : i1
    %9 = bufferization.to_tensor %arg1 restrict writable : memref<16x16xf32>
    %10 = bufferization.to_tensor %arg2 restrict writable : memref<16x16xf32>
    %11 = tensor.empty() : tensor<16x16xf32>
    %12 = arith.extui %extracted : i1 to i32
    %13 = arith.cmpi sgt, %12, %c0_i32 : i32
    %14 = scf.if %13 -> (tensor<16x16xf32>) {
      %15 = hivm.hir.mmadL1 ins(%9, %10, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %15 : tensor<16x16xf32>
    } else {
      scf.yield %9 : tensor<16x16xf32>
    }
    return %14 : tensor<16x16xf32>
  }
}

// -----

// CHECK-LABEL: gather_and_matmul_mix_aic(
// CHECK: hivm.hir.matmul
// CHECK-NOT: hivm.hir.gather_load
// CHECK-LABEL: gather_and_matmul_mix_aiv(
// CHECK: hivm.hir.gather_load
// CHECK-NOT: hivm.hir.matmul
module {
  func.func @gather_and_matmul(%base : memref<?xf32>,
                               %indices : tensor<8xi64>,
                               %gather_dst : tensor<8xf32>,
                               %lhs : tensor<8x8xf16>,
                               %rhs : tensor<8x8xf16>,
                               %matmul_dst : tensor<8x8xf16>)
      -> (tensor<8xf32>, tensor<8x8xf16>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%gather_dst : tensor<8xf32>) -> tensor<8xf32>
    %1 = hivm.hir.matmul
         ins(%lhs, %rhs : tensor<8x8xf16>, tensor<8x8xf16>)
         outs(%matmul_dst : tensor<8x8xf16>) -> tensor<8x8xf16>
    return %0, %1 : tensor<8xf32>, tensor<8x8xf16>
  }
}

// -----

// CHECK-LABEL: scatter_and_matmul_mix_aic(
// CHECK: hivm.hir.matmul
// CHECK-NOT: hivm.hir.scatter_store
// CHECK-LABEL: scatter_and_matmul_mix_aiv(
// CHECK: hivm.hir.scatter_store
// CHECK-NOT: hivm.hir.matmul
module {
  func.func @scatter_and_matmul(%base : memref<?xf32>,
                                %indices : tensor<8xi64>,
                                %data : tensor<8xf32>,
                                %lhs : tensor<8x8xf16>,
                                %rhs : tensor<8x8xf16>,
                                %matmul_dst : tensor<8x8xf16>)
      -> tensor<8x8xf16>
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.scatter_store ins(%indices : tensor<8xi64>,
                               %data : tensor<8xf32>,
                               %c1_i32 : i32)
                           outs(%base : memref<?xf32>)
    %0 = hivm.hir.matmul
         ins(%lhs, %rhs : tensor<8x8xf16>, tensor<8x8xf16>)
         outs(%matmul_dst : tensor<8x8xf16>) -> tensor<8x8xf16>
    return %0 : tensor<8x8xf16>
  }
}
