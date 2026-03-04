// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-mem-scope -split-input-file | FileCheck %s

// CHECK-LABEL: test_infer_mem_scope_basic
module {
  func.func @test_infer_mem_scope_basic(%arg0: i32, %arg1: i32, %arg2: i32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c128 = arith.constant 128 : index
    // CHECK: #hivm.address_space<cc>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    %0 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %alloc) -> (memref<128x128xf32>)  : i32 {
      // CHECK: #hivm.address_space<cbuf>
      %alloc_0 = memref.alloc() : memref<128x128xf16>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_1 = memref.alloc() : memref<128x128xf16>
      %1 = arith.cmpi eq, %arg3, %arg1 : i32
      hivm.hir.mmadL1 ins(%alloc_0, %alloc_1, %1, %c128, %c128, %c128 : memref<128x128xf16>, memref<128x128xf16>, i1, index, index, index) outs(%arg4 : memref<128x128xf32>)
      scf.yield %arg4 : memref<128x128xf32>
    }
    return
  }
}

// -----

// CHECK: test_infer_mem_scope_complicated(
// CHECK: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32
// CHECK-SAME: %[[A:.*]]: memref<*xf16, #hivm.address_space<gm>>, %[[B:.*]]: memref<*xf16, #hivm.address_space<gm>>
// CHECK-SAME: %[[C:.*]]: memref<*xf32, #hivm.address_space<gm>>
// CHECK-SAME: %[[M:.*]]: index, %[[N:.*]]: index, %[[K:.*]]: index
module {
  func.func @test_infer_mem_scope_complicated(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<*xf16>, %arg4: memref<*xf16>, %arg5: memref<*xf32>, %arg6: index, %arg7: index, %arg8: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    // CHECK: #hivm.address_space<cc>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%c0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[?, ?], offset: ?>>
    %alloc_0 = memref.alloc() : memref<128x128xf16>
    // CHECK: (memref<128x128xf16, #hivm.address_space<ub>>) -> ()
    "some_op"(%alloc_0) : (memref<128x128xf16>) -> ()
    // CHECK: scf.for
    // CHECK-SAME: -> (memref<128x128xf32, #hivm.address_space<cc>>)
    %0 = scf.for %arg9 = %arg0 to %arg1 step %arg2 iter_args(%arg10 = %alloc) -> (memref<128x128xf32>)  : i32 {
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%c0], sizes: [128, 128], strides: [1, 1] : memref<*xf16> to memref<128x128xf16, strided<[?, ?], offset: ?>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [128, 128], strides: [1, 1] : memref<*xf16> to memref<128x128xf16, strided<[?, ?], offset: ?>>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_3 = memref.alloc() : memref<128x128xf16>
      %subview = memref.subview %reinterpret_cast_1[0, 0] [%arg6, %arg7] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_4 = memref.subview %alloc_3[0, 0] [%arg6, %arg7] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview, %subview_4 : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      // CHECK: #hivm.address_space<cbuf>
      %alloc_5 = memref.alloc() : memref<128x128xf16>
      %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%arg7, %arg8] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%arg7, %arg8] [1, 1] : memref<128x128xf16> to memref<?x?xf16, strided<[128, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[128, 1]>>
      %1 = arith.cmpi eq, %arg9, %arg1 : i32
      hivm.hir.mmadL1 ins(%alloc_3, %alloc_5, %1, %arg6, %arg7, %arg8 : memref<128x128xf16>, memref<128x128xf16>, i1, index, index, index) outs(%arg10 : memref<128x128xf32>)
      scf.yield %arg10 : memref<128x128xf32>
    }
    hivm.hir.fixpipe {enable_nz2nd} ins(%0 : memref<128x128xf32>) outs(%reinterpret_cast : memref<128x128xf32, strided<[?, ?], offset: ?>>)
    return
  }
}

// -----
module {
  // CHECK: #hivm.address_space<gm>
  func.func @device_func_0(%arg0: memref<1048576xf32>, %arg1: memref<1048576xf32>, %arg2: memref<1048576xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    return
  }
  // CHECK: #hivm.address_space<gm>
  func.func @device_func(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>, %arg3: memref<1024x1024xf32>, %arg4: memref<1024x1024xf32>)
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    "some_operation"(%arg0, %arg1, %arg2) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()
    %collapse_shape_0 = memref.collapse_shape %arg2 [[0, 1]] : memref<1024x1024xf32> into memref<1048576xf32>
    %collapse_shape_1 = memref.collapse_shape %arg3 [[0, 1]] : memref<1024x1024xf32> into memref<1048576xf32>
    %collapse_shape_2 = memref.collapse_shape %arg4 [[0, 1]] : memref<1024x1024xf32> into memref<1048576xf32>
    call @device_func_0(%collapse_shape_0, %collapse_shape_1, %collapse_shape_2) : (memref<1048576xf32>, memref<1048576xf32>, memref<1048576xf32>) -> ()
    return
  }
}

// -----

// CHECK: func.func private @extern_device_func(
// CHECK-SAME: #hivm.address_space<gm>
// CHECK-SAME: #hivm.address_space<gm>
// CHECK-SAME: #hivm.address_space<gm>
// CHECK-SAME: ->  memref<?xf32, #hivm.address_space<gm>>
func.func private @extern_device_func(memref<?xf32>, memref<?xf32>, memref<?xf32>) -> (memref<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>}

// -----

module {
  func.func private @fused_kernel_0(i64, memref<16xf32>, memref<16xf32>) -> memref<16xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>}
  func.func @fused_kernel_1(%arg0: i64, %arg1: memref<16xf32>, %arg2: memref<16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: #hivm.address_space<ub>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    return
  }
  func.func @main(%arg0: i64) -> tensor<16xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    // CHECK: #hivm.address_space<gm>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    // CHECK: #hivm.address_space<gm>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    %0 = call @fused_kernel_0(%arg0, %alloc, %alloc_0) : (i64, memref<16xf32>, memref<16xf32>) -> memref<16xf32>
    %1 = bufferization.to_tensor %0 : memref<16xf32>
    call @fused_kernel_1(%arg0, %alloc, %alloc_0) : (i64, memref<16xf32>, memref<16xf32>) -> ()
    return %1 : tensor<16xf32>
  }
}

// -----

// CHECK: func.func private @extern_host_func(
// CHECK-NOT: #hivm.address_space<gm>
func.func private @extern_host_func(memref<?xf32>, memref<?xf32>, memref<?xf32>) -> (memref<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>}

// -----

// CHECK: func.func @test_scf_if_0
// CHECK: scf.if
// CHECK-SAME: hivm.address_space<ub>>
func.func @test_scf_if_0(%arg0: memref<19xf32>, %arg1: memref<17xf32>, %arg2: index, %arg3: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 5.000000e+00 : f32
  %0 = arith.cmpi ne, %arg3, %c0 : index
  %subview = memref.subview %arg0[%arg2] [%arg3] [1] : memref<19xf32> to memref<?xf32, strided<[1], offset: ?>>
  %1 = scf.if %0 -> (memref<?xf32, strided<[?], offset: ?>>) {
    %alloc = memref.alloc(%arg3) {alignment = 64 : i64} : memref<?xf32>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<?xf32>)
    %cast = memref.cast %alloc : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
    scf.yield %cast : memref<?xf32, strided<[?], offset: ?>>
  } else {
    %subview_0 = memref.subview %arg1[%arg2] [%arg3] [1] : memref<17xf32> to memref<?xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc(%arg3) {alignment = 64 : i64} : memref<?xf32>
    hivm.hir.load ins(%subview_0 : memref<?xf32, strided<[1], offset: ?>>) outs(%alloc : memref<?xf32>) pad_mode = <PadValue> pad_value = %cst : f32
    %cast = memref.cast %alloc : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
    scf.yield %cast : memref<?xf32, strided<[?], offset: ?>>
  }
  hivm.hir.store ins(%1 : memref<?xf32, strided<[?], offset: ?>>) outs(%subview : memref<?xf32, strided<[1], offset: ?>>) atomic = <none>
  return
}

// -----

#map = affine_map<()[s0] -> (s0 + 32)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  // CHECK-LABEL: test_scf_yield
  func.func @test_scf_yield(%arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c96_i32 = arith.constant 96 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<32xf32>)
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
    %cast = memref.cast %reinterpret_cast : memref<32xf32, strided<[1]>> to memref<32xf32, strided<[?], offset: ?>>
    %0:3 = scf.for %arg7 = %c0_i32 to %c96_i32 step %c32_i32 iter_args(%arg8 = %alloc, %arg9 = %cast, %arg10 = %c0) -> (memref<32xf32>, memref<32xf32, strided<[?], offset: ?>>, index)  : i32 {
      %alloc_1 = memref.alloc() : memref<32xf32>
      hivm.hir.load ins(%arg9 : memref<32xf32, strided<[?], offset: ?>>) outs(%alloc_1 : memref<32xf32>)
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
      hivm.hir.vadd ins(%arg8, %alloc_1 : memref<32xf32>, memref<32xf32>) outs(%alloc_2 : memref<32xf32>)
      %1 = affine.apply #map()[%arg10]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
      %cast_4 = memref.cast %reinterpret_cast_3 : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[?], offset: ?>>
      // CHECK: scf.yield
      // CHECK-SAME: memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, index
      scf.yield %alloc_2, %cast_4, %1 : memref<32xf32>, memref<32xf32, strided<[?], offset: ?>>, index
    }
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
    hivm.hir.store ins(%0#0 : memref<32xf32>) outs(%reinterpret_cast_0 : memref<32xf32, strided<[1]>>)
    return
  }
}

// -----

func.func @test_pointer_cast() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c127_i64 = arith.constant 127 : i64
  %c1 = arith.constant 1 : index
  // CHECK: hivm.hir.pointer_cast(%{{.*}}) [%{{.*}}] : memref<?xi8, #hivm.address_space<gm>>
  %0 = hivm.hir.pointer_cast(%c127_i64) [%c1] : memref<?xi8>
  annotation.mark %0 {address_space = #hivm.address_space<gm>} : memref<?xi8>
  %reinterpret_cast = memref.reinterpret_cast %0 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
  return
}
// -----
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @_attn_fwd_outlined_vf_3(%arg0: memref<16x16xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    return
  }
  // CHECK-LABEL: _attn_fwd_mix_aiv
  func.func @_attn_fwd_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: #hivm.address_space<ub>
    %memspacecast = memref.memory_space_cast %alloc : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    // CHECK: hivm.address_space<ub>>
    call @_attn_fwd_outlined_vf_3(%memspacecast) {hivm.vector_function} : (memref<16x16xf32>) -> ()
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @test_mmadL1_tightly_coupled(
  func.func @test_mmadL1_tightly_coupled() {
    // CHECK: #hivm.address_space<cbuf>
    %alloc_7 = memref.alloc() : memref<16x128xf16>
    // CHECK: #hivm.address_space<cc>
    %alloc_3 = memref.alloc() : memref<16x128xf32>
    %33 = arith.constant true
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    // CHECK: #hivm.address_space<cbuf>
    %alloc_8 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_8 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: to memref<16x16xf16, #hivm.address_space<cbuf>>
    %memspacecast = memref.memory_space_cast %alloc_8 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
    // CHECK: memref<16x128xf32, #hivm.address_space<cc>
    hivm.hir.mmadL1 ins(%memspacecast, %alloc_7, %33, %c16, %c16, %c128
        : memref<16x16xf16>, memref<16x128xf16>, i1, index, index, index)
      outs(%alloc_3 : memref<16x128xf32>)
    return
  }
}

// -----
module {
  // CHECK-LABEL: func.func @test_infer_mem_scope_set_cbuf_for_aic_func_unused_alloc(
  func.func @test_infer_mem_scope_set_cbuf_for_aic_func_unused_alloc() attributes{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>}{ 
    // CHECK: #hivm.address_space<cbuf> 
    %alloc =  memref.alloc() : memref<8x16xf32> 
    // CHECK: #hivm.address_space<cbuf> 
    annotation.mark %alloc {MayImplicitTransposeWithLastAxis} : memref<8x16xf32> 
    return 
  } 
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func private @_attn_fwd_mix_aiv_outlined_vf_14(%arg0: memref<4x16x4x16xf16>, %arg1: memref<4x4x16x16xf16>) attributes {hivm.vector_function, no_inline} {
    return
  }
  // CHECK-LABEL: func.func @test_coupled_buffer_aiv(
  func.func @test_coupled_buffer_aiv() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x4x16xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x4x16x16xf16>
    call @_attn_fwd_mix_aiv_outlined_vf_14(%alloc, %alloc_0) {hivm.vector_function, no_inline} : (memref<4x16x4x16xf16>, memref<4x4x16x16xf16>) -> ()
    %alloc_1 = memref.alloc() : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    // CHECK: to memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %memspacecast = memref.memory_space_cast %alloc_1 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16>
    // CHECK: ins({{.*}} : memref<4x4x16x16xf16, #hivm.address_space<ub>>) outs({{.*}} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>)
    hivm.hir.copy ins(%alloc_0 : memref<4x4x16x16xf16>) outs(%memspacecast : memref<4x4x16x16xf16>)
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // CHECK-LABEL: func.func @test_coupled_buffer_aic(
  func.func @test_coupled_buffer_aic() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16>
    %alloc_0 = memref.alloc() : memref<64x128xf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
    hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_already_inserted = true} ins(%alloc, %alloc_0, %true, %c64, %c128, %c64 : memref<64x128xf16>, memref<64x128xf16>, i1, index, index, index) outs(%alloc_1 : memref<64x64xf32>)
    // CHECK: #hivm.address_space<ub>
    %alloc_2 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
    // CHECK: #hivm.address_space<ub>
    annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64x64xf32, #hivm.address_space<ub>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_S>] flag = 0
    // CHECK: ins({{.*}} : memref<64x64xf32, #hivm.address_space<cc>>) outs({{.*}} : memref<64x64xf32, #hivm.address_space<ub>>)
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_1 : memref<64x64xf32>) outs(%alloc_2 : memref<64x64xf32, #hivm.address_space<ub>>)
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 1
    %alloc_3 = memref.alloc() : memref<64x128xf16>
    return
  }
}

// -----

// CHECK-LABEL: test_infer_mem_scope_while
module {
  func.func @test_infer_mem_scope_while(%arg0 : memref<128xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %true = arith.constant true
    %false = arith.constant false
    // CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128xi32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xi32>
    %1 = scf.while(%arg1 = %alloc_0, %arg2 = %false) : (memref<128xi32>, i1) -> memref<128xi32> {
      %2 = arith.xori %arg2, %true : i1
      // CHECK:             %[[VAL_8:.*]] = memref.alloc() : memref<128xi32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<128xi32>
      memref.copy %arg1, %alloc_1 : memref<128xi32> to memref<128xi32>
      scf.condition(%2) %alloc_1 : memref<128xi32>
    } do {
    // CHECK:           ^bb0(%[[VAL_9:.*]]: memref<128xi32, #hivm.address_space<ub>>):
    ^bb0(%arg1: memref<128xi32>):
      scf.yield %arg1, %true : memref<128xi32>, i1
    }
    return
  }
}

// -----

// CHECK-LABEL: test_infer_mem_scope_while
module {
  func.func @test_infer_mem_scope_while(%arg0 : memref<128xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %true = arith.constant true
    %false = arith.constant false
    // CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128xi32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xi32>
    %1:2 = scf.while(%arg1 = %alloc_0, %arg2 = %false) : (memref<128xi32>, i1) -> (memref<128xi32>, i1) {
      %2 = arith.xori %arg2, %true : i1
      // CHECK:             %[[VAL_8:.*]] = memref.alloc() : memref<128xi32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<128xi32>
      memref.copy %arg1, %alloc_1 : memref<128xi32> to memref<128xi32>
      scf.condition(%2) %alloc_1, %arg2 : memref<128xi32>, i1
    } do {
    // CHECK:           ^bb0(%[[VAL_9:.*]]: memref<128xi32, #hivm.address_space<ub>>, %[[VAL_10:.*]]: i1):
    ^bb0(%arg1: memref<128xi32>, %arg2: i1):
      scf.yield %arg1, %arg2 : memref<128xi32>, i1
    }
    return
  }
}

