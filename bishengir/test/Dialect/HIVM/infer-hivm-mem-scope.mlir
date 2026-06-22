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
  func.func @test_infer_mem_scope_complicated(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<*xf16>, %arg4: memref<*xf16>, %arg5: memref<*xf32>, %arg6: index, %arg7: index, %arg8: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    // CHECK: #hivm.address_space<cc>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%c0], sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[?, ?], offset: ?>>
    %alloc_0 = memref.alloc() : memref<128x128xf16>
    // CHECK: (memref<128x128xf16, #hivm.address_space<cbuf>>) -> ()
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
  func.func @fused_kernel_1(%arg0: i64, %arg1: memref<16xf32>, %arg2: memref<16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
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
  func.func @test_scf_yield(%arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
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

func.func @test_arith_select_cast(%arg0: memref<64x32xi1>, %arg1: i32, %arg2: memref<64x32xf16>, %arg3: memref<64x32xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0_i32 = arith.constant 0 : i32
  %true = arith.constant true
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x32xi1>
  hivm.hir.vbrc ins(%true : i1) outs(%alloc : memref<64x32xi1>)
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x32xi1>
  hivm.hir.vnot ins(%arg0 : memref<64x32xi1>) outs(%alloc_0 : memref<64x32xi1>)
  %0 = arith.cmpi sgt, %arg1, %c0_i32 : i32
  %1 = arith.select %0, %alloc_0, %alloc : memref<64x32xi1>
  %alloc_1 = memref.alloc() : memref<64x32xf16>
  hivm.hir.load ins(%arg2 : memref<64x32xf16>) outs(%alloc_1 : memref<64x32xf16>)
  %alloc_2 = memref.alloc() : memref<64x32xf16>
  hivm.hir.load ins(%arg3 : memref<64x32xf16>) outs(%alloc_2 : memref<64x32xf16>)
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<64x32xf16>
  // CHECK: hivm.hir.vsel ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<64x32xi1, #hivm.address_space<ub>>, memref<64x32xf16, #hivm.address_space<ub>>, memref<64x32xf16, #hivm.address_space<ub>>) outs(%{{.*}} : memref<64x32xf16, #hivm.address_space<ub>>)
  hivm.hir.vsel ins(%1, %alloc_1, %alloc_2 : memref<64x32xi1>, memref<64x32xf16>, memref<64x32xf16>) outs(%alloc_3 : memref<64x32xf16>)
  return
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

// -----

// CHECK-LABEL: test_infer_mem_scope_set_cbuf_for_aic_func_unused_alloc
module {
  func.func @test_infer_mem_scope_set_cbuf_for_aic_func_unused_alloc() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK: #hivm.address_space<cbuf>
    %alloc = memref.alloc() : memref<8x16xf32>
    // CHECK: #hivm.address_space<cbuf>
    annotation.mark %alloc {MayImplicitTransposeWithLastAxis} : memref<8x16xf32>
    return
  }
}

// -----

// CHECK-LABEL: test_infer_mem_scope_unrealized_conversion_cast
module {
  func.func @test_infer_mem_scope_unrealized_conversion_cast(%arg0 : memref<128xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    // CHECK: #hivm.address_space<gm>
    %1 = builtin.unrealized_conversion_cast %arg0 : memref<128xi32> to memref<256xi16>
    return
  }
}

// -----
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 * 35584)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 * 35584 + 16384)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0] -> (s0 * 35584 + 19456)>
// CHECK-LABEL:   func.func @triton_conv1d_mix_aic
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           hivm.hir.set_ffts_base_addr %{{.*}}
// CHECK:           hivm.hir.set_mask_norm
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           annotation.mark %{{.*}} {logical_block_num} : i32
// CHECK:           %{{.*}} = hivm.hir.get_block_idx -> i64
// CHECK:           %{{.*}} = arith.index_cast %{{.*}} : i64 to index
// CHECK:           %{{.*}} = affine.apply #[[$ATTR_0]](){{\[}}%{{.*}}]
// CHECK:           %{{.*}} = memref.view %{{.*}}{{\[}}%{{.*}}][] : memref<?xi8, #hivm.address_space<gm>> to memref<2x2x1x128x16xf16, #hivm.address_space<gm>>
// CHECK:           hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
// CHECK:           %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<2x2x1x128x16xf16, #hivm.address_space<cbuf>>
// CHECK:           hivm.hir.load ins(%{{.*}} : memref<2x2x1x128x16xf16, #hivm.address_space<gm>>) outs(%{{.*}} : memref<2x2x1x128x16xf16, #hivm.address_space<cbuf>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
// CHECK:           %{{.*}} = affine.apply #[[$ATTR_1]](){{\[}}%{{.*}}]
// CHECK:           %{{.*}} = memref.view %{{.*}}{{\[}}%{{.*}}][] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1x3x32x16xf16, #hivm.address_space<gm>>
// CHECK:           hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
// CHECK:           %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<1x1x3x32x16xf16, #hivm.address_space<cbuf>>
// CHECK:           hivm.hir.load ins(%{{.*}} : memref<1x1x3x32x16xf16, #hivm.address_space<gm>>) outs(%{{.*}} : memref<1x1x3x32x16xf16, #hivm.address_space<cbuf>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
// CHECK:           %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32, #hivm.address_space<cc>>
// CHECK:           hivm.hir.Conv1dL1 {fixpipe_already_inserted = true, groups = 2 : i32, outputAlreadyNormalized, padding = 0 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<2x2x1x128x16xf16, #hivm.address_space<cbuf>>, memref<1x1x3x32x16xf16, #hivm.address_space<cbuf>>, i1) outs(%{{.*}} : memref<128x64xf32, #hivm.address_space<cc>>)
// CHECK:           %{{.*}} = memref.subview %{{.*}}[0, 0] [126, 64] [1, 1] : memref<128x64xf32, #hivm.address_space<cc>> to memref<126x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>
// CHECK:           %{{.*}} = affine.apply #[[$ATTR_2]](){{\[}}%{{.*}}]
// CHECK:           %{{.*}} = memref.view %{{.*}}{{\[}}%{{.*}}][] : memref<?xi8, #hivm.address_space<gm>> to memref<126x64xf16, #hivm.address_space<gm>>
// CHECK:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%{{.*}} : memref<126x64xf32, strided<[64, 1]>, #hivm.address_space<cc>>) outs(%{{.*}} : memref<126x64xf16, #hivm.address_space<gm>>)
// CHECK:           annotation.mark %{{.*}} : memref<126x64xf16, #hivm.address_space<gm>>
// CHECK:           hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
// CHECK:           return
// CHECK:         }
#map = affine_map<()[s0] -> (s0 * 35584)>
#map1 = affine_map<()[s0] -> (s0 * 35584 + 16384)>
#map2 = affine_map<()[s0] -> (s0 * 35584 + 19456)>
func.func @triton_conv1d_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "aiv", parallel_mode = "simd"} {
  %true = arith.constant true
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.index_cast %2 : i64 to index
  %4 = affine.apply #map()[%3]
  %view = memref.view %arg2[%4][] : memref<?xi8> to memref<2x2x1x128x16xf16>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x1x128x16xf16>
  hivm.hir.load ins(%view : memref<2x2x1x128x16xf16>) outs(%alloc : memref<2x2x1x128x16xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %5 = affine.apply #map1()[%3]
  %view_0 = memref.view %arg2[%5][] : memref<?xi8> to memref<1x1x3x32x16xf16>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x3x32x16xf16>
  hivm.hir.load ins(%view_0 : memref<1x1x3x32x16xf16>) outs(%alloc_1 : memref<1x1x3x32x16xf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
  hivm.hir.Conv1dL1 {fixpipe_already_inserted = true, groups = 2 : i32, outputAlreadyNormalized, padding = 0 : i32} ins(%alloc, %alloc_1, %true : memref<2x2x1x128x16xf16>, memref<1x1x3x32x16xf16>, i1) outs(%alloc_2 : memref<128x64xf32>)
  %subview = memref.subview %alloc_2[0, 0] [126, 64] [1, 1] : memref<128x64xf32> to memref<126x64xf32, strided<[64, 1]>>
  %6 = affine.apply #map2()[%3]
  %view_3 = memref.view %arg2[%6][] : memref<?xi8> to memref<126x64xf16>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%subview : memref<126x64xf32, strided<[64, 1]>>) outs(%view_3 : memref<126x64xf16>)
  annotation.mark %view_3 : memref<126x64xf16>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  return
}

// -----
// CHECK-LABEL:   func.func @triton_conv2d_fp32_aligned_mix_aic
// CHECK:           %{{.*}} = arith.constant true
// CHECK:           hivm.hir.set_ffts_base_addr %{{.*}}
// CHECK:           hivm.hir.set_mask_norm
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           annotation.mark %{{.*}} {logical_block_num} : i32
// CHECK:           %{{.*}} = memref_ext.alloc_workspace() from %{{.*}} offset = [%{{.*}}] : from memref<?xi8, #hivm.address_space<gm>> to memref<1x2x32x32x8xf32, #hivm.address_space<gm>>
// CHECK:           annotation.mark %{{.*}} {hivm_data_layout = #hivm.data_layout<NC1HWC0>} : memref<1x2x32x32x8xf32, #hivm.address_space<gm>>
// CHECK:           hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
// CHECK:           %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<1x2x32x32x8xf32, #hivm.address_space<cbuf>>
// CHECK:           hivm.hir.load ins(%{{.*}} : memref<1x2x32x32x8xf32, #hivm.address_space<gm>>) outs(%{{.*}} : memref<1x2x32x32x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
// CHECK:           %{{.*}} = memref_ext.alloc_workspace() from %{{.*}} offset = [%{{.*}}] : from memref<?xi8, #hivm.address_space<gm>> to memref<2x3x3x16x8xf32, #hivm.address_space<gm>>
// CHECK:           annotation.mark %{{.*}} {hivm_data_layout = #hivm.data_layout<C1HWNC0>} : memref<2x3x3x16x8xf32, #hivm.address_space<gm>>
// CHECK:           hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
// CHECK:           %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<2x3x3x16x8xf32, #hivm.address_space<cbuf>>
// CHECK:           hivm.hir.load ins(%{{.*}} : memref<2x3x3x16x8xf32, #hivm.address_space<gm>>) outs(%{{.*}} : memref<2x3x3x16x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
// CHECK:           %{{.*}} = memref.alloc() {alignment = 64 : i64} : memref<912x16xf32, #hivm.address_space<cc>>
// CHECK:           hivm.hir.Conv2dL1 {fixpipe_already_inserted = true, groups = 1 : i32, outputAlreadyNormalized, padding = 0 : i32} ins(%{{.*}}, %{{.*}}, %{{.*}} : memref<1x2x32x32x8xf32, #hivm.address_space<cbuf>>, memref<2x3x3x16x8xf32, #hivm.address_space<cbuf>>, i1) outs(%{{.*}} : memref<912x16xf32, #hivm.address_space<cc>>)
// CHECK:           %{{.*}} = memref.subview %{{.*}}[0, 0] [900, 16] [1, 1] : memref<912x16xf32, #hivm.address_space<cc>> to memref<900x16xf32, strided<[16, 1]>, #hivm.address_space<cc>>
// CHECK:           %{{.*}} = memref_ext.alloc_workspace() from %{{.*}} offset = [%{{.*}}] : from memref<?xi8, #hivm.address_space<gm>> to memref<900x16xf32, #hivm.address_space<gm>>
// CHECK:           hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%{{.*}} : memref<900x16xf32, strided<[16, 1]>, #hivm.address_space<cc>>) outs(%{{.*}} : memref<900x16xf32, #hivm.address_space<gm>>)
// CHECK:           annotation.mark %{{.*}} : memref<900x16xf32, #hivm.address_space<gm>>
// CHECK:           hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
// CHECK:           return
// CHECK:         }
func.func @triton_conv2d_fp32_aligned_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "aiv", parallel_mode = "simd"} {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c65536 = arith.constant 65536 : index
  %c74752 = arith.constant 74752 : index
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<1x2x32x32x8xf32>
  annotation.mark %2 {hivm_data_layout = #hivm.data_layout<NC1HWC0>} : memref<1x2x32x32x8xf32>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x32x32x8xf32>
  hivm.hir.load ins(%2 : memref<1x2x32x32x8xf32>) outs(%alloc : memref<1x2x32x32x8xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %3 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8> to memref<2x3x3x16x8xf32>
  annotation.mark %3 {hivm_data_layout = #hivm.data_layout<C1HWNC0>} : memref<2x3x3x16x8xf32>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x3x3x16x8xf32>
  hivm.hir.load ins(%3 : memref<2x3x3x16x8xf32>) outs(%alloc_0 : memref<2x3x3x16x8xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<912x16xf32>
  hivm.hir.Conv2dL1 {fixpipe_already_inserted = true, groups = 1 : i32, outputAlreadyNormalized, padding = 0 : i32} ins(%alloc, %alloc_0, %true : memref<1x2x32x32x8xf32>, memref<2x3x3x16x8xf32>, i1) outs(%alloc_1 : memref<912x16xf32>)
  %subview = memref.subview %alloc_1[0, 0] [900, 16] [1, 1] : memref<912x16xf32> to memref<900x16xf32, strided<[16, 1]>>
  %4 = memref_ext.alloc_workspace() from %arg2 offset = [%c74752] : from memref<?xi8> to memref<900x16xf32>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%subview : memref<900x16xf32, strided<[16, 1]>>) outs(%4 : memref<900x16xf32>)
  annotation.mark %4 : memref<900x16xf32>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  return
}
