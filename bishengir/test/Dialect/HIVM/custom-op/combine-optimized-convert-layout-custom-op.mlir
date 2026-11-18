// RUN: bishengir-opt %s --canonicalize --hivm-combine-optimized-convert-layout --split-input-file | FileCheck %s

// Fold Fractal->ND convert_layout before an hivm.hir.custom FixpipeOp: the
// custom op's first input is declared with data_layout = Fractal, so the
// convert_layout is redundant and the custom op consumes the fractal tensor
// directly.
// CHECK-LABEL: func.func @fold_convert_fixpipe_custom
// CHECK-SAME: %[[DST:.*]]: memref<?xf32>
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: hivm.hir.custom {{.*}} "FixpipeOp" ins(%{{.*}} : tensor<1x1x16x16xf32>
func.func @fold_convert_fixpipe_custom(%dst: memref<?xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %false = arith.constant false
  %fractal = arith.constant dense<1.0> : tensor<1x1x16x16xf32>
  %nd = hivm.hir.convert_layout %fractal output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<1x1x16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.custom {arg_attrs = [{address_space = #hivm.address_space<cc>, data_layout = #hivm.data_layout<Fractal>}, {}, {}, {}, {}, {}, {}, {}, {address_space = #hivm.address_space<gm>, data_layout = #hivm.data_layout<dotC_ND>}], bitcode = "fixpipe_l0c_to_gm.bc", hivm.pipe = #hivm.pipe<PIPE_FIX>, hivm.tcore_type = #hivm.tcore_type<CUBE>, indexing_map = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>], symbol = "fixpipe_nz2dn_float_to_float_4d_to_2d_gm"} "FixpipeOp" ins(%nd, %c0_i32, %cst, %c0_i32, %false, %false, %c0_i32, %c0_i32, %dst : tensor<16x16xf32>, i32, f32, i32, i1, i1, i32, i32, memref<?xf32>)
  return
}

// -----
// Fold ND->Fractal convert_layout after a custom op whose output already
// declares data_layout = Fractal in arg_attrs. The convert_layout is
// redundant because the custom op already produces the fractal layout.
// CHECK-LABEL: func.func @fold_custom_op_nd2nz_output
// CHECK: %[[FRACTAL:.+]] = hivm.hir.convert_layout
// CHECK: %[[CUSTOM:.+]] = hivm.hir.custom {{.*}} outs(%[[FRACTAL]]
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: return %[[CUSTOM]] : tensor<1x1x16x16xf16>
func.func @fold_custom_op_nd2nz_output(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %fractal = hivm.hir.convert_layout %arg output_shape [1, 1, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
                        : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [{}, {address_space = #hivm.address_space<cbuf>, data_layout = #hivm.data_layout<Fractal>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    indexing_map = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
    symbol = "test_op"
  } "TestOp" ins(%arg : tensor<16x16xf16>) outs(%fractal : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %identity = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
                        : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %identity : tensor<1x1x16x16xf16>
}

// -----
// Do NOT fold when the custom op's declared output layout does not match
// the convert_layout's dstLayout.
// CHECK-LABEL: func.func @no_fold_custom_op_layout_mismatch
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_layout_mismatch(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %fractal = hivm.hir.convert_layout %arg output_shape [1, 1, 16, 16] {
      srcLayout = #hivm.data_layout<ND>,
      dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
                        : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [{address_space = #hivm.address_space<gm>, data_layout = #hivm.data_layout<ND>},
                 {address_space = #hivm.address_space<cbuf>, data_layout = #hivm.data_layout<ND>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    indexing_map = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#hivm.iterator_type<parallel>, #hivm.iterator_type<parallel>],
    symbol = "test_op"
  } "TestOp" ins(%arg : tensor<16x16xf16>) outs(%fractal : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %identity = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
                        : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %identity : tensor<1x1x16x16xf16>
}

// -----
// Simplified _nd2nz_mmad_kernel: nd2nz + mmadL1 + fixpipe.
// (a) The convert_layout after the ND2NZ custom op is redundant and is dropped,
//     so the custom op result feeds mmadL1 directly.
// (b) The convert_layout after hivm.hir.load folds into an explicit
//     hivm.hir.nd2nz into a new fractal L1 buffer.
// (c) The convert_layout before the nz2nd fixpipe is folded into the fixpipe.
// CHECK-LABEL: func.func @fold_nd2nz_mmad_kernel
// CHECK-SAME: %[[B:.*]]: memref<?xf16>, %[[RES:.*]]: memref<?xf16>
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: %[[CUSTOM:.*]] = hivm.hir.custom
// CHECK-SAME: "ND2NZOp" ins({{.*}}) outs(%{{.*}} : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK: %[[BUFR:.*]] = memref.reinterpret_cast %[[B]] to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
// CHECK: %[[L1:.*]] = memref.alloc() : memref<1x1x16x16xf16>
// CHECK: hivm.hir.nd2nz {dst_continuous} ins(%[[BUFR]] : memref<16x16xf16, strided<[16, 1]>>) outs(%[[L1]] : memref<1x1x16x16xf16>)
// CHECK: %[[BT:.*]] = bufferization.to_tensor %[[L1]] restrict writable : memref<1x1x16x16xf16>
// CHECK: %[[MMAD:.*]] = hivm.hir.mmadL1
// CHECK-SAME: ins(%[[BT]], %[[CUSTOM]], {{.*}} : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index)
// CHECK: %[[OUTR:.*]] = memref.reinterpret_cast %[[RES]] to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
// CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%[[MMAD]] : tensor<1x1x16x16xf32>) outs(%[[OUTR]] : memref<16x16xf16, strided<[16, 1]>>)
func.func @fold_nd2nz_mmad_kernel(
    %bsrc: memref<?xf16>, %result: memref<?xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %empty_b = tensor.empty() : tensor<16x16xf16>
  %b_in = memref.reinterpret_cast %bsrc to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
  %b_custom = hivm.hir.custom {
    arg_attrs = [{address_space = #hivm.address_space<gm>, data_layout = #hivm.data_layout<ND>},
                 {},
                 {address_space = #hivm.address_space<cbuf>, data_layout = #hivm.data_layout<Fractal>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    indexing_map = [affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> ()>,
                    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#hivm.iterator_type<parallel>,
                      #hivm.iterator_type<parallel>],
    symbol = "nd2nz_gm_to_l1_half"
  } "ND2NZOp"
      ins(%b_in, %c0_i32 : memref<16x16xf16, strided<[16, 1]>>, i32)
      outs(%empty_b : tensor<16x16xf16>) -> tensor<16x16xf16>
  %b_fractal = hivm.hir.convert_layout %b_custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  %a_in = memref.reinterpret_cast %bsrc to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
  %a_alloc = memref.alloc() : memref<16x16xf16>
  hivm.hir.load ins(%a_in : memref<16x16xf16, strided<[16, 1]>>) outs(%a_alloc : memref<16x16xf16>) eviction_policy = <EvictFirst>
  %a_nd = bufferization.to_tensor %a_alloc restrict writable : memref<16x16xf16>
  %a_fractal = hivm.hir.convert_layout %a_nd output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  %cc = tensor.empty() : tensor<1x1x16x16xf32>
  %mmad = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C}
      ins(%a_fractal, %b_fractal, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index)
      outs(%cc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
  %mmad_nd = hivm.hir.convert_layout %mmad output_shape [16, 16]
      {dstLayout = #hivm.data_layout<ND>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<1x1x16x16xf32>) -> tensor<16x16xf32>
  %res = memref.reinterpret_cast %result to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16, strided<[16, 1]>>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%mmad_nd : tensor<16x16xf32>) outs(%res : memref<16x16xf16, strided<[16, 1]>>)
  return
}

// -----

// A rank-changing convert after a custom op with a temp buffer is folded into
// the custom op. The temp buffer is preserved on the rebuilt op.
// CHECK-LABEL: func.func @fold_custom_op_with_temp_buffer
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x1x16x16xf16>
// CHECK: %[[CUSTOM:.*]] = hivm.hir.custom
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1x1x16x16xf16>)
// CHECK-SAME: tmps(%{{.*}} : memref<256xf16>)
// CHECK-SAME: -> tensor<1x1x16x16xf16>
// CHECK-NOT: hivm.hir.convert_layout
func.func @fold_custom_op_with_temp_buffer(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %tmp = memref.alloc() : memref<256xf16>
  %empty = tensor.empty() : tensor<16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [{data_layout = #hivm.data_layout<ND>},
                 {data_layout = #hivm.data_layout<Fractal>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "nd2nz_with_tmp"
  } "ND2NZTmp"
      ins(%arg : tensor<16x16xf16>)
      outs(%empty : tensor<16x16xf16>)
      tmps(%tmp : memref<256xf16>) -> tensor<16x16xf16>
  %fractal = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  return %fractal : tensor<1x1x16x16xf16>
}

// -----

// Two live users of the custom result: the convert is not the sole user, so
// neither convert is folded.
// CHECK-LABEL: func.func @no_fold_custom_op_multiple_uses
// CHECK: hivm.hir.convert_layout
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_multiple_uses(%arg : tensor<16x16xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) {
  %empty = tensor.empty() : tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [{}, {data_layout = #hivm.data_layout<Fractal>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "multi_use"
  } "MultiUse" ins(%arg : tensor<16x16xf16>) outs(%empty : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %c0 = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %c1 = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %c0, %c1 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
}

// -----

// More than one result: the declared output layout is ambiguous, so the
// convert stays.
// CHECK-LABEL: func.func @no_fold_custom_op_multiple_results
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_multiple_results(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %e0 = tensor.empty() : tensor<1x1x16x16xf16>
  %e1 = tensor.empty() : tensor<1x1x16x16xf16>
  %custom:2 = hivm.hir.custom {
    arg_attrs = [{},
                 {data_layout = #hivm.data_layout<Fractal>},
                 {data_layout = #hivm.data_layout<Fractal>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "two_results"
  } "TwoResults" ins(%arg : tensor<16x16xf16>) outs(%e0, %e1 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>)
  %c0 = hivm.hir.convert_layout %custom#0 output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %c0 : tensor<1x1x16x16xf16>
}

// -----

// No data_layout on the output entry: the convert stays.
// CHECK-LABEL: func.func @no_fold_custom_op_missing_data_layout
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_missing_data_layout(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [{}, {address_space = #hivm.address_space<cbuf>}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "no_layout"
  } "NoLayout" ins(%arg : tensor<16x16xf16>) outs(%empty : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %c0 = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %c0 : tensor<1x1x16x16xf16>
}

// -----

// No arg_attrs attribute: nothing to match the convert's dstLayout against.
// CHECK-LABEL: func.func @no_fold_custom_op_missing_arg_attrs
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_missing_arg_attrs(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "no_arg_attrs"
  } "NoArgAttrs" ins(%arg : tensor<16x16xf16>) outs(%empty : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %c0 = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %c0 : tensor<1x1x16x16xf16>
}

// -----

// The output's arg_attrs index is past the end of the array.
// CHECK-LABEL: func.func @no_fold_custom_op_arg_attrs_oob
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_arg_attrs_oob(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [{}],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "oob_layout"
  } "OobLayout" ins(%arg : tensor<16x16xf16>) outs(%empty : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %c0 = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %c0 : tensor<1x1x16x16xf16>
}

// -----

// A non-dictionary arg_attrs entry has no data_layout.
// CHECK-LABEL: func.func @no_fold_custom_op_nondict_arg_attrs
// CHECK: hivm.hir.convert_layout
func.func @no_fold_custom_op_nondict_arg_attrs(%arg : tensor<16x16xf16>) -> tensor<1x1x16x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16x16xf16>
  %custom = hivm.hir.custom {
    arg_attrs = [true, true],
    hivm.pipe = #hivm.pipe<PIPE_MTE2>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "nondict_layout"
  } "NondictLayout" ins(%arg : tensor<16x16xf16>) outs(%empty : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  %c0 = hivm.hir.convert_layout %custom output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
  return %c0 : tensor<1x1x16x16xf16>
}

// -----

// A FIX-pipe custom op is not a FixpipeOp. Fractal->ND convert_layout in
// front of it stays, and the custom op keeps the ND tensor.
// CHECK-LABEL: func.func @no_fold_convert_into_fixpipe_custom
// CHECK: %[[ND:.*]] = hivm.hir.convert_layout
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[ND]] : tensor<16x16xf32>)
// CHECK: hivm.hir.fixpipe
// CHECK-SAME: ins(%{{.*}} : tensor<1x1x16x16xf32>)
func.func @no_fold_convert_into_fixpipe_custom(%dst: memref<16x16xf32, strided<[16, 1], offset: ?>>) -> tensor<16x16xf32> {
  %fractal = arith.constant dense<0.0> : tensor<1x1x16x16xf32>
  %nd = hivm.hir.convert_layout %fractal output_shape [16, 16]
      {dstLayout = #hivm.data_layout<ND>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<1x1x16x16xf32>) -> tensor<16x16xf32>
  %empty = tensor.empty() : tensor<16x16xf32>
  %custom = hivm.hir.custom {
    hivm.pipe = #hivm.pipe<PIPE_FIX>,
    hivm.tcore_type = #hivm.tcore_type<CUBE>,
    symbol = "fixpipe_nz2nd_dummy"
  } "FixpipeCustom" ins(%nd : tensor<16x16xf32>) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%nd : tensor<16x16xf32>) outs(%dst : memref<16x16xf32, strided<[16, 1], offset: ?>>)
  return %custom : tensor<16x16xf32>
}
