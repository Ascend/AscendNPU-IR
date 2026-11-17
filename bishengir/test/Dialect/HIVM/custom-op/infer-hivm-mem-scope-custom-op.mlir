// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-mem-scope -split-input-file | FileCheck %s

// The memory scope of a CustomOp's operands is inferred from the
// (hivm.tcore_type, hivm.pipe) pair: the pipe defines where the operand
// lives before and after the custom kernel runs.

// -----

// VECTOR / PIPE_V: the source and the destination both live in GM.
// CHECK-LABEL: func.func @test_custom_pipe_v
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
// CHECK: return %[[RES]]
func.func @test_custom_pipe_v(%arg0: memref<128xf32, #hivm.address_space<gm>>, %arg1: memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_pipe_v"}
      "k_pipe_v"
      ins(%arg0 : memref<128xf32, #hivm.address_space<gm>>)
      outs(%arg1 : memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
  return %0 : memref<128xf32, #hivm.address_space<gm>>
}

// -----

// CUBE / PIPE_FIX: the operand is read from L0C and written back to GM.
// CHECK-LABEL: func.func @test_custom_pipe_fix
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<cc>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<cc>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
// CHECK: return %[[RES]]
func.func @test_custom_pipe_fix(%arg0: memref<128xf32, #hivm.address_space<cc>>, %arg1: memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<cc>}, {address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_FIX>,
       hivm.tcore_type = #hivm.tcore_type<CUBE>,
       symbol = "k_pipe_fix"}
      "k_pipe_fix"
      ins(%arg0 : memref<128xf32, #hivm.address_space<cc>>)
      outs(%arg1 : memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
  return %0 : memref<128xf32, #hivm.address_space<gm>>
}

// -----

// CUBE / PIPE_MTE2: the operand is loaded from GM into L1.
// CHECK-LABEL: func.func @test_custom_cube_pipe_mte2
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<cbuf>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<cbuf>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<cbuf>>
// CHECK: return %[[RES]]
func.func @test_custom_cube_pipe_mte2(%arg0: memref<128xf32, #hivm.address_space<gm>>,
                                      %arg1: memref<128xf32, #hivm.address_space<cbuf>>) -> memref<128xf32, #hivm.address_space<cbuf>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<cbuf>}, {address_space = #hivm.address_space<cbuf>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE2>,
       hivm.tcore_type = #hivm.tcore_type<CUBE>,
       symbol = "k_cube_mte2"}
      "k_cube_mte2"
      ins(%arg0 : memref<128xf32, #hivm.address_space<gm>>)
      outs(%arg1 : memref<128xf32, #hivm.address_space<cbuf>>) -> memref<128xf32, #hivm.address_space<cbuf>>
  return %0 : memref<128xf32, #hivm.address_space<cbuf>>
}

// -----

// CUBE / PIPE_MTE3: the operand is read from L1 and written back to GM.
// CHECK-LABEL: func.func @test_custom_cube_pipe_mte3
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<cbuf>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<cbuf>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
// CHECK: return %[[RES]]
func.func @test_custom_cube_pipe_mte3(%arg0: memref<128xf32, #hivm.address_space<cbuf>>,
                                      %arg1: memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<cbuf>}, {address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE3>,
       hivm.tcore_type = #hivm.tcore_type<CUBE>,
       symbol = "k_cube_mte3"}
      "k_cube_mte3"
      ins(%arg0 : memref<128xf32, #hivm.address_space<cbuf>>)
      outs(%arg1 : memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
  return %0 : memref<128xf32, #hivm.address_space<gm>>
}

// -----

// VECTOR / PIPE_MTE2: the source comes from GM, the destination is in UB.
// CHECK-LABEL: func.func @test_custom_vector_pipe_mte2
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<ub>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<ub>>
// CHECK: return %[[RES]]
func.func @test_custom_vector_pipe_mte2(%arg0: memref<128xf32, #hivm.address_space<gm>>,
                                        %arg1: memref<128xf32, #hivm.address_space<ub>>) -> memref<128xf32, #hivm.address_space<ub>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<ub>}, {address_space = #hivm.address_space<ub>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE2>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_vector_mte2"}
      "k_vector_mte2"
      ins(%arg0 : memref<128xf32, #hivm.address_space<gm>>)
      outs(%arg1 : memref<128xf32, #hivm.address_space<ub>>) -> memref<128xf32, #hivm.address_space<ub>>
  return %0 : memref<128xf32, #hivm.address_space<ub>>
}

// -----

// VECTOR / PIPE_MTE3: the source is in UB, the destination is in GM.
// CHECK-LABEL: func.func @test_custom_vector_pipe_mte3
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<ub>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<ub>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
// CHECK: return %[[RES]]
func.func @test_custom_vector_pipe_mte3(%arg0: memref<128xf32, #hivm.address_space<ub>>,
                                        %arg1: memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<ub>}, {address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE3>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_vector_mte3"}
      "k_vector_mte3"
      ins(%arg0 : memref<128xf32, #hivm.address_space<ub>>)
      outs(%arg1 : memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
  return %0 : memref<128xf32, #hivm.address_space<gm>>
}

// -----

// The custom op user is found through a view-like op: the function argument
// feeding the subview is the destination of the custom op, so it must be
// inferred as the pipe destination (GM for VECTOR / PIPE_V).
// CHECK-LABEL: func.func @test_custom_through_view
// CHECK-SAME: %[[ARG:.*]]: memref<?x?xf32, #hivm.address_space<gm>>
// CHECK: %[[VIEW:.*]] = memref.subview %[[ARG]]
// CHECK-SAME: memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[VIEW]] : memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
// CHECK-SAME: outs(%[[VIEW]] : memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
// CHECK: return %[[RES]]
func.func @test_custom_through_view(%arg0: memref<?x?xf32, #hivm.address_space<gm>>) -> memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %view = memref.subview %arg0[%c0, %c0] [128, 128] [1, 1]
      : memref<?x?xf32, #hivm.address_space<gm>> to memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_view"}
      "k_view"
      ins(%view : memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      outs(%view : memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      -> memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
  return %0 : memref<128x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
}

// -----

// The custom op result is scoped after the pipe destination, even when the
// output buffer is an internal allocation.
// CHECK-LABEL: func.func @test_custom_result_scope
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128xf32>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: outs(%[[ALLOC]] : memref<128xf32>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
// CHECK: return %[[RES]]
func.func @test_custom_result_scope(%arg0: memref<128xf32, #hivm.address_space<gm>>) -> memref<128xf32, #hivm.address_space<gm>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc = memref.alloc() : memref<128xf32>
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_result"}
      "k_result"
      ins(%arg0 : memref<128xf32, #hivm.address_space<gm>>)
      outs(%alloc : memref<128xf32>) -> memref<128xf32, #hivm.address_space<gm>>
  return %0 : memref<128xf32, #hivm.address_space<gm>>
}

// -----

// Unscoped arguments take the address_space declared in arg_attrs. The
// result memref starts unscoped and is inferred from the output entry.
// CHECK-LABEL: func.func @test_custom_infer_unscoped_operands
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<cc>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: ins(%[[IN]] : memref<128xf32, #hivm.address_space<cc>>)
// CHECK-SAME: outs(%[[OUT]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
func.func @test_custom_infer_unscoped_operands(%arg0: memref<128xf32>, %arg1: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<cc>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_FIX>,
       hivm.tcore_type = #hivm.tcore_type<CUBE>,
       symbol = "k_infer"}
      "k_infer"
      ins(%arg0 : memref<128xf32>)
      outs(%arg1 : memref<128xf32>) -> memref<128xf32>
  return
}

// -----

// No arg_attrs at all: every traced operand defaults to GM.
// CHECK-LABEL: func.func @test_custom_missing_arg_attrs
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>, %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
func.func @test_custom_missing_arg_attrs(%arg0: memref<128xf32>, %arg1: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  hivm.hir.custom
      {hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_no_attrs"}
      "k_no_attrs"
      ins(%arg0 : memref<128xf32>)
      outs(%arg1 : memref<128xf32>)
  return
}

// -----

// A dictionary without address_space defaults to GM.
// CHECK-LABEL: func.func @test_custom_dict_without_address_space
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>
func.func @test_custom_dict_without_address_space(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.custom
      {arg_attrs = [{data_layout = #hivm.data_layout<ND>}, {address_space = #hivm.address_space<ub>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_no_space_key"}
      "k_no_space_key"
      ins(%arg0 : memref<128xf32>)
      outs(%alloc : memref<128xf32, #hivm.address_space<ub>>)
  return
}

// -----

// arg_attrs is shorter than the operand list, so the output index is out of
// range and the unscoped output defaults to GM.
// CHECK-LABEL: func.func @test_custom_arg_attrs_out_of_range
// CHECK-SAME: %[[OUT:.*]]: memref<128xf32, #hivm.address_space<gm>>
func.func @test_custom_arg_attrs_out_of_range(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %src = memref.alloc() : memref<128xf32, #hivm.address_space<gm>>
  hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_oob"}
      "k_oob"
      ins(%src : memref<128xf32, #hivm.address_space<gm>>)
      outs(%arg0 : memref<128xf32>)
  return
}

// -----

// A non-dictionary arg_attrs entry cannot carry address_space, so the
// operand defaults to GM.
// CHECK-LABEL: func.func @test_custom_nondict_arg_attrs
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>
func.func @test_custom_nondict_arg_attrs(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.custom
      {arg_attrs = [true, {address_space = #hivm.address_space<ub>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_nondict"}
      "k_nondict"
      ins(%arg0 : memref<128xf32>)
      outs(%alloc : memref<128xf32, #hivm.address_space<ub>>)
  return
}

// -----

// A temp buffer is not an input or an output, so arg_attrs does not describe
// it and the traced argument defaults to GM.
// CHECK-LABEL: func.func @test_custom_temp_buffer_defaults_gm
// CHECK-SAME: %[[TMP:.*]]: memref<256xf16, #hivm.address_space<gm>>
// CHECK: hivm.hir.custom
// CHECK-SAME: tmps(%[[TMP]] : memref<256xf16, #hivm.address_space<gm>>)
func.func @test_custom_temp_buffer_defaults_gm(%tmp: memref<256xf16>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
  hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<cbuf>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE2>,
       hivm.tcore_type = #hivm.tcore_type<CUBE>,
       symbol = "k_tmp"}
      "k_tmp"
      ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
      outs(%dst : memref<16x16xf16, #hivm.address_space<cbuf>>)
      tmps(%tmp : memref<256xf16>)
  return
}

// -----

// The same select consumes the argument twice. The second use hits the
// visited set, and the select result is still traced to the custom op.
// CHECK-LABEL: func.func @test_custom_through_select
// CHECK-SAME: %[[ARG:.*]]: memref<128xf32, #hivm.address_space<ub>>
// CHECK: %[[SEL:.*]] = arith.select %{{.*}}, %[[ARG]], %[[ARG]] : memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[SEL]] : memref<128xf32, #hivm.address_space<ub>>)
func.func @test_custom_through_select(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %true = arith.constant true
  %sel = arith.select %true, %arg0, %arg0 : memref<128xf32>
  %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<gm>>
  hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<ub>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE3>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_select"}
      "k_select"
      ins(%sel : memref<128xf32>)
      outs(%alloc : memref<128xf32, #hivm.address_space<gm>>)
  return
}

// -----

// Transpose is a single-result propagatable memref op, so the root argument
// receives the custom op input scope.
// CHECK-LABEL: func.func @test_custom_through_transpose
// CHECK-SAME: %[[ARG:.*]]: memref<16x16xf32, #hivm.address_space<cbuf>>
// CHECK: %[[T:.*]] = memref.transpose %[[ARG]]
// CHECK-SAME: memref<16x16xf32, strided<[1, 16]>, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[T]] : memref<16x16xf32, strided<[1, 16]>, #hivm.address_space<cbuf>>)
func.func @test_custom_through_transpose(%arg0: memref<16x16xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %t = memref.transpose %arg0 (i, j) -> (j, i) : memref<16x16xf32> to memref<16x16xf32, strided<[1, 16]>>
  %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<gm>>
  hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<cbuf>}, {address_space = #hivm.address_space<gm>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE3>,
       hivm.tcore_type = #hivm.tcore_type<CUBE>,
       symbol = "k_transpose"}
      "k_transpose"
      ins(%t : memref<16x16xf32, strided<[1, 16]>>)
      outs(%alloc : memref<16x16xf32, #hivm.address_space<gm>>)
  return
}

// -----

// Bitcast is traced the same way: the root argument gets the input scope.
// CHECK-LABEL: func.func @test_custom_through_bitcast
// CHECK-SAME: %[[ARG:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[B:.*]] = hivm.hir.bitcast %[[ARG]] : memref<128xf32, #hivm.address_space<gm>> -> memref<128xi32, #hivm.address_space<gm>>
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[B]] : memref<128xi32, #hivm.address_space<gm>>)
func.func @test_custom_through_bitcast(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %b = hivm.hir.bitcast %arg0 : memref<128xf32> -> memref<128xi32>
  %alloc = memref.alloc() : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<ub>}],
       hivm.pipe = #hivm.pipe<PIPE_MTE2>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_bitcast"}
      "k_bitcast"
      ins(%b : memref<128xi32>)
      outs(%alloc : memref<128xi32, #hivm.address_space<ub>>)
  return
}

// -----

// An argument with no custom user stays on the default GM scope.
// CHECK-LABEL: func.func @test_arg_without_custom_user
// CHECK-SAME: memref<128xf32, #hivm.address_space<gm>>
func.func @test_arg_without_custom_user(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  return
}

// -----

// A tensor result is not a memref, so result-scope inference skips it.
// The memref input is still scoped from arg_attrs.
// CHECK-LABEL: func.func @test_custom_tensor_result_skipped
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.custom
// CHECK-SAME: -> tensor<128xf32>
func.func @test_custom_tensor_result_skipped(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<128xf32>
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<ub>}, {}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_tensor"}
      "k_tensor"
      ins(%arg0 : memref<128xf32>)
      outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  return
}

// -----

// The result already carries a memory space, so inference leaves it alone
// even when arg_attrs names a different space.
// CHECK-LABEL: func.func @test_custom_result_already_scoped
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: -> memref<128xf32, #hivm.address_space<cbuf>>
func.func @test_custom_result_already_scoped(%arg0: memref<128xf32>) -> memref<128xf32, #hivm.address_space<cbuf>>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<cbuf>>
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {address_space = #hivm.address_space<ub>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_already"}
      "k_already"
      ins(%arg0 : memref<128xf32>)
      outs(%alloc : memref<128xf32, #hivm.address_space<cbuf>>)
      -> memref<128xf32, #hivm.address_space<cbuf>>
  return %0 : memref<128xf32, #hivm.address_space<cbuf>>
}

// -----

// The output arg_attrs entry has no address_space, so the unscoped result
// is left unchanged.
// CHECK-LABEL: func.func @test_custom_result_missing_address_space
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<gm>>
// CHECK: %[[RES:.*]] = hivm.hir.custom
// CHECK-SAME: -> memref<128xf32>
func.func @test_custom_result_missing_address_space(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc = memref.alloc() : memref<128xf32>
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<gm>}, {data_layout = #hivm.data_layout<ND>}],
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_result_nospace"}
      "k_result_nospace"
      ins(%arg0 : memref<128xf32>)
      outs(%alloc : memref<128xf32>) -> memref<128xf32>
  return
}

// -----

// Distributed custom ops skip arg_attrs result inference. The input keeps the
// scope from arg_attrs (UB). The unscoped result is then marked GM by the
// distributed-custom rule, not UB from arg_attrs.
// CHECK-LABEL: func.func @test_distributed_custom_skips_result
// CHECK-SAME: %[[IN:.*]]: memref<128xf32, #hivm.address_space<ub>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<128xf32, #hivm.address_space<gm>>
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.is_distributed
// CHECK-SAME: outs(%[[ALLOC]] : memref<128xf32, #hivm.address_space<gm>>)
// CHECK-SAME: -> memref<128xf32, #hivm.address_space<gm>>
func.func @test_distributed_custom_skips_result(%arg0: memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc = memref.alloc() : memref<128xf32>
  %0 = hivm.hir.custom
      {arg_attrs = [{address_space = #hivm.address_space<ub>}, {address_space = #hivm.address_space<ub>}],
       hivm.is_distributed,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_dist"}
      "k_dist"
      ins(%arg0 : memref<128xf32>)
      outs(%alloc : memref<128xf32>) -> memref<128xf32>
  return
}
