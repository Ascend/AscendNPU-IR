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
