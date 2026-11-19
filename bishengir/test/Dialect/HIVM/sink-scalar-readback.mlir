// RUN: bishengir-opt %s -hivm-sink-scalar-readback | FileCheck %s

func.func private @vf_0(tensor<1xf32>) -> tensor<1xf32> attributes {hivm.vector_function}
func.func private @vf_1(tensor<128xf32>) -> tensor<128xf32> attributes {hivm.vector_function}
func.func private @vf_2(tensor<128xf32>) -> tensor<128xf32> attributes {hivm.vector_function}
func.func private @vf_3(tensor<128xf32>, f32) -> tensor<128xf32> attributes {hivm.vector_function}
func.func private @vf_4(tensor<128xf32>, f32, f32) -> tensor<128xf32> attributes {hivm.vector_function}
func.func private @vf_5(tensor<1xf32>) -> tensor<1xf32> attributes {hivm.vector_function}
func.func private @vf_6(tensor<1xf32>, f32) -> tensor<128xf32> attributes {hivm.vector_function}
func.func private @vf_7() -> tensor<4xf32> attributes {hivm.vector_function}
func.func private @vf_8() -> tensor<?xf32> attributes {hivm.vector_function}
func.func private @vf_9() -> tensor<1x1xf32> attributes {hivm.vector_function}

// The readback chain (tensor.extract + dependent scalar arith) is moved past
// the intermediate vector calls, right before its consumer.

// CHECK-LABEL: func.func @sink_across_vector_calls
// CHECK: call @vf_0
// CHECK: call @vf_1
// CHECK: call @vf_2
// CHECK: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK-NEXT: call @vf_3
func.func @sink_across_vector_calls(%arg0: tensor<1xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %div = arith.divf %cst, %extracted : f32
  %vf1 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf2 = func.call @vf_2(%vf1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf3 = func.call @vf_3(%vf2, %div) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf3 : tensor<128xf32>
}

// Two independent chains, C1 (extract + divf, consumer vf_4) and C2 (extract,
// consumers vf_3 and vf_4): each chain lands before its own earliest consumer.

// CHECK-LABEL: func.func @sink_two_chains
// CHECK: call @vf_0
// CHECK: call @vf_5
// CHECK: call @vf_2
// CHECK: tensor.extract
// CHECK-NEXT: call @vf_3
// CHECK: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK-NEXT: call @vf_4
func.func @sink_two_chains(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %div = arith.divf %cst, %extracted : f32
  %vf1 = func.call @vf_5(%arg1) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted_0 = tensor.extract %vf1[%c0] : tensor<1xf32>
  %vf2 = func.call @vf_2(%arg2) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf3 = func.call @vf_3(%vf2, %extracted_0) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  %vf4 = func.call @vf_4(%vf3, %div, %extracted_0) {hivm.vector_function} : (tensor<128xf32>, f32, f32) -> tensor<128xf32>
  return %vf4 : tensor<128xf32>
}

// A source produced by bufferization.to_tensor is a view of a mutable memref:
// the readback must not be moved.

// CHECK-LABEL: func.func @no_sink_from_to_tensor
// CHECK: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK: call @vf_1
func.func @no_sink_from_to_tensor(%arg0: memref<1xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %tensor = bufferization.to_tensor %arg0 : memref<1xf32>
  %extracted = tensor.extract %tensor[%c0] : tensor<1xf32>
  %div = arith.divf %cst, %extracted : f32
  %vf0 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf1 = func.call @vf_3(%vf0, %div) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// A consumer inside a region is a different block: fail closed, no movement.

// CHECK-LABEL: func.func @no_sink_into_region
// CHECK: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK: call @vf_1
// CHECK: scf.if
func.func @no_sink_into_region(%arg0: tensor<1xf32>, %arg1: tensor<128xf32>, %cond: i1) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %div = arith.divf %cst, %extracted : f32
  %vf1 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  scf.if %cond {
    %vf2 = func.call @vf_3(%vf1, %div) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  }
  return %vf1 : tensor<128xf32>
}

// No displacement: the consumer directly follows the chain, leave the IR alone.

// CHECK-LABEL: func.func @no_sink_when_adjacent
// CHECK: call @vf_0
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK-NEXT: call @vf_6
func.func @no_sink_when_adjacent(%arg0: tensor<1xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %div = arith.divf %cst, %extracted : f32
  %vf1 = func.call @vf_6(%arg0, %div) {hivm.vector_function} : (tensor<1xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// A side-effecting user bounds the chain and is also the earliest consumer, so
// nothing is moved.

// CHECK-LABEL: func.func @no_sink_past_store
// CHECK: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK-NEXT: memref.store
// CHECK: call @vf_3
func.func @no_sink_past_store(%arg0: tensor<1xf32>, %out: memref<f32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %div = arith.divf %cst, %extracted : f32
  memref.store %div, %out[] : memref<f32>
  %vf1 = func.call @vf_3(%arg1, %div) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// math ops on the chain move together with the extract.

// CHECK-LABEL: func.func @sink_math_chain
// CHECK: call @vf_0
// CHECK: call @vf_1
// CHECK: tensor.extract
// CHECK-NEXT: math.exp
// CHECK-NEXT: arith.divf
// CHECK-NEXT: call @vf_3
func.func @sink_math_chain(%arg0: tensor<1xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %exp = math.exp %extracted : f32
  %div = arith.divf %cst, %exp : f32
  %vf1 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf2 = func.call @vf_3(%vf1, %div) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf2 : tensor<128xf32>
}

// A chained readback that reads a sliced `to_tensor` view must not move: the
// slice keeps the mutable storage, so the later read could observe a write that
// happens in between.  The chain simply bounds at the view read.

// CHECK-LABEL: func.func @no_sink_chained_to_tensor_extract
// CHECK: call @vf_0
// CHECK-NEXT: bufferization.to_tenso
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.fptosi
// CHECK-NEXT: arith.index_cast
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.divf
// CHECK-NEXT: call @vf_1
func.func @no_sink_chained_to_tensor_extract(%arg0: tensor<1xf32>, %mem: memref<8xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %extracted = tensor.extract %vf0[%c0] : tensor<1xf32>
  %i = arith.fptosi %extracted : f32 to i32
  %idx = arith.index_cast %i : i32 to index
  %view = bufferization.to_tensor %mem : memref<8xf32>
  %other = tensor.extract %view[%idx] : tensor<8xf32>
  %div = arith.divf %cst, %other : f32
  %vf1 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf2 = func.call @vf_3(%vf1, %div) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf2 : tensor<128xf32>
}

// Two chains that share a consumer and interleave: A = {extract, mulf} and
// B = {extract, addf}, where addf consumes both.  Each chain lands before its
// own earliest consumer, and one visit per seed is enough.  The two chains
// leapfrog each other on every round, so a fixed-point loop spins forever here.

// CHECK-LABEL: func.func @sink_two_chains_overlap
// CHECK: call @vf_0
// CHECK: call @vf_5
// CHECK: call @vf_2
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.addf
// CHECK-NEXT: call @vf_3
// CHECK-NEXT: call @vf_4
func.func @sink_two_chains_overlap(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %ea = tensor.extract %vf0[%c0] : tensor<1xf32>
  %vf5 = func.call @vf_5(%arg1) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %eb = tensor.extract %vf5[%c0] : tensor<1xf32>
  %sum = arith.addf %ea, %eb : f32
  %prod = arith.mulf %ea, %cst : f32
  %vf2 = func.call @vf_2(%arg2) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf3 = func.call @vf_3(%vf2, %sum) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  %vf4 = func.call @vf_4(%vf3, %prod, %eb) {hivm.vector_function} : (tensor<128xf32>, f32, f32) -> tensor<128xf32>
  return %vf4 : tensor<128xf32>
}

// A view of `bufferization.alloc_tensor` storage may be written through a
// memref in between, so a readback of that view must not move either.

// CHECK-LABEL: func.func @no_sink_from_alloc_tensor_view
// CHECK: bufferization.alloc_tensor
// CHECK: tensor.extract_slice
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_1
// CHECK: memref.store
// CHECK: call @vf_3
func.func @no_sink_from_alloc_tensor_view(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %buf = bufferization.alloc_tensor() : tensor<8xf32>
  %mem = bufferization.to_memref %buf : memref<8xf32>
  %slice = tensor.extract_slice %buf[0] [1] [1] : tensor<8xf32> to tensor<1xf32>
  %x = tensor.extract %slice[%c0] : tensor<1xf32>
  %vf0 = func.call @vf_1(%arg0) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  memref.store %cst, %mem[%c0] : memref<8xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// Views derived from a value-semantic producer stay immutable, so the readback
// of a view keeps sinking.  These lock the recursive cases below the seed.

// CHECK-LABEL: func.func @sink_extract_slice_view
// CHECK: call @vf_7
// CHECK: tensor.extract_slice
// CHECK: call @vf_1
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_3
func.func @sink_extract_slice_view(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %vf = func.call @vf_7() {hivm.vector_function} : () -> tensor<4xf32>
  %slice = tensor.extract_slice %vf[0] [1] [1] : tensor<4xf32> to tensor<1xf32>
  %x = tensor.extract %slice[%c0] : tensor<1xf32>
  %vf0 = func.call @vf_1(%arg0) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// CHECK-LABEL: func.func @sink_cast_view
// CHECK: call @vf_8
// CHECK: tensor.cast
// CHECK: call @vf_1
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_3
func.func @sink_cast_view(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %vf = func.call @vf_8() {hivm.vector_function} : () -> tensor<?xf32>
  %cast = tensor.cast %vf : tensor<?xf32> to tensor<1xf32>
  %x = tensor.extract %cast[%c0] : tensor<1xf32>
  %vf0 = func.call @vf_1(%arg0) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// CHECK-LABEL: func.func @sink_collapse_shape_view
// CHECK: call @vf_9
// CHECK: tensor.collapse_shape
// CHECK: call @vf_1
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_3
func.func @sink_collapse_shape_view(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %vf = func.call @vf_9() {hivm.vector_function} : () -> tensor<1x1xf32>
  %collapsed = tensor.collapse_shape %vf[[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  %x = tensor.extract %collapsed[%c0] : tensor<1xf32>
  %vf0 = func.call @vf_1(%arg0) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// CHECK-LABEL: func.func @sink_expand_shape_view
// CHECK: call @vf_0
// CHECK: tensor.expand_shape
// CHECK: call @vf_1
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_3
func.func @sink_expand_shape_view(%arg0: tensor<1xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %vf = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %expanded = tensor.expand_shape %vf[[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
  %x = tensor.extract %expanded[%c0, %c0] : tensor<1x1xf32>
  %vf0 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// `arith.select` forwards the storage of one of its operands without copying,
// so a readback through a `to_tensor` view must not move.

// CHECK-LABEL: func.func @no_sink_select_of_to_tensor
// CHECK: bufferization.to_tensor
// CHECK: arith.select
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_1
// CHECK: memref.store
// CHECK: call @vf_3
func.func @no_sink_select_of_to_tensor(%mem: memref<1xf32>, %cond: i1, %fallback: tensor<1xf32>, %arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %view = bufferization.to_tensor %mem : memref<1xf32>
  %sel = arith.select %cond, %view, %fallback : tensor<1xf32>
  %x = tensor.extract %sel[%c0] : tensor<1xf32>
  %vf0 = func.call @vf_1(%arg0) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  memref.store %cst, %mem[%c0] : memref<1xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// Both operands of the select are value-semantic, so the readback keeps sinking.

// CHECK-LABEL: func.func @sink_select_of_vector_calls
// CHECK: call @vf_0
// CHECK: call @vf_5
// CHECK: arith.select
// CHECK: call @vf_1
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_3
func.func @sink_select_of_vector_calls(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %cond: i1, %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %a = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %b = func.call @vf_5(%arg1) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %sel = arith.select %cond, %a, %b : tensor<1xf32>
  %x = tensor.extract %sel[%c0] : tensor<1xf32>
  %vf0 = func.call @vf_1(%arg2) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %vf1 = func.call @vf_3(%vf0, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %vf1 : tensor<128xf32>
}

// A scalar branch that only feeds a later consumer must not block the prefix:
// the extract lands before its earliest consumer and `addf` stays in place.

// CHECK-LABEL: func.func @sink_prefix_before_earliest_consumer
// CHECK: call @vf_0
// CHECK: call @vf_1
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: call @vf_3
// CHECK-NEXT: arith.addf
// CHECK-NEXT: call @vf_3
func.func @sink_prefix_before_earliest_consumer(%arg0: tensor<1xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.000000e+00 : f32
  %vf0 = func.call @vf_0(%arg0) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  %x = tensor.extract %vf0[%c0] : tensor<1xf32>
  %vf1 = func.call @vf_1(%arg1) {hivm.vector_function} : (tensor<128xf32>) -> tensor<128xf32>
  %c1 = func.call @vf_3(%vf1, %x) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  %y = arith.addf %x, %one : f32
  %c2 = func.call @vf_3(%c1, %y) {hivm.vector_function} : (tensor<128xf32>, f32) -> tensor<128xf32>
  return %c2 : tensor<128xf32>
}
