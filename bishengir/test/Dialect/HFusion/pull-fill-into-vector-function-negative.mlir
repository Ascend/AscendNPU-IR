// RUN: bishengir-opt %s --hfusion-pull-slice-into-vector-function --split-input-file --verify-each | FileCheck %s

// A second, unmarked call still prevents specialization of the callee.
// CHECK-LABEL: func.func @shared_vf(
// CHECK: return %arg0
func.func @shared_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @shared_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @shared_vf(%[[F]]) {hivm.vector_function}
// CHECK: call @shared_vf(%arg0)
func.func @shared_caller(%dst: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %a = call @shared_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  %b = call @shared_vf(%dst) : (tensor<f32>) -> tensor<f32>
  return %a, %b : tensor<f32>, tensor<f32>
}

// -----

// Skip a fill rewrite when the callee contains the masked reshape pattern that
// the canonicalizer can incorrectly forward. The fill and call remain intact.
// CHECK-LABEL: func.func @masked_reshape_vf(
// CHECK: vector.transfer_read %{{.*}}, %{{.*}}, %{{.*}}
func.func @masked_reshape_vf(%dst: tensor<f32>, %mask: vector<1xi1>)
    -> vector<1xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %seed = arith.constant dense<7.0> : vector<f32>
  %padding = arith.constant 9.0 : f32
  %init = vector.transfer_write %seed, %dst[] : vector<f32>, tensor<f32>
  %expanded = tensor.expand_shape %init [] output_shape [1]
      : tensor<f32> into tensor<1xf32>
  %read = vector.transfer_read %expanded[%c0], %padding, %mask
      {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
  return %read : vector<1xf32>
}

// CHECK-LABEL: func.func @masked_reshape_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @masked_reshape_vf(%[[F]], %{{.*}})
func.func @masked_reshape_caller(%dst: tensor<f32>, %mask: vector<1xi1>)
    -> vector<1xf32> {
  %c = arith.constant 7.0 : f32
  %filled = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %read = call @masked_reshape_vf(%filled, %mask) {hivm.vector_function}
      : (tensor<f32>, vector<1xi1>) -> vector<1xf32>
  return %read : vector<1xf32>
}

// -----

// Even compatible multiple callers are intentionally outside the initial scope.
func.func @same_seed_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @same_seed_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @same_seed_vf(%[[F]])
// CHECK: call @same_seed_vf(%[[F]])
func.func @same_seed_caller(%dst: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %a = call @same_seed_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  %b = call @same_seed_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %a, %b : tensor<f32>, tensor<f32>
}

// -----

// Address-taken VFs have additional symbol uses besides their direct call.
func.func @address_taken_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
func.func @get_vf() -> ((tensor<f32>) -> tensor<f32>) {
  %f = func.constant @address_taken_vf : (tensor<f32>) -> tensor<f32>
  return %f : (tensor<f32>) -> tensor<f32>
}
// CHECK-LABEL: func.func @address_taken_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @address_taken_vf(%[[F]])
func.func @address_taken_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @address_taken_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Do not infer zero (or any reduction identity) for a dynamic seed.
func.func @dynamic_seed_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @dynamic_seed_caller(
// CHECK: %[[F:.*]] = linalg.fill ins(%arg1
// CHECK: call @dynamic_seed_vf(%[[F]])
func.func @dynamic_seed_caller(%dst: tensor<f32>, %seed: f32) -> tensor<f32> {
  %f = linalg.fill ins(%seed : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @dynamic_seed_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Non-zero-rank tensors are rejected.
func.func @rank_one_vf(%dst: tensor<1xf32>) -> tensor<1xf32> attributes {hivm.vector_function} {
  return %dst : tensor<1xf32>
}
// CHECK-LABEL: func.func @rank_one_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @rank_one_vf(%[[F]])
func.func @rank_one_caller(%dst: tensor<1xf32>) -> tensor<1xf32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  %r = call @rank_one_vf(%f) {hivm.vector_function} : (tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}

// i1 remains unsupported until its device lowering is validated.
func.func @i1_vf(%dst: tensor<i1>) -> tensor<i1> attributes {hivm.vector_function} {
  return %dst : tensor<i1>
}
// CHECK-LABEL: func.func @i1_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @i1_vf(%[[F]])
func.func @i1_caller(%dst: tensor<i1>) -> tensor<i1> {
  %c = arith.constant true
  %f = linalg.fill ins(%c : i1) outs(%dst : tensor<i1>) -> tensor<i1>
  %r = call @i1_vf(%f) {hivm.vector_function} : (tensor<i1>) -> tensor<i1>
  return %r : tensor<i1>
}

// -----

func.func @encoded_vf(%dst: tensor<f32, "test">) -> tensor<f32, "test">
    attributes {hivm.vector_function} {
  return %dst : tensor<f32, "test">
}
// CHECK-LABEL: func.func @encoded_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @encoded_vf(%[[F]])
func.func @encoded_caller(%dst: tensor<f32, "test">) -> tensor<f32, "test"> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32, "test">) -> tensor<f32, "test">
  %r = call @encoded_vf(%f) {hivm.vector_function} : (tensor<f32, "test">) -> tensor<f32, "test">
  return %r : tensor<f32, "test">
}

// -----

// A buffer fill's write is observable. The tensor rewrite must not sink it.
func.func @buffer_vf(%dst: memref<f32>) attributes {hivm.vector_function} {
  return
}
// CHECK-LABEL: func.func @buffer_caller(
// CHECK: linalg.fill
// CHECK: call @buffer_vf(%arg0)
func.func @buffer_caller(%dst: memref<f32>) {
  %c = arith.constant 0.0 : f32
  linalg.fill ins(%c : f32) outs(%dst : memref<f32>)
  call @buffer_vf(%dst) {hivm.vector_function} : (memref<f32>) -> ()
  return
}

// -----

// Both call and callee must be marked as VFs.
func.func @ordinary_func(%dst: tensor<f32>) -> tensor<f32> {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @ordinary_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @ordinary_func(%[[F]])
func.func @ordinary_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @ordinary_func(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

func.func @unmarked_call_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @unmarked_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @unmarked_call_vf(%[[F]])
func.func @unmarked_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @unmarked_call_vf(%f) : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

func.func @simt_vf(%dst: tensor<f32>) -> tensor<f32>
    attributes {hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @simt_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @simt_vf(%[[F]])
func.func @simt_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @simt_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

func.func @mixed_call_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @mixed_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @mixed_call_vf(%[[F]])
func.func @mixed_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @mixed_call_vf(%f) {hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<MIX>}
      : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

func.func private @external_vf(tensor<f32>) -> tensor<f32> attributes {hivm.vector_function}
// CHECK-LABEL: func.func @external_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @external_vf(%[[F]])
func.func @external_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @external_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Do not chase a running accumulator through a caller's loop iter_arg.
func.func @running_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @running_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[F]])
// CHECK: call @running_vf(%[[ACC]])
func.func @running_caller(%dst: tensor<f32>, %n: index) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f) -> tensor<f32> {
    %v = func.call @running_vf(%acc) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
    scf.yield %v : tensor<f32>
  }
  return %r : tensor<f32>
}

// -----

// No need to reconstruct an unused argument (SimplifyVFArgs can remove it).
func.func @unused_vf(%unused: tensor<f32>, %other: tensor<f32>) -> tensor<f32>
    attributes {hivm.vector_function} {
  return %other : tensor<f32>
}
// CHECK-LABEL: func.func @unused_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @unused_vf(%[[F]], %arg1)
func.func @unused_caller(%dst: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @unused_vf(%f, %other) {hivm.vector_function}
      : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// The scalar can have a different type from the result: linalg.fill performs
// the conversion in its region. Do not reinterpret the scalar's attribute bits.
func.func @cast_seed_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}
// CHECK-LABEL: func.func @cast_seed_caller(
// CHECK: %[[F:.*]] = linalg.fill ins(%{{.*}} : f64)
// CHECK: call @cast_seed_vf(%[[F]])
func.func @cast_seed_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 1.0000000000000002 : f64
  %f = linalg.fill ins(%c : f64) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @cast_seed_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Integer tensors are unsupported until their downstream lowering is validated.
func.func @i1_rejected_vf(%dst: tensor<i1>) -> tensor<i1>
    attributes {hivm.vector_function} {
  return %dst : tensor<i1>
}
// CHECK-LABEL: func.func @i1_rejected_caller(
// CHECK: linalg.fill
// CHECK: call @i1_rejected_vf
func.func @i1_rejected_caller(%dst: tensor<i1>) -> tensor<i1> {
  %c = arith.constant true
  %f = linalg.fill ins(%c : i1) outs(%dst : tensor<i1>) -> tensor<i1>
  %r = call @i1_rejected_vf(%f) {hivm.vector_function}
      : (tensor<i1>) -> tensor<i1>
  return %r : tensor<i1>
}

// Index values are also rejected until their lowering is explicitly validated.
func.func @index_rejected_vf(%dst: tensor<index>) -> tensor<index>
    attributes {hivm.vector_function} {
  return %dst : tensor<index>
}
// CHECK-LABEL: func.func @index_rejected_caller(
// CHECK: linalg.fill
// CHECK: call @index_rejected_vf
func.func @index_rejected_caller(%dst: tensor<index>) -> tensor<index> {
  %c = arith.constant 0 : index
  %f = linalg.fill ins(%c : index) outs(%dst : tensor<index>) -> tensor<index>
  %r = call @index_rejected_vf(%f) {hivm.vector_function}
      : (tensor<index>) -> tensor<index>
  return %r : tensor<index>
}


// -----

// Dynamic extents are outside the full rank-zero initialization case.
func.func @dynamic_rank_one_vf(%dst: tensor<?xf32>) -> tensor<?xf32> attributes {hivm.vector_function} {
  return %dst : tensor<?xf32>
}
// CHECK-LABEL: func.func @dynamic_rank_one_caller(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @dynamic_rank_one_vf(%[[F]])
func.func @dynamic_rank_one_caller(%dst: tensor<?xf32>) -> tensor<?xf32> {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
  %r = call @dynamic_rank_one_vf(%f) {hivm.vector_function} : (tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// A self-recursive VF is not an outlined single-caller specialization target.
// CHECK-LABEL: func.func @recursive_vf(
// CHECK: %[[F:.*]] = linalg.fill
// CHECK: call @recursive_vf(%[[F]])
func.func @recursive_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  %c = arith.constant 0.0 : f32
  %f = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @recursive_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}
