// RUN: bishengir-opt %s --hfusion-pull-slice-into-vector-function --split-input-file --verify-each | FileCheck %s

// Preserve the output argument and all function/call/argument/result attributes.
// A public symbol is normal for an outlined, compiler-internal VF.
// CHECK-LABEL: func.func @zero_vf(
// CHECK-SAME: %[[DST:.*]]: tensor<f32> {test.arg}
// CHECK-SAME: -> (tensor<f32> {test.result})
// CHECK-SAME: attributes {hivm.vector_function, no_inline, test.func}
// CHECK: %[[SEED:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
// CHECK: %[[INIT:.*]] = vector.transfer_write %[[SEED]], %[[DST]][] : vector<f32>, tensor<f32>
// CHECK: return %[[INIT]] : tensor<f32>
func.func @zero_vf(%dst: tensor<f32> {test.arg}) -> (tensor<f32> {test.result})
    attributes {hivm.vector_function, no_inline, test.func} {
  return %dst : tensor<f32>
}

// CHECK-LABEL: func.func @zero_caller(
// CHECK-SAME: %[[DST:.*]]: tensor<f32>
// CHECK-NOT: linalg.fill
// CHECK: %[[R:.*]] = call @zero_vf(%[[DST]]) {hivm.vector_function, no_inline, test.call}
// CHECK-NOT: linalg.fill
// CHECK: return %[[R]]
func.func @zero_caller(%dst: tensor<f32>) -> tensor<f32> {
  %c = arith.constant 0.0 : f32
  %filled = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @zero_vf(%filled) {hivm.vector_function, no_inline, test.call}
      : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @rank_zero_types_vf(
// CHECK-DAG: arith.constant dense<1.000000e+00> : vector<f16>
// CHECK-DAG: arith.constant dense<2.000000e+00> : vector<bf16>
// CHECK-DAG: arith.constant dense<3.000000e+00> : vector<f32>
func.func @rank_zero_types_vf(%f16: tensor<f16>, %bf16: tensor<bf16>, %f32: tensor<f32>)
    -> (tensor<f16>, tensor<bf16>, tensor<f32>)
    attributes {hivm.vector_function} {
  return %f16, %bf16, %f32 : tensor<f16>, tensor<bf16>, tensor<f32>
}
// CHECK-LABEL: func.func @rank_zero_types_caller(
// CHECK-NOT: linalg.fill
// CHECK: call @rank_zero_types_vf(%{{.*}}, %{{.*}}, %{{.*}})
func.func @rank_zero_types_caller(%dst16: tensor<f16>, %dst_bf16: tensor<bf16>, %dst32: tensor<f32>)
    -> (tensor<f16>, tensor<bf16>, tensor<f32>) {
  %c16 = arith.constant 1.0 : f16
  %c_bf16 = arith.constant 2.0 : bf16
  %c32 = arith.constant 3.0 : f32
  %f16 = linalg.fill ins(%c16 : f16) outs(%dst16 : tensor<f16>) -> tensor<f16>
  %f_bf16 = linalg.fill ins(%c_bf16 : bf16) outs(%dst_bf16 : tensor<bf16>) -> tensor<bf16>
  %f32 = linalg.fill ins(%c32 : f32) outs(%dst32 : tensor<f32>) -> tensor<f32>
  %r:3 = call @rank_zero_types_vf(%f16, %f_bf16, %f32) {hivm.vector_function}
      : (tensor<f16>, tensor<bf16>, tensor<f32>)
      -> (tensor<f16>, tensor<bf16>, tensor<f32>)
  return %r#0, %r#1, %r#2 : tensor<f16>, tensor<bf16>, tensor<f32>
}

// -----

// Exact seeds: max identity, min identity, negative zero, and a NaN payload.
// Multiple eligible operands must each be reconstructed in the callee.
// CHECK-LABEL: func.func private @seeds_vf(
// CHECK-SAME: %[[A:.*]]: tensor<f32>, %[[B:.*]]: tensor<f32>, %[[C:.*]]: tensor<f32>, %[[D:.*]]: tensor<f32>
// CHECK-DAG: %[[NINF:.*]] = arith.constant dense<0xFF800000> : vector<f32>
// CHECK-DAG: %[[PINF:.*]] = arith.constant dense<0x7F800000> : vector<f32>
// CHECK-DAG: %[[NEGZERO:.*]] = arith.constant dense<-0.000000e+00> : vector<f32>
// CHECK-DAG: %[[NAN:.*]] = arith.constant dense<0x7FC12345> : vector<f32>
// CHECK-DAG: %[[WA:.*]] = vector.transfer_write %[[NINF]], %[[A]][]
// CHECK-DAG: %[[WB:.*]] = vector.transfer_write %[[PINF]], %[[B]][]
// CHECK-DAG: %[[WC:.*]] = vector.transfer_write %[[NEGZERO]], %[[C]][]
// CHECK-DAG: %[[WD:.*]] = vector.transfer_write %[[NAN]], %[[D]][]
// CHECK: return %[[WA]], %[[WB]], %[[WC]], %[[WD]]
func.func private @seeds_vf(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>)
    -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
    attributes {hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>} {
  return %a, %b, %c, %d : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func.func @seeds_caller(
// CHECK-SAME: %[[A:.*]]: tensor<f32>, %[[B:.*]]: tensor<f32>, %[[C:.*]]: tensor<f32>, %[[D:.*]]: tensor<f32>
// CHECK-NOT: linalg.fill
// CHECK: call @seeds_vf(%[[A]], %[[B]], %[[C]], %[[D]])
// CHECK: return
func.func @seeds_caller(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>)
    -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  %ninf = arith.constant 0xFF800000 : f32
  %pinf = arith.constant 0x7F800000 : f32
  %negzero = arith.constant -0.0 : f32
  %nan = arith.constant 0x7FC12345 : f32
  %fa = linalg.fill ins(%ninf : f32) outs(%a : tensor<f32>) -> tensor<f32>
  %fb = linalg.fill ins(%pinf : f32) outs(%b : tensor<f32>) -> tensor<f32>
  %fc = linalg.fill ins(%negzero : f32) outs(%c : tensor<f32>) -> tensor<f32>
  %fd = linalg.fill ins(%nan : f32) outs(%d : tensor<f32>) -> tensor<f32>
  %r:4 = call @seeds_vf(%fa, %fb, %fc, %fd) {hivm.vector_function}
      : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
      -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
  return %r#0, %r#1, %r#2, %r#3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

// The old fill remains necessary for its other consumer. Its destination's
// original value is also observable; do not replace either tensor SSA value.
// CHECK-LABEL: func.func @shared_fill_vf(
// CHECK: arith.constant dense<3.500000e+00> : vector<f32>
// CHECK: vector.transfer_write
func.func @shared_fill_vf(%dst: tensor<f32>) -> tensor<f32>
    attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}

// CHECK-LABEL: func.func @shared_fill_caller(
// CHECK-SAME: %[[DST:.*]]: tensor<f32>
// CHECK: %[[FILLED:.*]] = linalg.fill
// CHECK: %[[R:.*]] = call @shared_fill_vf(%[[DST]])
// CHECK: return %[[R]], %[[FILLED]], %[[DST]]
func.func @shared_fill_caller(%dst: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
  %c = arith.constant 3.5 : f32
  %filled = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @shared_fill_vf(%filled) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r, %filled, %dst : tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

// The same fill can feed two distinct arguments of the sole call.
// CHECK-LABEL: func.func @duplicate_vf(
// CHECK-SAME: %[[A:.*]]: tensor<f32>, %[[B:.*]]: tensor<f32>
// CHECK-DAG: vector.transfer_write %{{.*}}, %[[A]][]
// CHECK-DAG: vector.transfer_write %{{.*}}, %[[B]][]
// CHECK: return
func.func @duplicate_vf(%a: tensor<f32>, %b: tensor<f32>) -> (tensor<f32>, tensor<f32>)
    attributes {hivm.vector_function} {
  return %a, %b : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func.func @duplicate_caller(
// CHECK-SAME: %[[DST:.*]]: tensor<f32>
// CHECK-NOT: linalg.fill
// CHECK: call @duplicate_vf(%[[DST]], %[[DST]])
func.func @duplicate_caller(%dst: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %c = arith.constant 1.0 : f32
  %filled = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r:2 = call @duplicate_vf(%filled, %filled) {hivm.vector_function}
      : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return %r#0, %r#1 : tensor<f32>, tensor<f32>
}

// -----

// One syntactic caller may execute repeatedly and conditionally. The seed is
// recreated on each actual invocation, rather than hoisted into the caller.
// CHECK-LABEL: func.func @conditional_vf(
// CHECK: arith.constant dense<2.000000e+00> : vector<f32>
// CHECK: vector.transfer_write
func.func @conditional_vf(%dst: tensor<f32>) -> tensor<f32>
    attributes {hivm.vector_function} {
  return %dst : tensor<f32>
}

// CHECK-LABEL: func.func @conditional_caller(
// CHECK: scf.for
// CHECK: scf.if
// CHECK-NOT: linalg.fill
// CHECK: call @conditional_vf(%{{.*}})
// CHECK: else
// CHECK: scf.yield
func.func @conditional_caller(%dst: tensor<f32>, %n: index, %cond: i1) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seed = arith.constant 2.0 : f32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %dst) -> tensor<f32> {
    %next = scf.if %cond -> tensor<f32> {
      %filled = linalg.fill ins(%seed : f32) outs(%acc : tensor<f32>) -> tensor<f32>
      %v = func.call @conditional_vf(%filled) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
      scf.yield %v : tensor<f32>
    } else {
      scf.yield %acc : tensor<f32>
    }
    scf.yield %next : tensor<f32>
  }
  return %r : tensor<f32>
}

// -----

// The entry initializer must dominate every return, including a CFG exit that
// bypasses the computation entirely.
// CHECK-LABEL: func.func @cfg_vf(
// CHECK-SAME: %[[DST:.*]]: tensor<f32>
// CHECK: %[[INIT:.*]] = vector.transfer_write %{{.*}}, %[[DST]][]
// CHECK: cf.cond_br
// CHECK: return %[[INIT]]
// CHECK: tensor.extract %[[INIT]][]
func.func @cfg_vf(%dst: tensor<f32>, %cond: i1) -> tensor<f32>
    attributes {hivm.vector_function} {
  cf.cond_br %cond, ^early, ^work
^early:
  return %dst : tensor<f32>
^work:
  %s = tensor.extract %dst[] : tensor<f32>
  %next = arith.addf %s, %s : f32
  %w = tensor.insert %next into %dst[] : tensor<f32>
  return %w : tensor<f32>
}
// CHECK-LABEL: func.func @cfg_caller(
// CHECK-NOT: linalg.fill
// CHECK: call @cfg_vf(%arg0, %arg1)
func.func @cfg_caller(%dst: tensor<f32>, %cond: i1) -> tensor<f32> {
  %seed = arith.constant 7.0 : f32
  %f = linalg.fill ins(%seed : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @cfg_vf(%f, %cond) {hivm.vector_function} : (tensor<f32>, i1) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Compose with the existing slice pull on a different operand of the same VF.
// CHECK-LABEL: func.func @slice_and_fill_vf(
// CHECK-SAME: %[[SRC:.*]]: tensor<128xf32>, %[[DST:.*]]: tensor<f32>
// CHECK-DAG: tensor.extract_slice %[[SRC]][32] [64] [1]
// CHECK-DAG: vector.transfer_write %{{.*}}, %[[DST]][]
// CHECK: return
func.func @slice_and_fill_vf(%input: tensor<64xf32>, %dst: tensor<f32>)
    -> (tensor<64xf32>, tensor<f32>) attributes {hivm.vector_function} {
  return %input, %dst : tensor<64xf32>, tensor<f32>
}

// CHECK-LABEL: func.func @slice_and_fill_caller(
// CHECK-SAME: %[[SRC:.*]]: tensor<128xf32>, %[[DST:.*]]: tensor<f32>
// CHECK-NOT: linalg.fill
// CHECK-NOT: tensor.extract_slice
// CHECK: call @slice_and_fill_vf(%[[SRC]], %[[DST]])
func.func @slice_and_fill_caller(%input: tensor<128xf32>, %dst: tensor<f32>)
    -> (tensor<64xf32>, tensor<f32>) {
  %c = arith.constant 0.0 : f32
  %filled = linalg.fill ins(%c : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %slice = tensor.extract_slice %input[32] [64] [1] : tensor<128xf32> to tensor<64xf32>
  %r:2 = call @slice_and_fill_vf(%slice, %filled) {hivm.vector_function}
      : (tensor<64xf32>, tensor<f32>) -> (tensor<64xf32>, tensor<f32>)
  return %r#0, %r#1 : tensor<64xf32>, tensor<f32>
}
