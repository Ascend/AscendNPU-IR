// RUN: bishengir-opt %s --hfusion-pull-slice-into-vector-function --hfusion-simplify-vf-arg --loop-invariant-subset-hoisting --canonicalize --loop-invariant-promotion --canonicalize --split-input-file --verify-each | FileCheck %s --check-prefix=PROMOTE --implicit-check-not=linalg.fill
// RUN: bishengir-opt %s --hfusion-pull-slice-into-vector-function --hfusion-simplify-vf-arg --loop-invariant-subset-hoisting --canonicalize --loop-invariant-promotion --canonicalize --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --drop-equivalent-buffer-results --canonicalize --split-input-file --verify-each | FileCheck %s --check-prefix=BUFFER --implicit-check-not=linalg.fill --implicit-check-not=memref.alloc --implicit-check-not=memref.copy

// The output pointer remains, but is never loaded. Initialization is once per
// invocation, outside the reduction loop. Its final write also handles n == 0.
// PROMOTE-LABEL: func.func @sum_vf(
// PROMOTE-SAME: %[[INPUT:.*]]: tensor<8xf32>, %[[DST:.*]]: tensor<f32>, %[[N:.*]]: index
// PROMOTE: %[[SEED:.*]] = arith.constant dense<7.000000e+00> : vector<f32>
// PROMOTE-NOT: vector.transfer_read %[[DST]]
// PROMOTE-NOT: vector.transfer_write
// PROMOTE: %[[LOOP:.*]] = scf.for {{.*}} to %[[N]] {{.*}} iter_args(%[[ACC:.*]] = %[[SEED]]) -> (vector<f32>)
// PROMOTE: vector.transfer_read
// PROMOTE: vector.extractelement %[[ACC]][]
// PROMOTE: arith.addf
// PROMOTE-NOT: vector.transfer_write
// PROMOTE: scf.yield
// PROMOTE: %[[RESULT:.*]] = vector.transfer_write %[[LOOP]], %[[DST]][]
// PROMOTE-NEXT: return %[[RESULT]]
// BUFFER-LABEL: func.func @sum_vf(
// BUFFER-SAME: %[[INPUT:.*]]: memref<8xf32>, %[[DST:.*]]: memref<f32>, %[[N:.*]]: index)
// BUFFER: %[[SEED:.*]] = arith.constant dense<7.000000e+00> : vector<f32>
// BUFFER-NOT: vector.transfer_read %[[DST]]
// BUFFER-NOT: vector.transfer_write
// BUFFER: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[SEED]]) -> (vector<f32>)
// BUFFER: vector.transfer_read
// BUFFER-NOT: vector.transfer_write
// BUFFER: scf.yield
// BUFFER: vector.transfer_write %[[LOOP]], %[[DST]][] : vector<f32>, memref<f32>
// BUFFER-NEXT: return
func.func @sum_vf(%input: tensor<8xf32>, %dst: tensor<f32>, %n: index) -> tensor<f32>
    attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %n step %c4 iter_args(%acc = %dst) -> tensor<f32> {
    %slice = tensor.extract_slice %input[%i] [4] [1] : tensor<8xf32> to tensor<4xf32>
    %v = vector.transfer_read %slice[%c0], %zero {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
    %a = vector.transfer_read %acc[], %zero : tensor<f32>, vector<f32>
    %s = vector.extractelement %a[] : vector<f32>
    %partial = vector.reduction <add>, %v : vector<4xf32> into f32
    %next = arith.addf %s, %partial : f32
    %new = vector.broadcast %next : f32 to vector<f32>
    %w = vector.transfer_write %new, %acc[] : vector<f32>, tensor<f32>
    scf.yield %w : tensor<f32>
  }
  return %r : tensor<f32>
}

// PROMOTE-LABEL: func.func @sum_caller(
// PROMOTE-SAME: %[[INPUT:.*]]: tensor<8xf32>, %[[DST:.*]]: tensor<f32>, %[[N:.*]]: index
// PROMOTE: call @sum_vf(%[[INPUT]], %[[DST]], %[[N]])
// BUFFER-LABEL: func.func @sum_caller(
// BUFFER-SAME: %[[INPUT:.*]]: memref<8xf32>, %[[DST:.*]]: memref<f32>, %[[N:.*]]: index)
// BUFFER: call @sum_vf(%[[INPUT]], %[[DST]], %[[N]])
// BUFFER-NEXT: return
func.func @sum_caller(%input: tensor<8xf32>, %dst: tensor<f32>, %n: index) -> tensor<f32> {
  %seed = arith.constant 7.0 : f32
  %filled = linalg.fill ins(%seed : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @sum_vf(%input, %filled, %n) {hivm.vector_function, no_inline}
      : (tensor<8xf32>, tensor<f32>, index) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// PROMOTE-LABEL: func.func @max_vf(
// PROMOTE-SAME: %[[INPUT:.*]]: tensor<8xf32>, %[[DST:.*]]: tensor<f32>, %[[N:.*]]: index
// PROMOTE: %[[SEED:.*]] = arith.constant dense<0xFF800000> : vector<f32>
// PROMOTE-NOT: vector.transfer_read %[[DST]]
// PROMOTE-NOT: vector.transfer_write
// PROMOTE: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[SEED]]) -> (vector<f32>)
// PROMOTE: arith.maximumf
// PROMOTE-NOT: vector.transfer_write
// PROMOTE: scf.yield
// PROMOTE: vector.transfer_write %[[LOOP]], %[[DST]][]
// BUFFER-LABEL: func.func @max_vf(
// BUFFER-SAME: %[[INPUT:.*]]: memref<8xf32>, %[[DST:.*]]: memref<f32>, %[[N:.*]]: index)
// BUFFER: %[[SEED:.*]] = arith.constant dense<0xFF800000> : vector<f32>
// BUFFER-NOT: vector.transfer_read %[[DST]]
// BUFFER-NOT: vector.transfer_write
// BUFFER: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[SEED]]) -> (vector<f32>)
// BUFFER-NOT: vector.transfer_write
// BUFFER: scf.yield
// BUFFER: vector.transfer_write %[[LOOP]], %[[DST]][]
// BUFFER-NEXT: return
func.func @max_vf(%input: tensor<8xf32>, %dst: tensor<f32>, %n: index) -> tensor<f32>
    attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %n step %c4 iter_args(%acc = %dst) -> tensor<f32> {
    %slice = tensor.extract_slice %input[%i] [4] [1] : tensor<8xf32> to tensor<4xf32>
    %v = vector.transfer_read %slice[%c0], %zero {in_bounds = [true]} : tensor<4xf32>, vector<4xf32>
    %a = vector.transfer_read %acc[], %zero : tensor<f32>, vector<f32>
    %s = vector.extractelement %a[] : vector<f32>
    %partial = vector.reduction <maximumf>, %v : vector<4xf32> into f32
    %next = arith.maximumf %s, %partial : f32
    %new = vector.broadcast %next : f32 to vector<f32>
    %w = vector.transfer_write %new, %acc[] : vector<f32>, tensor<f32>
    scf.yield %w : tensor<f32>
  }
  return %r : tensor<f32>
}
// PROMOTE-LABEL: func.func @max_caller(
// PROMOTE: call @max_vf(%arg0, %arg1, %arg2)
// BUFFER-LABEL: func.func @max_caller(
// BUFFER: call @max_vf(%arg0, %arg1, %arg2)
func.func @max_caller(%input: tensor<8xf32>, %dst: tensor<f32>, %n: index) -> tensor<f32> {
  %seed = arith.constant 0xFF800000 : f32
  %filled = linalg.fill ins(%seed : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @max_vf(%input, %filled, %n) {hivm.vector_function, no_inline}
      : (tensor<8xf32>, tensor<f32>, index) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Statically zero-trip loop: a write of the seed must still reach the caller's
// output buffer. Replacing only accumulator reads would lose this case.
// PROMOTE-LABEL: func.func @zero_trip_vf(
// PROMOTE-SAME: %[[DST:.*]]: tensor<f32>
// PROMOTE: %[[SEED:.*]] = arith.constant dense<7.000000e+00> : vector<f32>
// PROMOTE-NOT: scf.for
// PROMOTE: %[[RESULT:.*]] = vector.transfer_write %[[SEED]], %[[DST]][]
// PROMOTE-NEXT: return %[[RESULT]]
// BUFFER-LABEL: func.func @zero_trip_vf(
// BUFFER-SAME: %[[DST:.*]]: memref<f32>)
// BUFFER: %[[SEED:.*]] = arith.constant dense<7.000000e+00> : vector<f32>
// BUFFER-NOT: scf.for
// BUFFER: vector.transfer_write %[[SEED]], %[[DST]][]
// BUFFER-NEXT: return
func.func @zero_trip_vf(%dst: tensor<f32>) -> tensor<f32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %one = arith.constant dense<1.0> : vector<f32>
  %r = scf.for %i = %c0 to %c0 step %c1 iter_args(%acc = %dst) -> tensor<f32> {
    %a = vector.transfer_read %acc[], %zero : tensor<f32>, vector<f32>
    %next = arith.addf %a, %one : vector<f32>
    %w = vector.transfer_write %next, %acc[] : vector<f32>, tensor<f32>
    scf.yield %w : tensor<f32>
  }
  return %r : tensor<f32>
}
// PROMOTE-LABEL: func.func @zero_trip_caller(
// PROMOTE: call @zero_trip_vf(%arg0)
// BUFFER-LABEL: func.func @zero_trip_caller(
// BUFFER: call @zero_trip_vf(%arg0)
func.func @zero_trip_caller(%dst: tensor<f32>) -> tensor<f32> {
  %seed = arith.constant 7.0 : f32
  %f = linalg.fill ins(%seed : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @zero_trip_vf(%f) {hivm.vector_function} : (tensor<f32>) -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// A branch bypassing the computation must observe the seed too. Keep
// initialization at entry, not just in the computation's execution path.
// PROMOTE-LABEL: func.func @early_return_vf(
// PROMOTE-SAME: %[[DST:.*]]: tensor<f32>
// PROMOTE: %[[INIT:.*]] = vector.transfer_write %{{.*}}, %[[DST]][]
// PROMOTE: scf.if
// PROMOTE: scf.yield %[[INIT]]
// BUFFER-LABEL: func.func @early_return_vf(
// BUFFER: vector.transfer_write
// BUFFER: scf.if
// BUFFER: return
func.func @early_return_vf(%dst: tensor<f32>, %cond: i1) -> tensor<f32>
    attributes {hivm.vector_function} {
  %r = scf.if %cond -> tensor<f32> {
    scf.yield %dst : tensor<f32>
  } else {
    %zero = arith.constant 0.0 : f32
    %one = arith.constant dense<1.0> : vector<f32>
    %v = vector.transfer_read %dst[], %zero : tensor<f32>, vector<f32>
    %sum = arith.addf %v, %one : vector<f32>
    %w = vector.transfer_write %sum, %dst[] : vector<f32>, tensor<f32>
    scf.yield %w : tensor<f32>
  }
  return %r : tensor<f32>
}
// PROMOTE-LABEL: func.func @early_return_caller(
// PROMOTE: call @early_return_vf(%arg0, %arg1)
// BUFFER-LABEL: func.func @early_return_caller(
// BUFFER: call @early_return_vf(%arg0, %arg1)
func.func @early_return_caller(%dst: tensor<f32>, %cond: i1) -> tensor<f32> {
  %seed = arith.constant 7.0 : f32
  %f = linalg.fill ins(%seed : f32) outs(%dst : tensor<f32>) -> tensor<f32>
  %r = call @early_return_vf(%f, %cond) {hivm.vector_function}
      : (tensor<f32>, i1) -> tensor<f32>
  return %r : tensor<f32>
}
