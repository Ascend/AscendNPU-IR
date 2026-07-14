// RUN: bishengir-opt %s -loop-invariant-promotion -split-input-file -verify-each | FileCheck %s

// CHECK-LABEL: func.func @hoist_loop_carried_acc
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<64x16xf32>, %[[INIT:.*]]: tensor<16xf32>
//       CHECK:   %[[MASK:.*]] = vector.create_mask %{{.*}} : vector<16xi1>
//       CHECK:   %[[ACC0:.*]] = vector.transfer_read %[[INIT]][%{{.*}}], %[[PAD:.*]], %[[MASK]]
//  CHECK-SAME:       : tensor<16xf32>, vector<16xf32>
//       CHECK:   %[[PADV:.*]] = vector.broadcast %[[PAD]] : f32 to vector<16xf32>
//       CHECK:   %[[LOOP:.*]]:2 = scf.for
//  CHECK-SAME:       iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %[[ACC0]])
//  CHECK-SAME:       -> (tensor<16xf32>, vector<16xf32>)
//       CHECK:     %[[BLEND:.*]] = arith.select %[[MASK]], %[[ACC]], %[[PADV]]
//       CHECK:     %[[NEW:.*]] = arith.addf %[[BLEND]], %{{.*}} : vector<16xf32>
//   CHECK-NOT:     vector.transfer_write
//       CHECK:     scf.yield %{{.*}}, %[[NEW]] : tensor<16xf32>, vector<16xf32>
//       CHECK:   }
//       CHECK:   %[[RES:.*]] = vector.transfer_write %[[LOOP]]#1, %[[INIT]][%{{.*}}], %[[MASK]]
//  CHECK-SAME:       : vector<16xf32>, tensor<16xf32>
//       CHECK:   return %[[RES]] : tensor<16xf32>
func.func @hoist_loop_carried_acc(%arg0: tensor<64x16xf32>, %init: tensor<16xf32>)
    -> tensor<16xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<16xf32>) {
    %slice = tensor.extract_slice %arg0[%i, 0] [1, 16] [1, 1] : tensor<64x16xf32> to tensor<16xf32>
    %mask = vector.create_mask %c16 : vector<16xi1>
    %in = vector.transfer_read %slice[%c0], %cst, %mask {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %acc_v = vector.transfer_read %acc[%c0], %cst, %mask {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %new = arith.addf %acc_v, %in : vector<16xf32>
    %w = vector.transfer_write %new, %acc[%c0], %mask {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    scf.yield %w : tensor<16xf32>
  }
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @two_rmw
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<64x16xf32>, %[[INIT:.*]]: tensor<16xf32>
//       CHECK:   %[[ACC0:.*]] = vector.transfer_read %[[INIT]][
//       CHECK:   %[[LOOP:.*]]:2 = scf.for
//  CHECK-SAME:       iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %[[ACC0]])
//  CHECK-SAME:       -> (tensor<16xf32>, vector<16xf32>)
//       CHECK:     %[[N0:.*]] = arith.addf %[[ACC]], %{{.*}} : vector<16xf32>
//       CHECK:     %[[N1:.*]] = arith.mulf %[[N0]], %{{.*}} : vector<16xf32>
//   CHECK-NOT:     vector.transfer_write
//       CHECK:     scf.yield %{{.*}}, %[[N1]] : tensor<16xf32>, vector<16xf32>
//       CHECK:   }
//       CHECK:   %[[RES:.*]] = vector.transfer_write %[[LOOP]]#1, %[[INIT]][
//       CHECK:   return %[[RES]]
func.func @two_rmw(%arg0: tensor<64x16xf32>, %init: tensor<16xf32>)
    -> tensor<16xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<16xf32>) {
    %in = vector.transfer_read %init[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %v0 = vector.transfer_read %acc[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %n0 = arith.addf %v0, %in : vector<16xf32>
    %w0 = vector.transfer_write %n0, %acc[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    %v1 = vector.transfer_read %w0[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %n1 = arith.mulf %v1, %in : vector<16xf32>
    %w1 = vector.transfer_write %n1, %w0[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    scf.yield %w1 : tensor<16xf32>
  }
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @deadend
//  CHECK-SAME:     %[[INIT:.*]]: tensor<16xf32>, %[[D:.*]]: vector<16xf32>
//       CHECK:   %[[R0:.*]] = vector.transfer_read %[[INIT]][
//       CHECK:   %[[N0:.*]] = arith.addf %[[R0]], %[[D]] : vector<16xf32>
//       CHECK:   %[[LOOP:.*]]:2 = scf.for
//  CHECK-SAME:       iter_args(%{{.*}} = %[[INIT]], %[[S:.*]] = %[[D]])
//       CHECK:     %[[S2:.*]] = arith.addf %[[S]], %[[N0]] : vector<16xf32>
//   CHECK-NOT:     vector.transfer_read
//   CHECK-NOT:     vector.transfer_write
//       CHECK:     scf.yield %{{.*}}, %[[S2]]
//       CHECK:   }
//       CHECK:   return %[[INIT]], %[[LOOP]]#1
func.func @deadend(%init: tensor<16xf32>, %d: vector<16xf32>)
    -> (tensor<16xf32>, vector<16xf32>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r:2 = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init, %s = %d)
      -> (tensor<16xf32>, vector<16xf32>) {
    %v0 = vector.transfer_read %acc[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %n0 = arith.addf %v0, %d : vector<16xf32>
    %w0 = vector.transfer_write %n0, %acc[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    %v1 = vector.transfer_read %w0[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %s2 = arith.addf %s, %v1 : vector<16xf32>
    scf.yield %acc, %s2 : tensor<16xf32>, vector<16xf32>
  }
  return %r#0, %r#1 : tensor<16xf32>, vector<16xf32>
}

// -----

// CHECK-LABEL: func.func @sliced_two_rmw
//  CHECK-SAME:     %[[INIT:.*]]: tensor<2x8xf32>, %[[D:.*]]: vector<8xf32>
//       CHECK:   %[[SL0:.*]] = tensor.extract_slice %[[INIT]][0, 0] [1, 8] [1, 1]
//       CHECK:   %[[ACC0:.*]] = vector.transfer_read %[[SL0]][
//       CHECK:   %[[LOOP:.*]]:2 = scf.for
//  CHECK-SAME:       iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %[[ACC0]])
//  CHECK-SAME:       -> (tensor<2x8xf32>, vector<8xf32>)
//       CHECK:     %[[N0:.*]] = arith.addf %[[ACC]], %[[D]] : vector<8xf32>
//       CHECK:     %[[N1:.*]] = arith.addf %[[N0]], %[[D]] : vector<8xf32>
//   CHECK-NOT:     vector.transfer_write
//   CHECK-NOT:     tensor.insert_slice
//       CHECK:     scf.yield %{{.*}}, %[[N1]] : tensor<2x8xf32>, vector<8xf32>
//       CHECK:   }
//       CHECK:   %[[SL1:.*]] = tensor.extract_slice %[[INIT]][0, 0] [1, 8] [1, 1]
//       CHECK:   %[[W:.*]] = vector.transfer_write %[[LOOP]]#1, %[[SL1]][
//       CHECK:   %[[RES:.*]] = tensor.insert_slice %[[W]] into %[[INIT]][0, 0] [1, 8] [1, 1]
//       CHECK:   return %[[RES]] : tensor<2x8xf32>
func.func @sliced_two_rmw(%init: tensor<2x8xf32>, %d: vector<8xf32>)
    -> tensor<2x8xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<2x8xf32>) {
    %s0 = tensor.extract_slice %acc[0, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
    %v0 = vector.transfer_read %s0[%c0], %cst {in_bounds = [true]} : tensor<8xf32>, vector<8xf32>
    %n0 = arith.addf %v0, %d : vector<8xf32>
    %w0 = vector.transfer_write %n0, %s0[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    %acc1 = tensor.insert_slice %w0 into %acc[0, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<2x8xf32>
    %s1 = tensor.extract_slice %acc1[0, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
    %v1 = vector.transfer_read %s1[%c0], %cst {in_bounds = [true]} : tensor<8xf32>, vector<8xf32>
    %n1 = arith.addf %v1, %d : vector<8xf32>
    %w1 = vector.transfer_write %n1, %s1[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    %acc2 = tensor.insert_slice %w1 into %acc1[0, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<2x8xf32>
    scf.yield %acc2 : tensor<2x8xf32>
  }
  return %r : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: func.func @nested_rmw
//  CHECK-SAME:     %[[INIT:.*]]: tensor<4x2x8xf32>, %[[D:.*]]: vector<8xf32>
//       CHECK:   %[[E0:.*]] = tensor.extract_slice %[[INIT]][1, 0, 0] [1, 2, 8] [1, 1, 1]
//       CHECK:   %[[E1:.*]] = tensor.extract_slice %[[E0]][1, 0] [1, 8] [1, 1]
//       CHECK:   %[[ACC0:.*]] = vector.transfer_read %[[E1]][
//       CHECK:   %[[LOOP:.*]]:2 = scf.for
//  CHECK-SAME:       iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %[[ACC0]])
//  CHECK-SAME:       -> (tensor<4x2x8xf32>, vector<8xf32>)
//       CHECK:     %[[N:.*]] = arith.addf %[[ACC]], %[[D]] : vector<8xf32>
//   CHECK-NOT:     vector.transfer_write
//   CHECK-NOT:     tensor.insert_slice
//       CHECK:     scf.yield %{{.*}}, %[[N]]
//       CHECK:   }
//       CHECK:   %[[WE0:.*]] = tensor.extract_slice %[[INIT]][1, 0, 0] [1, 2, 8] [1, 1, 1]
//       CHECK:   %[[WE1:.*]] = tensor.extract_slice %[[WE0]][1, 0] [1, 8] [1, 1]
//       CHECK:   %[[W:.*]] = vector.transfer_write %[[LOOP]]#1, %[[WE1]][
//       CHECK:   %[[I0:.*]] = tensor.insert_slice %[[W]] into %[[WE0]][1, 0] [1, 8] [1, 1]
//       CHECK:   %[[RES:.*]] = tensor.insert_slice %[[I0]] into %[[INIT]][1, 0, 0] [1, 2, 8] [1, 1, 1]
//       CHECK:   return %[[RES]]
func.func @nested_rmw(%init: tensor<4x2x8xf32>, %d: vector<8xf32>)
    -> tensor<4x2x8xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<4x2x8xf32>) {
    %s1 = tensor.extract_slice %acc[1, 0, 0] [1, 2, 8] [1, 1, 1] : tensor<4x2x8xf32> to tensor<2x8xf32>
    %s2 = tensor.extract_slice %s1[1, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
    %v = vector.transfer_read %s2[%c0], %cst {in_bounds = [true]} : tensor<8xf32>, vector<8xf32>
    %n = arith.addf %v, %d : vector<8xf32>
    %w = vector.transfer_write %n, %s2[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    %i1 = tensor.insert_slice %w into %s1[1, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<2x8xf32>
    %acc1 = tensor.insert_slice %i1 into %acc[1, 0, 0] [1, 2, 8] [1, 1, 1] : tensor<2x8xf32> into tensor<4x2x8xf32>
    scf.yield %acc1 : tensor<4x2x8xf32>
  }
  return %r : tensor<4x2x8xf32>
}

// -----

// CHECK-LABEL: func.func @sliced_masked_rmw
//  CHECK-SAME:     %[[INIT:.*]]: tensor<2x8xf32>, %[[D:.*]]: vector<8xf32>
//       CHECK:   %[[MASK:.*]] = vector.create_mask %{{.*}} : vector<8xi1>
//       CHECK:   %[[SL0:.*]] = tensor.extract_slice %[[INIT]][1, 0] [1, 8] [1, 1]
//       CHECK:   %[[ACC0:.*]] = vector.transfer_read %[[SL0]][%{{.*}}], %[[PAD:.*]], %[[MASK]]
//       CHECK:   %[[PADV:.*]] = vector.broadcast %[[PAD]] : f32 to vector<8xf32>
//       CHECK:   %[[LOOP:.*]]:2 = scf.for
//  CHECK-SAME:       iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %[[ACC0]])
//       CHECK:     %[[BLEND:.*]] = arith.select %[[MASK]], %[[ACC]], %[[PADV]]
//       CHECK:     %[[N0:.*]] = arith.addf %[[BLEND]], %[[D]] : vector<8xf32>
//       CHECK:     scf.yield %{{.*}}, %[[N0]]
//       CHECK:   }
//       CHECK:   %[[SL1:.*]] = tensor.extract_slice %[[INIT]][1, 0] [1, 8] [1, 1]
//       CHECK:   %[[W:.*]] = vector.transfer_write %[[LOOP]]#1, %[[SL1]][%{{.*}}], %[[MASK]]
//       CHECK:   %[[RES:.*]] = tensor.insert_slice %[[W]] into %[[INIT]][1, 0] [1, 8] [1, 1]
//       CHECK:   return %[[RES]]
func.func @sliced_masked_rmw(%init: tensor<2x8xf32>, %d: vector<8xf32>)
    -> tensor<2x8xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %c8 : vector<8xi1>
  %r = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<2x8xf32>) {
    %s0 = tensor.extract_slice %acc[1, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
    %v0 = vector.transfer_read %s0[%c0], %cst, %mask {in_bounds = [true]} : tensor<8xf32>, vector<8xf32>
    %n0 = arith.addf %v0, %d : vector<8xf32>
    %w0 = vector.transfer_write %n0, %s0[%c0], %mask {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    %acc1 = tensor.insert_slice %w0 into %acc[1, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<2x8xf32>
    scf.yield %acc1 : tensor<2x8xf32>
  }
  return %r : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: func.func @no_hoist_variant_slice_offset
//       CHECK:   scf.for
//  CHECK-SAME:       -> (tensor<256xi1>)
//       CHECK:     tensor.extract_slice %{{.*}}[%{{.*}}] [64] [1]
//       CHECK:     vector.transfer_write
//       CHECK:     tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}] [64] [1]
//       CHECK:     scf.yield
func.func @no_hoist_variant_slice_offset(%init: tensor<256xi1>, %in: tensor<256xi32>, %b: i32)
    -> tensor<256xi1> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c0i = arith.constant 0 : i32
  %bv = vector.broadcast %b : i32 to vector<64xi32>
  %r = scf.for %i = %c0 to %c256 step %c64 iter_args(%acc = %init) -> (tensor<256xi1>) {
    %si = tensor.extract_slice %in[%i] [64] [1] : tensor<256xi32> to tensor<64xi32>
    %sa = tensor.extract_slice %acc[%i] [64] [1] : tensor<256xi1> to tensor<64xi1>
    %v = vector.transfer_read %si[%c0], %c0i {in_bounds = [true]} : tensor<64xi32>, vector<64xi32>
    %c = arith.cmpi slt, %v, %bv : vector<64xi32>
    %w = vector.transfer_write %c, %sa[%c0] {in_bounds = [true]} : vector<64xi1>, tensor<64xi1>
    %acc1 = tensor.insert_slice %w into %acc[%i] [64] [1] : tensor<64xi1> into tensor<256xi1>
    scf.yield %acc1 : tensor<256xi1>
  }
  return %r : tensor<256xi1>
}

// -----

// CHECK-LABEL: func.func @two_accumulators
//  CHECK-SAME:     %[[A:.*]]: tensor<16xf32>, %[[B:.*]]: tensor<16xf32>, %[[D:.*]]: vector<16xf32>
//   CHECK-DAG:   %[[RA0:.*]] = vector.transfer_read %[[A]][
//   CHECK-DAG:   %[[RB0:.*]] = vector.transfer_read %[[B]][
//       CHECK:   %[[LOOP:.*]]:4 = scf.for
//  CHECK-SAME:       -> (tensor<16xf32>, tensor<16xf32>, vector<16xf32>, vector<16xf32>)
//       CHECK:     arith.addf
//       CHECK:     arith.mulf
//   CHECK-NOT:     vector.transfer_write
//       CHECK:     scf.yield
//       CHECK:   }
//   CHECK-DAG:   vector.transfer_write %[[LOOP]]#2, %[[A]][
//   CHECK-DAG:   vector.transfer_write %[[LOOP]]#3, %[[B]][
//       CHECK:   return
func.func @two_accumulators(%a: tensor<16xf32>, %b: tensor<16xf32>, %d: vector<16xf32>)
    -> (tensor<16xf32>, tensor<16xf32>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r:2 = scf.for %i = %c0 to %c64 step %c1 iter_args(%pa = %a, %pb = %b)
      -> (tensor<16xf32>, tensor<16xf32>) {
    %va = vector.transfer_read %pa[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %na = arith.addf %va, %d : vector<16xf32>
    %wa = vector.transfer_write %na, %pa[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    %vb = vector.transfer_read %pb[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %nb = arith.mulf %vb, %d : vector<16xf32>
    %wb = vector.transfer_write %nb, %pb[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    scf.yield %wa, %wb : tensor<16xf32>, tensor<16xf32>
  }
  return %r#0, %r#1 : tensor<16xf32>, tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @no_hoist_mismatched_mask
//       CHECK:   scf.for
//  CHECK-SAME:       -> (tensor<16xf32>)
//       CHECK:     vector.transfer_read %{{.*}}, %{{.*}}, %{{.*}}
//       CHECK:     vector.transfer_write
//       CHECK:     scf.yield
func.func @no_hoist_mismatched_mask(%arg0: tensor<64x16xf32>, %init: tensor<16xf32>)
    -> tensor<16xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %maskR = vector.create_mask %c16 : vector<16xi1>
  %maskW = vector.create_mask %c8 : vector<16xi1>
  %0 = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<16xf32>) {
    %acc_v = vector.transfer_read %acc[%c0], %cst, %maskR {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %new = arith.addf %acc_v, %acc_v : vector<16xf32>
    %w = vector.transfer_write %new, %acc[%c0], %maskW {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    scf.yield %w : tensor<16xf32>
  }
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @no_hoist_escape
//       CHECK:   scf.for
//  CHECK-SAME:       -> (tensor<16xf32>)
//       CHECK:     vector.transfer_write
//       CHECK:     "test.sink"
//       CHECK:     scf.yield
func.func @no_hoist_escape(%init: tensor<16xf32>, %d: vector<16xf32>)
    -> tensor<16xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<16xf32>) {
    %v = vector.transfer_read %acc[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %n = arith.addf %v, %d : vector<16xf32>
    %w = vector.transfer_write %n, %acc[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    "test.sink"(%w) : (tensor<16xf32>) -> ()
    scf.yield %w : tensor<16xf32>
  }
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @no_hoist_different_subset
//       CHECK:   scf.for
//  CHECK-SAME:       -> (tensor<2x8xf32>)
//       CHECK:     tensor.extract_slice %{{.*}}[0, 0]
//       CHECK:     tensor.insert_slice %{{.*}} into %{{.*}}[0, 0]
//       CHECK:     tensor.extract_slice %{{.*}}[1, 0]
//       CHECK:     tensor.insert_slice %{{.*}} into %{{.*}}[1, 0]
//       CHECK:     scf.yield
func.func @no_hoist_different_subset(%init: tensor<2x8xf32>, %d: vector<8xf32>,
                                     %x: vector<8xf32>)
    -> tensor<2x8xf32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<2x8xf32>) {
    // row 0 == subset S
    %s0 = tensor.extract_slice %acc[0, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
    %v0 = vector.transfer_read %s0[%c0], %cst {in_bounds = [true]} : tensor<8xf32>, vector<8xf32>
    %n0 = arith.addf %v0, %d : vector<8xf32>
    %w0 = vector.transfer_write %n0, %s0[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    %acc1 = tensor.insert_slice %w0 into %acc[0, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<2x8xf32>
    // row 1 == a different subset
    %s1 = tensor.extract_slice %acc1[1, 0] [1, 8] [1, 1] : tensor<2x8xf32> to tensor<8xf32>
    %w1 = vector.transfer_write %x, %s1[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    %acc2 = tensor.insert_slice %w1 into %acc1[1, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<2x8xf32>
    scf.yield %acc2 : tensor<2x8xf32>
  }
  return %r : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: func.func @no_attr_skip
//       CHECK:   scf.for
//  CHECK-SAME:       -> (tensor<16xf32>)
//       CHECK:     vector.transfer_read
//       CHECK:     vector.transfer_write
//       CHECK:     scf.yield
func.func @no_attr_skip(%init: tensor<16xf32>, %d: vector<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> (tensor<16xf32>) {
    %v = vector.transfer_read %acc[%c0], %cst {in_bounds = [true]} : tensor<16xf32>, vector<16xf32>
    %n = arith.addf %v, %d : vector<16xf32>
    %w = vector.transfer_write %n, %acc[%c0] {in_bounds = [true]} : vector<16xf32>, tensor<16xf32>
    scf.yield %w : tensor<16xf32>
  }
  return %0 : tensor<16xf32>
}
