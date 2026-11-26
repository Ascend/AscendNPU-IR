// RUN: bishengir-opt %s --fuse-reduction-into-loop --verify-each -o %t.default
// RUN: bishengir-opt %s --fuse-reduction-into-loop="mode=zero-only" --verify-each -o %t.zero
// RUN: diff %t.default %t.zero
// RUN: FileCheck %s --check-prefix=ZEROONLY < %t.zero
// RUN: bishengir-opt %s --fuse-reduction-into-loop="mode=late" --verify-each --dump-pass-pipeline -o %t.late 2> %t.late.dump
// RUN: grep '^builtin.module(' %t.late.dump | tee %t.late.pipeline | FileCheck %s --check-prefix=PIPE-LATE
// RUN: FileCheck %s --check-prefix=LATE < %t.late
// RUN: bishengir-opt %s --pass-pipeline="$(cat %t.late.pipeline)" --verify-each -o %t.late.replay
// RUN: diff %t.late %t.late.replay
// RUN: bishengir-opt %t.late --fuse-reduction-into-loop="mode=late" --verify-each -o %t.late.twice
// RUN: diff %t.late %t.late.twice

// PIPE-LATE: builtin.module(func.func(fuse-reduction-into-loop{mode=late}))

// Only the output shape must be static; the reduction extent may be dynamic.
// ZEROONLY-LABEL: func.func @dynamic_extent(
// ZEROONLY: scf.for {{.*}} -> (tensor<?x8xf32>)
// LATE-LABEL: func.func @dynamic_extent(
// LATE: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<8xf32>)
// LATE: linalg.reduce ins(%arg1 : tensor<?x8xf32>)
// LATE: scf.yield
// LATE: %[[SEED:.*]] = linalg.reduce ins(%arg0 : tensor<?x8xf32>) outs(%arg2 : tensor<8xf32>)
// LATE: arith.addf %[[SEED]], %[[LOOP]]{{.*}} : tensor<8xf32>
func.func @dynamic_extent(%seed: tensor<?x8xf32>, %tile: tensor<?x8xf32>,
                          %init: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iv = %c0 to %n step %c1
      iter_args(%acc = %seed) -> tensor<?x8xf32> {
    %next = arith.addf %acc, %tile : tensor<?x8xf32>
    scf.yield %next : tensor<?x8xf32>
  }
  %result = linalg.reduce ins(%loop : tensor<?x8xf32>)
      outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<8xf32>
}

// Empty reductions keep the output initializer.
// ZEROONLY-LABEL: func.func @empty_extent(
// ZEROONLY: scf.for {{.*}} -> (tensor<0x8xf32>)
// LATE-LABEL: func.func @empty_extent(
// LATE: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<8xf32>)
// LATE: linalg.reduce ins(%arg1 : tensor<0x8xf32>)
// LATE: scf.yield
// LATE: %[[SEED:.*]] = linalg.reduce ins(%arg0 : tensor<0x8xf32>) outs(%arg2 : tensor<8xf32>)
// LATE: arith.addf %[[SEED]], %[[LOOP]]{{.*}} : tensor<8xf32>
func.func @empty_extent(%seed: tensor<0x8xf32>, %tile: tensor<0x8xf32>,
                        %init: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %loop = scf.for %iv = %c0 to %n step %c1
      iter_args(%acc = %seed) -> tensor<0x8xf32> {
    %next = arith.addf %acc, %tile : tensor<0x8xf32>
    scf.yield %next : tensor<0x8xf32>
  }
  %result = linalg.reduce ins(%loop : tensor<0x8xf32>)
      outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<8xf32>
}
// Reuse the original preloop tensor SSA value for the late reduction.
// ZEROONLY-LABEL: func.func @snapshot_seed(
// ZEROONLY: scf.for {{.*}} -> (tensor<4x8xf32>)
// LATE-LABEL: func.func @snapshot_seed(
// LATE: %[[SEED:.*]] = bufferization.to_tensor %arg0
// LATE-NOT: bufferization.to_tensor
// LATE: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<8xf32>)
// LATE: memref.store
// LATE: scf.yield
// LATE-NOT: bufferization.to_tensor
// LATE: %[[RED:.*]] = linalg.reduce ins(%[[SEED]] : tensor<4x8xf32>) outs(%arg2 : tensor<8xf32>)
// LATE: arith.addf %[[RED]], %[[LOOP]]{{.*}} : tensor<8xf32>
func.func @snapshot_seed(%source: memref<4x8xf32>, %tile: tensor<4x8xf32>,
                         %init: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  %seed = bufferization.to_tensor %source : memref<4x8xf32>
  %loop = scf.for %iv = %c0 to %n step %c1
      iter_args(%acc = %seed) -> tensor<4x8xf32> {
    memref.store %one, %source[%c0, %c0] : memref<4x8xf32>
    %next = arith.addf %acc, %tile : tensor<4x8xf32>
    scf.yield %next : tensor<4x8xf32>
  }
  %result = linalg.reduce ins(%loop : tensor<4x8xf32>)
      outs(%init : tensor<8xf32>) dimensions = [0]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<8xf32>
}

// A nonzero dense constant is not a zero seed, even for a unit reduction axis.
// ZEROONLY-LABEL: func.func @nonzero_constant(
// ZEROONLY: scf.for {{.*}} -> (tensor<2x1xf32>)
// LATE-LABEL: func.func @nonzero_constant(
// LATE: %[[SEED:.*]] = arith.constant dense<{{.*}}> : tensor<2x1xf32>
// LATE: %[[LOOP:.*]] = scf.for {{.*}} -> (tensor<2xf32>)
// LATE: linalg.reduce ins(%arg0 : tensor<2x1xf32>)
// LATE: scf.yield
// LATE: %[[RED:.*]] = linalg.reduce ins(%[[SEED]] : tensor<2x1xf32>) outs(%arg1 : tensor<2xf32>)
// LATE: arith.addf %[[RED]], %[[LOOP]]{{.*}} : tensor<2xf32>
func.func @nonzero_constant(%tile: tensor<2x1xf32>, %init: tensor<2xf32>,
                            %n: index) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seed = arith.constant dense<[[1.0], [3.0]]> : tensor<2x1xf32>
  %loop = scf.for %iv = %c0 to %n step %c1
      iter_args(%acc = %seed) -> tensor<2x1xf32> {
    %next = arith.addf %acc, %tile : tensor<2x1xf32>
    scf.yield %next : tensor<2x1xf32>
  }
  %result = linalg.reduce ins(%loop : tensor<2x1xf32>)
      outs(%init : tensor<2xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
    %sum = arith.addf %in, %acc : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<2xf32>
}

// The inner and outer reductions both keep their seeds.
// ZEROONLY-LABEL: func.func @nested(
// ZEROONLY: scf.for {{.*}} -> (tensor<2x8xf32>)
// ZEROONLY: scf.for {{.*}} -> (tensor<2x2x8xf32>)
// LATE-LABEL: func.func @nested(
// LATE: scf.for {{.*}} -> (tensor<8xf32>)
// LATE: scf.for {{.*}} -> (tensor<2x8xf32>)
// LATE: linalg.reduce ins(%arg2 : tensor<2x2x8xf32>)
// LATE: linalg.reduce ins(%arg1 : tensor<2x2x8xf32>)
// LATE: linalg.reduce ins(%{{.*}} : tensor<2x8xf32>)
// LATE: linalg.reduce ins(%arg0 : tensor<2x8xf32>)
func.func @nested(%outer_seed: tensor<2x8xf32>,
                  %inner_seed: tensor<2x2x8xf32>,
                  %tile: tensor<2x2x8xf32>, %n: index) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init_inner = arith.constant dense<0.0> : tensor<2x8xf32>
  %init_outer = arith.constant dense<0.0> : tensor<8xf32>
  %outer = scf.for %i = %c0 to %n step %c1
      iter_args(%acc = %outer_seed) -> tensor<2x8xf32> {
    %inner = scf.for %j = %c0 to %n step %c1
        iter_args(%inner_acc = %inner_seed) -> tensor<2x2x8xf32> {
      %next_inner = arith.addf %inner_acc, %tile : tensor<2x2x8xf32>
      scf.yield %next_inner : tensor<2x2x8xf32>
    }
    %red = linalg.reduce ins(%inner : tensor<2x2x8xf32>)
        outs(%init_inner : tensor<2x8xf32>) dimensions = [0]
        (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }
    %next = arith.addf %acc, %red : tensor<2x8xf32>
    scf.yield %next : tensor<2x8xf32>
  }
  %result = linalg.reduce ins(%outer : tensor<2x8xf32>)
      outs(%init_outer : tensor<8xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
    %sum = arith.addf %in, %init : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<8xf32>
}
