// REQUIRES: asserts
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ra=true enable-ar=true" | FileCheck %s
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ra=true enable-ar=true" --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops allow-unknown-ops" -o /dev/null
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ra=true enable-ar=true" -debug-only=tree-reduce-v2 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ra=true enable-ar=false ascend-style-ra=false" -debug-only=tree-reduce-v2 2>&1 | FileCheck %s --check-prefix=FALLBACK
//
// Multi-stage Ascend-style AR/RA coverage. With f32 compute, the vector
// length is 64, so R > 64 exercises the shared radix-16 folding builders.
// AR folds vectors along R; RA folds rows and keeps every A lane, merging
// directly with the accumulator without a horizontal reduction.

// ---------------------------------------------------------------------------
// AR: R=1024 has one main stage (16 vectors -> 1).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ar_1024
// CHECK: vector.multi_reduction <add>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ar_1024(%arg0: tensor<1x1024xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<1xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<1x1024xf32>, vector<1x1024xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<1xf32>, vector<1xf32>
    %red = vector.multi_reduction <add>, %src, %old [1]
        : vector<1x1024xf32> to vector<1xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<1xf32>, tensor<1xf32>
    scf.yield %w : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}

// ---------------------------------------------------------------------------
// AR: R=16384 has two main stages (256 -> 16 -> 1 vectors).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ar_16384
// CHECK: vector.multi_reduction <add>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ar_16384(%arg0: tensor<1x16384xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<1xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<1x16384xf32>, vector<1x16384xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<1xf32>, vector<1xf32>
    %red = vector.multi_reduction <add>, %src, %old [1]
        : vector<1x16384xf32> to vector<1xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<1xf32>, tensor<1xf32>
    scf.yield %w : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}

// ---------------------------------------------------------------------------
// AR: R=16448 = 256*64 + 64, so the one-vector tail folds into the head first.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ar_16448
// CHECK: vector.multi_reduction <add>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ar_16448(%arg0: tensor<1x16448xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<1xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<1x16448xf32>, vector<1x16448xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<1xf32>, vector<1xf32>
    %red = vector.multi_reduction <add>, %src, %old [1]
        : vector<1x16448xf32> to vector<1xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<1xf32>, tensor<1xf32>
    scf.yield %w : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}

// ---------------------------------------------------------------------------
// AR suffix stage in a multi-axis reduction uses the same shared builder.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ar_suffix_16448
// CHECK: tensor.collapse_shape {{.*}} into tensor<4x16448xf32>
// CHECK: vector.multi_reduction <add>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ar_suffix_16448(%arg0: tensor<2x2x16448xf32>,
                           %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<2xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<2x2x16448xf32>, vector<2x2x16448xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<2xf32>, vector<2xf32>
    %red = vector.multi_reduction <add>, %src, %old [0, 2]
        : vector<2x2x16448xf32> to vector<2xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<2xf32>, tensor<2xf32>
    scf.yield %w : tensor<2xf32>
  }
  return %0 : tensor<2xf32>
}

// ---------------------------------------------------------------------------
// RA: short axes and the power-of-two boundary.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_1
// CHECK-NOT: tensor.empty
// CHECK: vector.transfer_read {{.*}} : tensor<1x68xf32>, vector<64xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_1(%arg0: tensor<1x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<1x68xf32>, vector<1x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<1x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// CHECK-LABEL: func.func @ra_15
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_15(%arg0: tensor<15x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<15x68xf32>, vector<15x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<15x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// CHECK-LABEL: func.func @ra_16
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_16(%arg0: tensor<16x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<16x68xf32>, vector<16x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<16x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// CHECK-LABEL: func.func @ra_17
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_17(%arg0: tensor<17x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<17x68xf32>, vector<17x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<17x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// CHECK-LABEL: func.func @ra_255
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_255(%arg0: tensor<255x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<255x68xf32>, vector<255x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<255x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// CHECK-LABEL: func.func @ra_256
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_256(%arg0: tensor<256x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<256x68xf32>, vector<256x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<256x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// ---------------------------------------------------------------------------
// RA: R=257 folds a one-row tail, then 256 -> 16 -> 1 rows.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_257
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_257(%arg0: tensor<257x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<257x68xf32>, vector<257x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<257x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// ---------------------------------------------------------------------------
// RA: R=4097 folds a one-row tail, then 4096 -> 256 -> 16 -> 1 rows.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_4097
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_4097(%arg0: tensor<4097x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<4097x68xf32>, vector<4097x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<4097x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// ---------------------------------------------------------------------------
// RA: R=273 has a 17-row tail folded into the 256-row head.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_273
// CHECK-NOT: tensor.empty
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_273(%arg0: tensor<273x68xf32>, %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<273x68xf32>, vector<273x68xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<273x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// ---------------------------------------------------------------------------
// RA: f16 input with f32 compute and f32 accumulator.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_257_f16
// CHECK: tensor.empty() : tensor<256x68xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_257_f16(%arg0: tensor<257x68xf16>,
                       %arg1: tensor<68xf32>) -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  %f32zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf32> {
    %raw = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<257x68xf16>, vector<257x68xf16>
    %src = arith.extf %raw : vector<257x68xf16> to vector<257x68xf32>
    %old = vector.transfer_read %acc[%c0], %f32zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<257x68xf32> to vector<68xf32>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf32>, tensor<68xf32>
    scf.yield %w : tensor<68xf32>
  }
  return %0 : tensor<68xf32>
}

// ---------------------------------------------------------------------------
// RA: f16 input and f16 accumulator; conversion happens once and the result
// is truncated once at the accumulator merge.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_257_f16_acc
// CHECK: tensor.empty() : tensor<256x68xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: arith.truncf {{.*}} : vector<64xf32> to vector<64xf16>
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_257_f16_acc(%arg0: tensor<257x68xf16>,
                           %arg1: tensor<68xf16>) -> tensor<68xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<68xf16> {
    %src = vector.transfer_read %arg0[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<257x68xf16>, vector<257x68xf16>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<68xf16>, vector<68xf16>
    %red = vector.multi_reduction <add>, %src, %old [0]
        : vector<257x68xf16> to vector<68xf16>
    %w = vector.transfer_write %red, %acc[%c0] {in_bounds = [true]}
        : vector<68xf16>, tensor<68xf16>
    scf.yield %w : tensor<68xf16>
  }
  return %0 : tensor<68xf16>
}

// ---------------------------------------------------------------------------
// RA: BatchedRA3D [2, 257, 68] after a suffix R=2 stage.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @ra_batched_257
// CHECK: tensor.empty() : tensor<34952xf32>
// CHECK-NOT: tensor.empty() : tensor<2x256x68xf32>
// CHECK: vector.multi_reduction
// CHECK-NOT: vector.multi_reduction
// CHECK: return

func.func @ra_batched_257(%arg0: tensor<2x257x68x2xf32>,
                          %arg1: tensor<2x68xf32>) -> tensor<2x68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> tensor<2x68xf32> {
    %src = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %zero
        {in_bounds = [true, true, true, true]}
        : tensor<2x257x68x2xf32>, vector<2x257x68x2xf32>
    %old = vector.transfer_read %acc[%c0, %c0], %zero {in_bounds = [true, true]}
        : tensor<2x68xf32>, vector<2x68xf32>
    %red = vector.multi_reduction <add>, %src, %old [1, 3]
        : vector<2x257x68x2xf32> to vector<2x68xf32>
    %w = vector.transfer_write %red, %acc[%c0, %c0] {in_bounds = [true, true]}
        : vector<2x68xf32>, tensor<2x68xf32>
    scf.yield %w : tensor<2x68xf32>
  }
  return %0 : tensor<2x68xf32>
}

// ---------------------------------------------------------------------------
// Debug schedule checks.
// ---------------------------------------------------------------------------

// DEBUG: buildARReductionStage: outer=1 dimR=1024 mainR=1024 tailR=0 base=16 folds=4 mainTimes=1 tailFolds=0 tailVecs=0 numLeaves=1
// DEBUG: buildARReductionStage: stage 0: 16 -> 1 vectors
// DEBUG: buildARReductionStage: outer=1 dimR=16384 mainR=16384 tailR=0 base=256 folds=8 mainTimes=2 tailFolds=0 tailVecs=0 numLeaves=1
// DEBUG: buildARReductionStage: stage 0: 256 -> 16 vectors
// DEBUG: buildARReductionStage: stage 1: 16 -> 1 vectors
// DEBUG: buildARReductionStage: outer=1 dimR=16448 mainR=16384 tailR=64 base=256 folds=8 mainTimes=2 tailFolds=0 tailVecs=1 numLeaves=1
// DEBUG: buildARReductionStage: outer=4 dimR=16448 mainR=16384 tailR=64 base=256 folds=8 mainTimes=2 tailFolds=0 tailVecs=1 numLeaves=1
// DEBUG: buildBatchedRAReductionStage: outer=1 dimR=257 inner=68 mainR=256 tailR=1 folds=8 mainTimes=2 tailFolds=0 numLeaves=1 layout=RA2D
// DEBUG: buildBatchedRAReductionStage: stage 0: 256 -> 16
// DEBUG: buildBatchedRAReductionStage: stage 1: 16 -> 1
// DEBUG: buildBatchedRAReductionStage: outer=1 dimR=4097 inner=68 mainR=4096 tailR=1 folds=12 mainTimes=3 tailFolds=0 numLeaves=1 layout=RA2D
// DEBUG: buildBatchedRAReductionStage: stage 0: 4096 -> 256
// DEBUG: buildBatchedRAReductionStage: stage 1: 256 -> 16
// DEBUG: buildBatchedRAReductionStage: stage 2: 16 -> 1
// DEBUG: buildBatchedRAReductionStage: outer=1 dimR=273 inner=68 mainR=256 tailR=17 folds=8 mainTimes=2 tailFolds=0 numLeaves=1 layout=RA2D
// DEBUG: buildBatchedRAReductionStage: outer=2 dimR=257 inner=68 mainR=256 tailR=1 folds=8 mainTimes=2 tailFolds=0 numLeaves=1 layout=BatchedRA3D reuseInput=1
// DEBUG: buildBatchedRAReductionStage: stage 0: 256 -> 16
// DEBUG: buildBatchedRAReductionStage: stage 1: 16 -> 1

// FALLBACK: buildRA: dimR=257 dimA=68
// FALLBACK-NOT: rewriteSingleAxisRA
// FALLBACK: return
