// RUN: bishengir-opt --tree-reduce-v2="enable-ar" %s | FileCheck %s

// Tests for the AscendC-style AR folding (buildARReductionStage), which lowers
// reductions with dimR > vlElems as a multi-pass in-place fold mirroring the
// ReduceAR kernel.  With vlElems = 64 for f32:
//
//  * @ar_style_dimR384: dimR = 384 = 256 + 128.  The tail [256, 384) folds
//    into the head one vector at a time (tailVecs = 2), leaving 4 vectors
//    (2^tailFolds, tailFolds = 2) that go through one register tree before
//    the final collapse onto the accumulator.
//
//  * @ar_style_dimR1024: dimR = 1024 = 2^10.  No tail; the single main stage
//    folds 16 vectors into one (mainTimes = 1, tailFolds = 0) and the one
//    surviving vector collapses directly onto the accumulator.

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9579">} {

  // CHECK-LABEL: func.func @ar_style_dimR384(
  func.func @ar_style_dimR384(%arg0: tensor<1x384xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> (tensor<1xf32>) {
      %v = vector.transfer_read %arg0[%c0, %c0], %cst : tensor<1x384xf32>, vector<1x384xf32>
      %accv = vector.transfer_read %acc[%c0], %cst : tensor<1xf32>, vector<1xf32>
      %m = vector.multi_reduction <add>, %v, %accv [1] : vector<1x384xf32> to vector<1xf32>
      %w = vector.transfer_write %m, %acc[%c0] : vector<1xf32>, tensor<1xf32>
      scf.yield %w : tensor<1xf32>
    }
    return %0 : tensor<1xf32>
  }

  // CHECK: vector.broadcast {{.*}} : f32 to vector<64xf32>
  // Stage 0: fold the tail vectors [mainR, dimR) into the head in place,
  // zero-filling the last partial vector's inactive lanes.
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %arg0) -> (tensor<1x384xf32>) {
  // CHECK: scf.for
  // CHECK: arith.muli
  // CHECK: affine.min
  // CHECK: vector.create_mask {{.*}} : vector<64xi1>
  // CHECK: arith.addi
  // CHECK: vector.transfer_read {{.*}} : tensor<1x384xf32>, vector<64xf32>
  // CHECK: vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<1x384xf32>, vector<64xf32> }
  // CHECK: arith.select
  // CHECK: arith.addf {{.*}} : vector<64xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<64xf32>
  // Final fold: the 4 surviving vectors go through one register tree, then
  // collapse onto the accumulator element.
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %arg1) -> (tensor<1xf32>) {
  // CHECK-COUNT-4: vector.transfer_read {{.*}} : tensor<1x384xf32>, vector<64xf32>
  // CHECK-COUNT-3: arith.addf {{.*}} : vector<64xf32>
  // CHECK: vector.shape_cast {{.*}} : vector<64xf32> to vector<1x64xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}} {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<1xf32>, tensor<1xf32>

  // CHECK-LABEL: func.func @ar_style_dimR1024(
  func.func @ar_style_dimR1024(%arg0: tensor<1x1024xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.for %i = %c0 to %c1 step %c1 iter_args(%acc = %arg1) -> (tensor<1xf32>) {
      %v = vector.transfer_read %arg0[%c0, %c0], %cst : tensor<1x1024xf32>, vector<1x1024xf32>
      %accv = vector.transfer_read %acc[%c0], %cst : tensor<1xf32>, vector<1xf32>
      %m = vector.multi_reduction <add>, %v, %accv [1] : vector<1x1024xf32> to vector<1xf32>
      %w = vector.transfer_write %m, %acc[%c0] : vector<1xf32>, tensor<1xf32>
      scf.yield %w : tensor<1xf32>
    }
    return %0 : tensor<1xf32>
  }

  // Main stage: fold 16 vectors into one, carrying the live vector count
  // (16 -> 1) through the loop.
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %arg0, {{.*}} = %c16{{(_[0-9]+)?}}) -> (tensor<1x1024xf32>, index) {
  // CHECK: arith.divui
  // CHECK: arith.muli {{.*}} : index
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK-COUNT-16: vector.transfer_read {{.*}} : tensor<1x1024xf32>, vector<64xf32>
  // CHECK-COUNT-15: arith.addf {{.*}} : vector<64xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<64xf32>
  // Final fold: the single surviving vector collapses directly onto the
  // accumulator element.
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %arg1) -> (tensor<1xf32>) {
  // CHECK: vector.transfer_read {{.*}} : tensor<1x1024xf32>, vector<64xf32>
  // CHECK: vector.shape_cast {{.*}} : vector<64xf32> to vector<1x64xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}} {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>

  // The working tensor uses f32, but the final accumulator remains f16.
  // Derive the accumulator read padding and conversion from its actual type.
  // CHECK-LABEL: func.func @ar_style_dimR1024_f16_acc(
  // CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
  // CHECK: vector.transfer_read {{.*}}, {{.*}} : tensor<1xf16>, vector<1xf16>
  // CHECK: arith.extf {{.*}} : vector<1xf16> to vector<1xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}} {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
  // CHECK: arith.truncf {{.*}} : vector<1xf32> to vector<1xf16>
  func.func @ar_style_dimR1024_f16_acc(
      %arg0: tensor<1x1024xf16>, %arg1: tensor<1xf16>) -> tensor<1xf16>
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16
    %0 = scf.for %i = %c0 to %c1 step %c1
        iter_args(%acc = %arg1) -> tensor<1xf16> {
      %v = vector.transfer_read %arg0[%c0, %c0], %zero
          : tensor<1x1024xf16>, vector<1x1024xf16>
      %accv = vector.transfer_read %acc[%c0], %zero
          : tensor<1xf16>, vector<1xf16>
      %m = vector.multi_reduction <add>, %v, %accv [1]
          : vector<1x1024xf16> to vector<1xf16>
      %w = vector.transfer_write %m, %acc[%c0]
          : vector<1xf16>, tensor<1xf16>
      scf.yield %w : tensor<1xf16>
    }
    return %0 : tensor<1xf16>
  }

  // CHECK-LABEL: func.func @ar_style_dimR1024_bf16_acc(
  // CHECK: arith.extf {{.*}} : vector<64xbf16> to vector<64xf32>
  // CHECK: vector.transfer_read {{.*}}, {{.*}} : tensor<1xbf16>, vector<1xbf16>
  // CHECK: arith.extf {{.*}} : vector<1xbf16> to vector<1xf32>
  // CHECK: arith.truncf {{.*}} : vector<1xf32> to vector<1xbf16>
  func.func @ar_style_dimR1024_bf16_acc(
      %arg0: tensor<1x1024xbf16>, %arg1: tensor<1xbf16>) -> tensor<1xbf16>
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : bf16
    %0 = scf.for %i = %c0 to %c1 step %c1
        iter_args(%acc = %arg1) -> tensor<1xbf16> {
      %v = vector.transfer_read %arg0[%c0, %c0], %zero
          : tensor<1x1024xbf16>, vector<1x1024xbf16>
      %accv = vector.transfer_read %acc[%c0], %zero
          : tensor<1xbf16>, vector<1xbf16>
      %m = vector.multi_reduction <add>, %v, %accv [1]
          : vector<1x1024xbf16> to vector<1xbf16>
      %w = vector.transfer_write %m, %acc[%c0]
          : vector<1xbf16>, tensor<1xbf16>
      scf.yield %w : tensor<1xbf16>
    }
    return %0 : tensor<1xbf16>
  }

  // A one-vector tail crosses the first compact output group boundary.  The
  // generated mask clamps inactive iterations to a legal tail address rather
  // than introducing a vector-valued scf.if.
  // CHECK-LABEL: func.func @ar_style_dimR16448_f16_small_tail(
  // CHECK: tensor.empty() : tensor<1x1024xf32>
  // CHECK: arith.minui
  // CHECK: arith.subi
  // CHECK: arith.maxsi
  // CHECK: arith.minsi
  // CHECK: vector.create_mask {{.*}} : vector<64xi1>
  // CHECK-NOT: scf.if
  // CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<64xf32>
  func.func @ar_style_dimR16448_f16_small_tail(
      %arg0: tensor<1x16448xf16>, %arg1: tensor<1xf32>) -> tensor<1xf32>
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16
    %zero_acc = arith.constant 0.0 : f32
    %0 = scf.for %i = %c0 to %c1 step %c1
        iter_args(%acc = %arg1) -> tensor<1xf32> {
      %v = vector.transfer_read %arg0[%c0, %c0], %zero
          : tensor<1x16448xf16>, vector<1x16448xf16>
      %promoted = arith.extf %v : vector<1x16448xf16> to vector<1x16448xf32>
      %accv = vector.transfer_read %acc[%c0], %zero_acc
          : tensor<1xf32>, vector<1xf32>
      %m = vector.multi_reduction <add>, %promoted, %accv [1]
          : vector<1x16448xf32> to vector<1xf32>
      %w = vector.transfer_write %m, %acc[%c0]
          : vector<1xf32>, tensor<1xf32>
      scf.yield %w : tensor<1xf32>
    }
    return %0 : tensor<1xf32>
  }

  // A promoted AR input with a large tail fuses promotion, tail folding and
  // the first radix-16 stage.  Only the compact 16-vector result is
  // materialized: [2, 16384] f32 would exceed UB together with the input.
  // CHECK-LABEL: func.func @ar_style_dimR31740_f16_compact(
  // CHECK: tensor.empty() : tensor<2x1024xf32>
  // CHECK-NOT: tensor.empty() : tensor<2x16384xf32>
  // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (tensor<2x1024xf32>) {
  // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (tensor<2x1024xf32>) {
  // CHECK: vector.transfer_read {{.*}} : tensor<2x31740xf16>, vector<64xf16>
  // CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
  // CHECK: vector.create_mask {{.*}} : vector<64xi1>
  // CHECK: vector.mask {{.*}} { vector.transfer_read {{.*}} : tensor<2x31740xf16>, vector<64xf16> }
  // CHECK: arith.addf {{.*}} : vector<64xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<64xf32>
  // CHECK-NOT: scf.if
  // The remaining main stage starts with 16 compact vectors and folds them
  // into one before the final horizontal reduction.
  // CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}, {{.*}} = %c16{{(_[0-9]+)?}}) -> (tensor<2x1024xf32>, index) {
  // CHECK-COUNT-16: vector.transfer_read {{.*}} : tensor<2x1024xf32>, vector<64xf32>
  // CHECK-COUNT-15: arith.addf {{.*}} : vector<64xf32>
  // CHECK: vector.multi_reduction <add>, {{.*}} {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
  func.func @ar_style_dimR31740_f16_compact(
      %arg0: tensor<2x31740xf16>, %arg1: tensor<2xf32>) -> tensor<2xf32>
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16
    %zero_acc = arith.constant 0.0 : f32
    %0 = scf.for %i = %c0 to %c1 step %c1
        iter_args(%acc = %arg1) -> tensor<2xf32> {
      %v = vector.transfer_read %arg0[%c0, %c0], %zero
          : tensor<2x31740xf16>, vector<2x31740xf16>
      %promoted = arith.extf %v : vector<2x31740xf16> to vector<2x31740xf32>
      %accv = vector.transfer_read %acc[%c0], %zero_acc
          : tensor<2xf32>, vector<2xf32>
      %m = vector.multi_reduction <add>, %promoted, %accv [1]
          : vector<2x31740xf32> to vector<2xf32>
      %w = vector.transfer_write %m, %acc[%c0]
          : vector<2xf32>, tensor<2xf32>
      scf.yield %w : tensor<2xf32>
    }
    return %0 : tensor<2xf32>
  }
}
