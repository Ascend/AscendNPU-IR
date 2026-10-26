// RUN: bishengir-opt %s --tree-reduce-v2="enable-ar=true enable-ra=true" | FileCheck %s
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ra=true enable-ar=true only-marked=true direct-register-ra=true" | FileCheck %s --check-prefixes=CHECK,MARKED
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ar=true enable-ra=true" --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops allow-unknown-ops" -o %t
// RUN: bishengir-opt %s --tree-reduce-v2="enable-ar=false enable-ra=true" | FileCheck %s --check-prefix=NOAR

// CHECK-LABEL: func.func @separated_axes(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<17x3x6xf32> into tensor<51x6xf32>
// CHECK: tensor.empty() : tensor<51xf32>
// CHECK: vector.create_mask {{.*}} : vector<64xi1>
// CHECK: vector.multi_reduction <add>{{.*}} {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
// CHECK: vector.transfer_write {{.*}} : vector<1xf32>, tensor<51xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<51xf32> into tensor<17x3xf32>
// CHECK-NOT: tensor.empty() : tensor<16x3xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: vector.transfer_read {{.*}} : tensor<17x3xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<3xf32>
// CHECK-NOT: tensor.empty() : tensor<3xf32>
// CHECK: return
func.func @separated_axes(%input: tensor<17x3x6xf32>,
                          %init: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %outer = %c0 to %c1 step %c1
      iter_args(%outer_acc = %init) -> tensor<3xf32> {
    %inner_result = scf.for %inner = %c0 to %c1 step %c1
        iter_args(%acc = %outer_acc) -> tensor<3xf32> {
      %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
          {in_bounds = [true, true, true]}
          : tensor<17x3x6xf32>, vector<17x3x6xf32>
      %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
          : tensor<3xf32>, vector<3xf32>
      %reduced = vector.multi_reduction <add>, %source, %old [0, 2]
          : vector<17x3x6xf32> to vector<3xf32>
      %next = vector.transfer_write %reduced, %acc[%c0]
          {in_bounds = [true]} : vector<3xf32>, tensor<3xf32>
      scf.yield %next : tensor<3xf32>
    }
    scf.yield %inner_result : tensor<3xf32>
  }
  return %result : tensor<3xf32>
}

// Consecutive reduction dimensions form one stage. For [2, 3, 4], reducing
// [1, 2] produces one R=12 stage rather than separate R=4 and R=3 stages.

// CHECK-LABEL: func.func @contiguous_axes(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<2x3x4xf32> into tensor<2x12xf32>
// CHECK: vector.create_mask {{.*}} : vector<64xi1>
// CHECK: vector.multi_reduction <add>{{.*}} [1] : vector<1x64xf32> to vector<1xf32>
// CHECK: vector.create_mask {{.*}} : vector<1xi1>
// CHECK: vector.transfer_read {{.*}} : tensor<2xf32>, vector<1xf32>
// CHECK: arith.addf {{.*}} : vector<1xf32>
// CHECK: vector.transfer_write {{.*}} : vector<1xf32>, tensor<2xf32>
// CHECK-NOT: tensor.empty
// CHECK: return
func.func @contiguous_axes(%input: tensor<2x3x4xf32>,
                           %init: tensor<2xf32>) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<2xf32> {
    %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<2x3x4xf32>, vector<2x3x4xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<2xf32>, vector<2xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [1, 2]
        : vector<2x3x4xf32> to vector<2xf32>
    %next = vector.transfer_write %reduced, %acc[%c0]
        {in_bounds = [true]} : vector<2xf32>, tensor<2xf32>
    scf.yield %next : tensor<2xf32>
  }
  return %result : tensor<2xf32>
}

// CHECK-LABEL: func.func @scalar_output(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<2x3xf32> into tensor<6xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<6xf32> into tensor<1x6xf32>
// CHECK: vector.create_mask {{.*}} : vector<64xi1>
// CHECK-NOT: vector.multi_reduction
// CHECK: vector.transfer_read {{.*}}[]{{.*}} : tensor<f32>, vector<f32>
// CHECK: vector.reduction <add>{{.*}} : vector<64xf32> into f32
// CHECK: vector.transfer_write {{.*}}, {{.*}}[] : vector<f32>, tensor<f32>
// CHECK-NOT: vector.extract
// CHECK-NOT: tensor.empty
// CHECK: return
func.func @scalar_output(%input: tensor<2x3xf32>,
                         %init: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<f32> {
    %source = vector.transfer_read %input[%c0, %c0], %zero
        {in_bounds = [true, true]} : tensor<2x3xf32>, vector<2x3xf32>
    %old = tensor.extract %acc[] : tensor<f32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 1]
        : vector<2x3xf32> to f32
    %next = tensor.insert %reduced into %acc[] : tensor<f32>
    scf.yield %next : tensor<f32>
  }
  return %result : tensor<f32>
}

// All stage intermediates use f32. The original f16 accumulator is extended
// once during the final merge and the result is truncated once before write.

// CHECK-LABEL: func.func @promote_f16(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<2x3x4xf16> into tensor<6x4xf16>
// CHECK: tensor.empty() : tensor<6xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: vector.multi_reduction <add>{{.*}} [1] : vector<1x64xf32> to vector<1xf32>
// CHECK: vector.transfer_write {{.*}} : vector<1xf32>, tensor<6xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<6xf32> into tensor<2x3xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<2x3xf32>, vector<64xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<3xf16>, vector<64xf16>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: arith.truncf {{.*}} : vector<64xf32> to vector<64xf16>
// CHECK: vector.transfer_write {{.*}} : vector<64xf16>, tensor<3xf16>
// CHECK-NOT: tensor.empty() : tensor<3xf32>
// CHECK: return
func.func @promote_f16(%input: tensor<2x3x4xf16>,
                       %init: tensor<3xf16>) -> tensor<3xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<3xf16> {
    %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<2x3x4xf16>, vector<2x3x4xf16>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<3xf16>, vector<3xf16>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 2]
        : vector<2x3x4xf16> to vector<3xf16>
    %next = vector.transfer_write %reduced, %acc[%c0]
        {in_bounds = [true]} : vector<3xf16>, tensor<3xf16>
    scf.yield %next : tensor<3xf16>
  }
  return %result : tensor<3xf16>
}

// Three separated reduction axes produce three tree-reduction stages. This
// also verifies that stage planning is not limited to rank-4 inputs.

// CHECK-LABEL: func.func @rank5_three_groups(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<2x2x2x2x2xf32> into tensor<16x2xf32>
// CHECK: tensor.empty() : tensor<16xf32>
// CHECK: vector.multi_reduction <add>{{.*}} [1] : vector<1x64xf32> to vector<1xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<16xf32> into tensor<2x2x2x2xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<2x2x2x2xf32> into tensor<4x2x2xf32>
// CHECK: tensor.empty() : tensor<8xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<4x2x2xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<8xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<8xf32> into tensor<2x2x2xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<2x2x2xf32> into tensor<2x4xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: vector.transfer_read {{.*}} : tensor<2x4xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<4xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<4xf32> into tensor<2x2xf32>
// CHECK: return
func.func @rank5_three_groups(%input: tensor<2x2x2x2x2xf32>,
                              %init: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<2x2xf32> {
    %source = vector.transfer_read %input[%c0, %c0, %c0, %c0, %c0], %zero
        {in_bounds = [true, true, true, true, true]}
        : tensor<2x2x2x2x2xf32>, vector<2x2x2x2x2xf32>
    %old = vector.transfer_read %acc[%c0, %c0], %zero
        {in_bounds = [true, true]}
        : tensor<2x2xf32>, vector<2x2xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 2, 4]
        : vector<2x2x2x2x2xf32> to vector<2x2xf32>
    %next = vector.transfer_write %reduced, %acc[%c0, %c0]
        {in_bounds = [true, true]}
        : vector<2x2xf32>, tensor<2x2xf32>
    scf.yield %next : tensor<2x2xf32>
  }
  return %result : tensor<2x2xf32>
}

// Conversion to f32 is fused with the first register-tree level. For the
// R=2 stage this directly produces the stage output without an f32 scratch.

// CHECK-LABEL: func.func @fused_convert_strided(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<5x3x2x68xf16> into tensor<15x2x68xf16>
// CHECK: tensor.empty() : tensor<1020xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<1020xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<1020xf32> into tensor<5x3x68xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<5x3x68xf32> into tensor<5x204xf32>
// CHECK-NOT: tensor.empty() : tensor<4x204xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<5x204xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<5x204xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<204xf32> into tensor<3x68xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: return
func.func @fused_convert_strided(%input: tensor<5x3x2x68xf16>,
                                 %init: tensor<3x68xf32>)
    -> tensor<3x68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_zero = arith.constant 0.0 : f16
  %acc_zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<3x68xf32> {
    %raw = vector.transfer_read %input[%c0, %c0, %c0, %c0], %input_zero
        {in_bounds = [true, true, true, true]}
        : tensor<5x3x2x68xf16>, vector<5x3x2x68xf16>
    %source = arith.extf %raw
        : vector<5x3x2x68xf16> to vector<5x3x2x68xf32>
    %old = vector.transfer_read %acc[%c0, %c0], %acc_zero
        {in_bounds = [true, true]}
        : tensor<3x68xf32>, vector<3x68xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 2]
        : vector<5x3x2x68xf32> to vector<3x68xf32>
    %next = vector.transfer_write %reduced, %acc[%c0, %c0]
        {in_bounds = [true, true]}
        : vector<3x68xf32>, tensor<3x68xf32>
    scf.yield %next : tensor<3x68xf32>
  }
  return %result : tensor<3x68xf32>
}

// A one-stage, non-suffix reduction fuses f16-to-f32 conversion, the complete
// register tree, and the accumulator merge. No compute-type output tensor is
// materialized because partialR is one.

// CHECK-LABEL: func.func @fused_convert_final_stage(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<2x3x68xf16> into tensor<6x68xf16>
// CHECK: tensor.empty() : tensor<4x68xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<4x68xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<4x68xf32>, vector<64xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<68xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<68xf32>
// CHECK-NOT: tensor.empty
// CHECK: return
func.func @fused_convert_final_stage(%input: tensor<2x3x68xf16>,
                                     %init: tensor<68xf32>)
    -> tensor<68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_zero = arith.constant 0.0 : f16
  %acc_zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<68xf32> {
    %raw = vector.transfer_read %input[%c0, %c0, %c0], %input_zero
        {in_bounds = [true, true, true]}
        : tensor<2x3x68xf16>, vector<2x3x68xf16>
    %source = arith.extf %raw
        : vector<2x3x68xf16> to vector<2x3x68xf32>
    %old = vector.transfer_read %acc[%c0], %acc_zero {in_bounds = [true]}
        : tensor<68xf32>, vector<68xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 1]
        : vector<2x3x68xf32> to vector<68xf32>
    %next = vector.transfer_write %reduced, %acc[%c0]
        {in_bounds = [true]} : vector<68xf32>, tensor<68xf32>
    scf.yield %next : tensor<68xf32>
  }
  return %result : tensor<68xf32>
}

// R=17 verifies one full 16-register conversion/reduction group, one tail
// register, and the follow-up reduction of the two partial rows.

// CHECK-LABEL: func.func @fused_convert_partial_groups(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<2x3x17x4xf16> into tensor<6x17x4xf16>
// CHECK: tensor.empty() : tensor<24xf32>
// CHECK: tensor.empty() : tensor<6x16x4xf32>
// CHECK: arith.extf {{.*}} : vector<64xf16> to vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<6x16x4xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<24xf32> into tensor<2x3x4xf32>
// CHECK: tensor.collapse_shape {{.*}} tensor<2x3x4xf32> into tensor<2x12xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<2x12xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<12xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<12xf32> into tensor<3x4xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: return
func.func @fused_convert_partial_groups(%input: tensor<2x3x17x4xf16>,
                                        %init: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_zero = arith.constant 0.0 : f16
  %acc_zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<3x4xf32> {
    %raw = vector.transfer_read %input[%c0, %c0, %c0, %c0], %input_zero
        {in_bounds = [true, true, true, true]}
        : tensor<2x3x17x4xf16>, vector<2x3x17x4xf16>
    %source = arith.extf %raw
        : vector<2x3x17x4xf16> to vector<2x3x17x4xf32>
    %old = vector.transfer_read %acc[%c0, %c0], %acc_zero
        {in_bounds = [true, true]}
        : tensor<3x4xf32>, vector<3x4xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 2]
        : vector<2x3x17x4xf32> to vector<3x4xf32>
    %next = vector.transfer_write %reduced, %acc[%c0, %c0]
        {in_bounds = [true, true]}
        : vector<3x4xf32>, tensor<3x4xf32>
    scf.yield %next : tensor<3x4xf32>
  }
  return %result : tensor<3x4xf32>
}

// This mirrors the recovered logical shape of triton_sum_5D_dim0_dim4:
// [15, 8, 7] reduced on [0, 2]. The suffix R=7 stage uses contiguous AR
// loads, while the remaining R=15 stage keeps the strided implementation.

// CHECK-LABEL: func.func @suffix_then_strided(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<15x8x7xf32> into tensor<120x7xf32>
// CHECK: tensor.empty() : tensor<120xf32>
// CHECK: vector.multi_reduction <add>{{.*}} [1] : vector<1x64xf32> to vector<1xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<120xf32> into tensor<15x8xf32>
// CHECK-NOT: tensor.empty() : tensor<8x8xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: vector.transfer_read {{.*}} : tensor<15x8xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_read {{.*}} : tensor<8xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<8xf32>
// CHECK-NOT: tensor.empty() : tensor<8xf32>
// CHECK: return
func.func @suffix_then_strided(%input: tensor<15x8x7xf32>,
                               %init: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<8xf32> {
    %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<15x8x7xf32>, vector<15x8x7xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<8xf32>, vector<8xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 2]
        : vector<15x8x7xf32> to vector<8xf32>
    %next = vector.transfer_write %reduced, %acc[%c0]
        {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
    scf.yield %next : tensor<8xf32>
  }
  return %result : tensor<8xf32>
}

// A first multi-axis RA stage may reuse its f32 input when the input is dead
// after the reduction. The [17, 2] reduction group becomes R=34, whose
// Ascend-style fold would otherwise allocate a [32, 4] working tensor.

// CHECK-LABEL: func.func @multi_ra_reuse_input(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<17x2x4xf32> into tensor<34x4xf32>
// CHECK-NOT: tensor.empty() : tensor<32x4xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: return
func.func @multi_ra_reuse_input(%input: tensor<17x2x4xf32>,
                                %init: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<4xf32> {
    %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<17x2x4xf32>, vector<17x2x4xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<4xf32>, vector<4xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 1]
        : vector<17x2x4xf32> to vector<4xf32>
    %next = vector.transfer_write %reduced, %acc[%c0]
        {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  }
  return %result : tensor<4xf32>
}

// An input use outside the rewritten reduction keeps the separate working
// tensor, so the in-place fold cannot overwrite the live source value.

// CHECK-LABEL: func.func @multi_ra_preserve_live_input(
// CHECK: tensor.collapse_shape %arg0 {{.*}} tensor<17x2x4xf32> into tensor<34x4xf32>
// CHECK: tensor.empty() : tensor<32x4xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: tensor.extract %arg0[
// CHECK: return
func.func @multi_ra_preserve_live_input(%input: tensor<17x2x4xf32>,
                                       %init: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<4xf32> {
    %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<17x2x4xf32>, vector<17x2x4xf32>
    %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
        : tensor<4xf32>, vector<4xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0, 1]
        : vector<17x2x4xf32> to vector<4xf32>
    %next = vector.transfer_write %reduced, %acc[%c0]
        {in_bounds = [true]} : vector<4xf32>, tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  }
  %live = tensor.extract %input[%c0, %c0, %c0] : tensor<17x2x4xf32>
  %old = tensor.extract %result[%c0] : tensor<4xf32>
  %sum = arith.addf %old, %live : f32
  %updated = tensor.insert %sum into %result[%c0] : tensor<4xf32>
  return %updated : tensor<4xf32>
}

// A rank-3 reduction with one reduction axis still needs staged lowering: its
// physical view is [outer, R, inner], not either of the legacy rank-2 forms.

// CHECK-LABEL: func.func @single_axis_batched_ra(
// CHECK: tensor.collapse_shape {{.*}} tensor<2x68xf32> into tensor<136xf32>
// CHECK-NOT: vector.multi_reduction
// CHECK: vector.transfer_read {{.*}} : tensor<2x257x68xf32>, vector<64xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>, tensor<136xf32>
// CHECK: tensor.expand_shape {{.*}} tensor<136xf32> into tensor<2x68xf32>
// CHECK: return
func.func @single_axis_batched_ra(%input: tensor<2x257x68xf32>,
                                  %init: tensor<2x68xf32>)
    -> tensor<2x68xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<2x68xf32> {
    %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
        {in_bounds = [true, true, true]}
        : tensor<2x257x68xf32>, vector<2x257x68xf32>
    %old = vector.transfer_read %acc[%c0, %c0], %zero
        {in_bounds = [true, true]}
        : tensor<2x68xf32>, vector<2x68xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [1]
        : vector<2x257x68xf32> to vector<2x68xf32>
    %next = vector.transfer_write %reduced, %acc[%c0, %c0]
        {in_bounds = [true, true]}
        : vector<2x68xf32>, tensor<2x68xf32>
    scf.yield %next : tensor<2x68xf32>
  }
  return %result : tensor<2x68xf32>
}

// only-marked may bypass the single-axis marker only for a perfect staged
// reduction loop nest. The extra add in the outer loop makes this nest
// ineligible.

// MARKED-LABEL: func.func @non_perfect_loop_nest(
// MARKED: vector.multi_reduction <add>
// MARKED: return
func.func @non_perfect_loop_nest(%input: tensor<2x3x4xf32>,
                                 %init: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %outer = %c0 to %c1 step %c1
      iter_args(%outer_acc = %init) -> tensor<3xf32> {
    %unused = arith.addi %outer, %c0 : index
    %inner_result = scf.for %inner = %c0 to %c1 step %c1
        iter_args(%acc = %outer_acc) -> tensor<3xf32> {
      %source = vector.transfer_read %input[%c0, %c0, %c0], %zero
          {in_bounds = [true, true, true]}
          : tensor<2x3x4xf32>, vector<2x3x4xf32>
      %old = vector.transfer_read %acc[%c0], %zero {in_bounds = [true]}
          : tensor<3xf32>, vector<3xf32>
      %reduced = vector.multi_reduction <add>, %source, %old [0, 2]
          : vector<2x3x4xf32> to vector<3xf32>
      %next = vector.transfer_write %reduced, %acc[%c0]
          {in_bounds = [true]} : vector<3xf32>, tensor<3xf32>
      scf.yield %next : tensor<3xf32>
    }
    scf.yield %inner_result : tensor<3xf32>
  }
  return %result : tensor<3xf32>
}

// Payload arithmetic is not a layout-only path back to an input tensor. The
// tree rewrite cannot discard it and reduce one of its operands instead.

// CHECK-LABEL: func.func @computed_multiply_source(
// CHECK: arith.mulf
// CHECK: vector.multi_reduction <add>
// CHECK: return
func.func @computed_multiply_source(%lhs: tensor<2048xf32>,
                                    %rhs: tensor<2048xf32>,
                                    %init: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c2048 step %c64
      iter_args(%acc = %init) -> tensor<f32> {
    %lhsVec = vector.transfer_read %lhs[%i], %zero
        : tensor<2048xf32>, vector<64xf32>
    %rhsVec = vector.transfer_read %rhs[%i], %zero
        : tensor<2048xf32>, vector<64xf32>
    %oldVec = vector.transfer_read %acc[], %zero
        : tensor<f32>, vector<f32>
    %old = vector.extractelement %oldVec[] : vector<f32>
    %product = arith.mulf %lhsVec, %rhsVec : vector<64xf32>
    %reduced = vector.multi_reduction <add>, %product, %old [0]
        : vector<64xf32> to f32
    %nextVec = vector.broadcast %reduced : f32 to vector<f32>
    %next = vector.transfer_write %nextVec, %acc[]
        : vector<f32>, tensor<f32>
    scf.yield %next : tensor<f32>
  }
  return %result : tensor<f32>
}

// CHECK-LABEL: func.func @computed_add_source(
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: vector.multi_reduction <add>
// CHECK: return
func.func @computed_add_source(%input: tensor<2048xf32>,
                               %weight: tensor<2048xf32>,
                               %value: tensor<2048xf32>, %offset: f32,
                               %init: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant 0.0 : f32
  %offsetVec = vector.broadcast %offset : f32 to vector<64xf32>
  %result = scf.for %i = %c0 to %c2048 step %c64
      iter_args(%acc = %init) -> tensor<f32> {
    %inputVec = vector.transfer_read %input[%i], %zero
        : tensor<2048xf32>, vector<64xf32>
    %weightVec = vector.transfer_read %weight[%i], %zero
        : tensor<2048xf32>, vector<64xf32>
    %valueVec = vector.transfer_read %value[%i], %zero
        : tensor<2048xf32>, vector<64xf32>
    %oldVec = vector.transfer_read %acc[], %zero
        : tensor<f32>, vector<f32>
    %old = vector.extractelement %oldVec[] : vector<f32>
    %delta = arith.subf %valueVec, %offsetVec : vector<64xf32>
    %weighted = arith.mulf %weightVec, %delta : vector<64xf32>
    %squared = arith.mulf %weighted, %delta : vector<64xf32>
    %source = arith.addf %inputVec, %squared : vector<64xf32>
    %reduced = vector.multi_reduction <add>, %source, %old [0]
        : vector<64xf32> to f32
    %nextVec = vector.broadcast %reduced : f32 to vector<f32>
    %next = vector.transfer_write %nextVec, %acc[]
        : vector<f32>, tensor<f32>
    scf.yield %next : tensor<f32>
  }
  return %result : tensor<f32>
}

// A pure rank-1 reduction is a staged suffix AR reduction. It must obey the
// AR direction flag even though RA is enabled.

// CHECK-LABEL: func.func @rank1_suffix_direction_gate(
// CHECK-NOT: vector.multi_reduction
// CHECK: vector.reduction <add>
// CHECK: return
// NOAR-LABEL: func.func @rank1_suffix_direction_gate(
// NOAR: vector.multi_reduction <add>
// NOAR: return
func.func @rank1_suffix_direction_gate(%input: tensor<2048xf32>,
                                       %init: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c2048 step %c64
      iter_args(%acc = %init) -> tensor<f32> {
    %source = vector.transfer_read %input[%i], %zero
        : tensor<2048xf32>, vector<64xf32>
    %oldVec = vector.transfer_read %acc[], %zero
        : tensor<f32>, vector<f32>
    %old = vector.extractelement %oldVec[] : vector<f32>
    %reduced = vector.multi_reduction <add>, %source, %old [0]
        : vector<64xf32> to f32
    %nextVec = vector.broadcast %reduced : f32 to vector<f32>
    %next = vector.transfer_write %nextVec, %acc[]
        : vector<f32>, tensor<f32>
    scf.yield %next : tensor<f32>
  }
  return %result : tensor<f32>
}
