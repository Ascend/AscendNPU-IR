// RUN: bishengir-opt %s --tree-reduce-v2="enable-ar=true enable-ra=true ascend-style-ra=true" | FileCheck %s

// TreeReduceV2 only lowers f32/f16/bf16 reductions. A reduction whose source
// was widened to f64 must be left untouched even though the traced input tensor
// itself is f32: rewriting it as an f32 reduction would silently drop the
// widening and change the result.
//
// CHECK-LABEL: func.func @f64_accumulation_is_kept(
// CHECK: vector.multi_reduction <add>, {{.*}} [0] : vector<64xf64> to f64
// CHECK: return
func.func @f64_accumulation_is_kept(%input: tensor<128xf32>,
                                   %init: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c128 step %c64
      iter_args(%acc = %init) -> (tensor<f32>) {
    %source = vector.transfer_read %input[%i], %zero
        : tensor<128xf32>, vector<64xf32>
    %wide = arith.extf %source : vector<64xf32> to vector<64xf64>
    %accVec = vector.transfer_read %acc[], %zero : tensor<f32>, vector<f32>
    %oldF = vector.extractelement %accVec[] : vector<f32>
    %oldWide = arith.extf %oldF : f32 to f64
    %reduced = vector.multi_reduction <add>, %wide, %oldWide [0]
        : vector<64xf64> to f64
    %narrow = arith.truncf %reduced : f64 to f32
    %bc = vector.broadcast %narrow : f32 to vector<f32>
    %next = vector.transfer_write %bc, %acc[] : vector<f32>, tensor<f32>
    scf.yield %next : tensor<f32>
  }
  return %result : tensor<f32>
}
