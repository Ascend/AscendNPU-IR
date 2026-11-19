// RUN: bishengir-opt %s -hfusion-legalize-fp8 -hfusion-normalize-ops -hfusion-legalize-scalar-fp8 | FileCheck %s --check-prefix=NORMALIZE
// RUN: bishengir-opt %s -hfusion-legalize-fp8 -hfusion-normalize-ops | FileCheck %s --check-prefix=WITHOUT

// WITHOUT-LABEL: func.func @masked_scalar_f8_to_f32_broadcast
// WITHOUT: scf.if {{.*}} -> (f8E4M3FN)
// NORMALIZE-LABEL: func.func @masked_scalar_f8_to_f32_broadcast
// NORMALIZE: %[[ZERO:.*]] = arith.constant 0 : i8
// NORMALIZE: %[[IF:.*]] = scf.if {{.*}} -> (i8) {
// NORMALIZE: %[[LD:.*]] = memref.load
// NORMALIZE: %[[BITS:.*]] = arith.bitcast %[[LD]] : f8E4M3FN to i8
// NORMALIZE: scf.yield %[[BITS]] : i8
// NORMALIZE: } else {
// NORMALIZE: scf.yield %[[ZERO]] : i8
// NORMALIZE: %[[VALUE:.*]] = arith.bitcast %[[IF]] : i8 to f8E4M3FN
// NORMALIZE: tensor.from_elements %[[VALUE]] : tensor<1xf8E4M3FN>
// NORMALIZE: hfusion.cast
// NORMALIZE: tensor.extract
// NORMALIZE: linalg.fill
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @masked_scalar_f8_to_f32_broadcast(%cond: i1, %src: memref<1xf8E4M3FN>, %dst: memref<8xf32>) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f8E4M3FN
    %v = scf.if %cond -> (f8E4M3FN) {
      %x = memref.load %src[%c0] : memref<1xf8E4M3FN>
      scf.yield %x : f8E4M3FN
    } else {
      scf.yield %zero : f8E4M3FN
    }
    %e = tensor.empty() : tensor<8xf8E4M3FN>
    %f = linalg.fill ins(%v : f8E4M3FN) outs(%e : tensor<8xf8E4M3FN>) -> tensor<8xf8E4M3FN>
    %o = tensor.empty() : tensor<8xf32>
    %r = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%f : tensor<8xf8E4M3FN>) outs(%o : tensor<8xf32>) -> tensor<8xf32>
    bufferization.materialize_in_destination %r in writable %dst : (tensor<8xf32>, memref<8xf32>) -> ()
    return
  }
}
