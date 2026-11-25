// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-cross-if-fusion=false enable-multiple-consumer-fusion=false" -split-input-file | FileCheck %s --check-prefix=CLOSE
// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-cross-if-fusion=true enable-multiple-consumer-fusion=false" -split-input-file | FileCheck %s --check-prefix=OPEN

// A vsstb expand_shape separated from its transpose by a region barrier
// cannot be fused while cross-if fusion is disabled. It must not stay in the
// planning graph as a fusable user that never receives a node: the all-users
// gate would then leave its whole producer chain unplanned and the pass would
// vectorize those ops at full size outside any outlined loop, which trips the
// in-pass verifier.

// CLOSE-LABEL: func.func @vsstb_expand_barrier
// CLOSE-NOT: vector<16x64xf32>
// CLOSE: outlined-loop-target
// CLOSE-NOT: vector<16x64xf32>
// CLOSE: return

// OPEN-LABEL: func.func @vsstb_expand_barrier
// OPEN-NOT: vector<16x64xf32>
// OPEN: outlined-loop-target
// OPEN-NOT: vector<16x64xf32>
// OPEN: return

#mapA = affine_map<(d0, d1) -> (d0, d1)>
#mapB = affine_map<(d0, d1) -> (d0)>

module {
  func.func @vsstb_expand_barrier(%arg0: tensor<16x64xf32>, %arg1: tensor<16x64xf32>) -> (tensor<16xf32>, tensor<8x16x8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
    %empty0 = tensor.empty() : tensor<16x64xf32>
    %src = linalg.generic {indexing_maps = [#mapA, #mapA, #mapA], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%empty0 : tensor<16x64xf32>) {
    ^bb0(%in: f32, %in1: f32, %out: f32):
      %m = arith.mulf %in, %in1 : f32
      linalg.yield %m : f32
    } -> tensor<16x64xf32>

    %init = tensor.empty() : tensor<16xf32>
    %red = linalg.generic {indexing_maps = [#mapA, #mapB], iterator_types = ["parallel", "reduction"]} ins(%src : tensor<16x64xf32>) outs(%init : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %a = arith.addf %in, %out : f32
      linalg.yield %a : f32
    } -> tensor<16xf32>

    %ex = tensor.expand_shape %src [[0], [1, 2]] output_shape [16, 8, 8] : tensor<16x64xf32> into tensor<16x8x8xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
    }

    %empty1 = tensor.empty() : tensor<8x16x8xf32>
    %tr = linalg.transpose ins(%ex : tensor<16x8x8xf32>) outs(%empty1 : tensor<8x16x8xf32>) permutation = [1, 0, 2]
    return %red, %tr : tensor<16xf32>, tensor<8x16x8xf32>
  }
}
