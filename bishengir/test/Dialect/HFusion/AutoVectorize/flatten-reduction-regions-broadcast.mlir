// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true" --hfusion-pre-vectorization-fusion --canonicalize --cse | FileCheck %s --check-prefix=GLOBAL
// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true" --hfusion-pre-vectorization-fusion --hfusion-flatten-reduction-regions --canonicalize --cse | FileCheck %s --check-prefix=LOCAL
// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false skip-scope=true" --hfusion-pre-vectorization-fusion="enable-triton-compile=true" --hfusion-prepare-i1-nx1-for-vectorization --hfusion-flatten-reduction-regions --cse --canonicalize="enable-extended-patterns=true" --normalize-tensor-ops="skip-aligned-slice=true" | FileCheck %s --check-prefix=LOCAL
// RUN: bishengir-opt %s --hfusion-flatten-ops="flatten-mode=tidy register-based=true multi-dynamic-shape=false skip-scope=true" --hfusion-pre-vectorization-fusion="enable-triton-compile=true" --hfusion-prepare-i1-nx1-for-vectorization --hfusion-flatten-reduction-regions --cse --canonicalize="enable-extended-patterns=true" --normalize-tensor-ops="skip-aligned-slice=true" --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=false" | FileCheck %s --check-prefix=V2

// Both broadcast constraints must remain in the same function as the cast and
// reductions. Removing them would only test ordinary whole-function flattening.
// GLOBAL-LABEL: func.func @broadcast_cast_reductions(
// GLOBAL: arith.extf
// GLOBAL: } -> tensor<6x80x8xf32>
// GLOBAL-COUNT-2: iterator_types = ["parallel", "reduction", "reduction"]

// The broadcast-containing producer stays three-dimensional and fused. Only
// its completed result crosses the local boundary; no full broadcast tensor or
// standalone square tensor is introduced. The cast stays separate and its
// collapsed result is shared by both reductions.
// LOCAL-LABEL: func.func @broadcast_cast_reductions(
// LOCAL: %[[MAIN:.*]] = linalg.generic
// LOCAL-SAME: tensor<80xf16>
// LOCAL: arith.subf
// LOCAL: arith.mulf
// LOCAL: } -> tensor<6x80x8xf16>
// LOCAL: %[[FLAT:.*]] = tensor.collapse_shape %[[MAIN]] {{\[\[}}0], [1, 2]] : tensor<6x80x8xf16> into tensor<6x640xf16>
// LOCAL-NOT: tensor.expand_shape
// LOCAL: %[[CAST:.*]] = linalg.generic
// LOCAL-SAME: iterator_types = ["parallel", "parallel"]
// LOCAL-SAME: ins(%[[FLAT]] : tensor<6x640xf16>)
// LOCAL: arith.extf
// LOCAL: } -> tensor<6x640xf32>
// LOCAL-NOT: arith.extf
// LOCAL: %[[SUM:.*]] = linalg.generic
// LOCAL-SAME: iterator_types = ["parallel", "reduction"]
// LOCAL-SAME: ins(%[[CAST]] : tensor<6x640xf32>)
// LOCAL-NOT: arith.extf
// LOCAL: arith.addf
// LOCAL-NOT: arith.extf
// LOCAL: %[[SQUARE_SUM:.*]] = linalg.generic
// LOCAL-SAME: iterator_types = ["parallel", "reduction"]
// LOCAL-SAME: ins(%[[CAST]] : tensor<6x640xf32>)
// LOCAL-NOT: arith.extf
// LOCAL: arith.mulf
// LOCAL-NOT: arith.extf
// LOCAL: arith.addf
// LOCAL-NOT: arith.extf
// LOCAL: return %[[MAIN]], %[[SUM]], %[[SQUARE_SUM]] : tensor<6x80x8xf16>, tensor<6xf32>, tensor<6xf32>

// V2 keeps the cast separate and reduces over the collapsed domain.
// V2-LABEL: func.func @broadcast_cast_reductions(
// V2: arith.extf {{.*}} : vector<{{.*}}f16> to vector<{{.*}}f32>
// V2: tensor<6x640xf32>
// V2-NOT: arith.extf
// V2: } {reductionLoop}
// V2: vector.multi_reduction <add>
// V2: vector.multi_reduction <add>
// V2-NOT: tensor<6x80x8xf32>
// V2: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @broadcast_cast_reductions(%x: tensor<6x80x8xf16>, %a: tensor<80xf16>, %c: tensor<6x8xf16>, %init: tensor<6xf32>) -> (tensor<6x80x8xf16>, tensor<6xf32>, tensor<6xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd", mix_mode = "aiv"} {
    %empty = tensor.empty() : tensor<6x80x8xf16>
    %a3 = linalg.broadcast ins(%a : tensor<80xf16>) outs(%empty : tensor<6x80x8xf16>) dimensions = [0, 2]
    %c3 = linalg.broadcast ins(%c : tensor<6x8xf16>) outs(%empty : tensor<6x80x8xf16>) dimensions = [1]
    %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%x, %a3 : tensor<6x80x8xf16>, tensor<6x80x8xf16>) outs(%empty : tensor<6x80x8xf16>) -> tensor<6x80x8xf16>
    %main = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%sub, %c3 : tensor<6x80x8xf16>, tensor<6x80x8xf16>) outs(%empty : tensor<6x80x8xf16>) -> tensor<6x80x8xf16>
    %empty_f32 = tensor.empty() : tensor<6x80x8xf32>
    %cast = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%main : tensor<6x80x8xf16>) outs(%empty_f32 : tensor<6x80x8xf32>) -> tensor<6x80x8xf32>
    %sum = linalg.reduce ins(%cast : tensor<6x80x8xf32>) outs(%init : tensor<6xf32>) dimensions = [1, 2]
        (%v: f32, %acc: f32) {
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
    }
    %square = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%cast, %cast : tensor<6x80x8xf32>, tensor<6x80x8xf32>) outs(%empty_f32 : tensor<6x80x8xf32>) -> tensor<6x80x8xf32>
    %square_sum = linalg.reduce ins(%square : tensor<6x80x8xf32>) outs(%init : tensor<6xf32>) dimensions = [1, 2]
        (%v: f32, %acc: f32) {
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
    }
    return %main, %sum, %square_sum : tensor<6x80x8xf16>, tensor<6xf32>, tensor<6xf32>
  }
}
