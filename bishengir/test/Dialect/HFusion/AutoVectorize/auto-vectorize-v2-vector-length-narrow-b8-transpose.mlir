// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2 | FileCheck %s --check-prefix=VL256
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="vector-length=200" | FileCheck %s --check-prefix=VL200
// RUN: bishengir-opt %s -hfusion-auto-vectorize-v2="vector-length=200" -outline-vector-function \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   -normalize-vector -cse | FileCheck %s --check-prefix=GATHER

// An i8 transpose of 200 elements per row. With the default vector length
// (256 bytes) the vectorizer pads the permuted read to a whole register and
// carries the real extent in masks.
// VL256-LABEL: func.func @transpose_i8_200
// VL256: vector.mask %{{.*}} { vector.transfer_read {{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<200x1xi8>, vector<1x256xi8> } : vector<256x1xi1> -> vector<1x256xi8>
// VL256: arith.addi %{{.*}}, %{{.*}} : vector<1x256xi8>

// A non-default vector length of 200 bytes gives an unmasked 200-lane
// permuted read, the width in (128, 256) that a single native B8 gather
// cannot cover.
// VL200-LABEL: func.func @transpose_i8_200
// VL200-NOT: vector.mask
// VL200: vector.transfer_read {{.*}} {permutation_map = #{{.*}}} : tensor<200x1xi8>, vector<1x200xi8>
// VL200: arith.addi %{{.*}}, %{{.*}} : vector<1x200xi8>

// After bufferization, NormalizeVector turns that read into one dual gather
// (two 128-index halves) that keeps the logical 200-element width.
// GATHER-LABEL: func.func @transpose_i8_200_outlined_vf_0
// GATHER-NOT: vector.transfer_read {{.*}} permutation_map
// GATHER: %[[MASK:.*]] = vector.constant_mask [200] : vector<200xi1>
// GATHER-NEXT: %[[G:.*]] = vector.gather %{{.*}}, %[[MASK]], %{{.*}} {secondary_index = dense<{{.*}}> : vector<128xi16>} : memref<200x1xi8, strided<[16, 1], offset: ?>>, vector<128xi16>, vector<200xi1>, vector<200xi8> into vector<200xi8>
// GATHER-NEXT: memref.subview
// GATHER-NEXT: vector.transfer_write %[[G]], %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<200xi8>, memref<200xi8, #{{.*}}>
// GATHER: arith.addi %{{.*}}, %{{.*}} : vector<200xi8>
module {
  func.func @transpose_i8_200(%arg0: tensor<200x16xi8>) -> tensor<16x200xi8> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
    %0 = tensor.empty() : tensor<16x200xi8>
    %transposed = linalg.transpose ins(%arg0 : tensor<200x16xi8>) outs(%0 : tensor<16x200xi8>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<16x200xi8>
    %2 = linalg.add ins(%transposed, %transposed : tensor<16x200xi8>, tensor<16x200xi8>) outs(%1 : tensor<16x200xi8>) -> tensor<16x200xi8>
    return %2 : tensor<16x200xi8>
  }
}
