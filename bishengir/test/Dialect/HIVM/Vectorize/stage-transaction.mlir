// RUN: bishengir-opt %s --hivm-vectorize -split-input-file 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMARK
// RUN: bishengir-opt %s --hivm-vectorize -split-input-file \
// RUN:   | FileCheck %s --check-prefix=IR

// The tensor phase is speculative: a candidate is published only when the
// attempt produces a verified result. A rejected attempt leaves the payload
// untouched and is not a silent fallback to the legacy vectorizer.

module {
  func.func @supported(%a: tensor<64xf32>, %b: tensor<64xf32>,
                       %init: tensor<64xf32>) -> tensor<64xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %sum : tensor<64xf32>
  }
}

// REMARK: remark: selected transaction ticket 1 with cost
// IR-LABEL: func.func private @supported_fused_0
// IR-SAME: attributes {hivm.vector_function, no_inline}
// IR-LABEL: func.func @supported(
// IR: call @supported_fused_0

// -----

module {
  func.func @rejected(%src: tensor<8x16xf32>, %init: tensor<16x8xf32>)
      -> tensor<16x8xf32> {
    %0 = hivm.hir.vtranspose ins(%src : tensor<8x16xf32>)
        outs(%init : tensor<16x8xf32>) permutation = [1, 0] -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}

// REMARK: remark: no transaction candidate published
// IR-LABEL: func.func @rejected(
// IR: hivm.hir.vtranspose
// IR-NOT: hivm.vector_function
