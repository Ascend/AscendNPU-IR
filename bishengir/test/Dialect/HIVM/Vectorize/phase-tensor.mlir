// RUN: bishengir-opt %s --hivm-vectorize=phase=tensor -split-input-file 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMARK
// RUN: bishengir-opt %s --hivm-vectorize=phase=tensor -split-input-file \
// RUN:   | FileCheck %s --check-prefix=IR

// The tensor phase runs the embedded transform sequence: partition, outline,
// lift, vectorize, cleanup and result selection. A supported group is
// published as a vector function.

module {
  func.func @elementwise(%a: tensor<512xf32>, %b: tensor<512xf32>,
                         %init: tensor<512xf32>) -> tensor<512xf32> {
    %sum = hivm.hir.vadd ins(%a, %b : tensor<512xf32>, tensor<512xf32>)
        outs(%init : tensor<512xf32>) -> tensor<512xf32>
    return %sum : tensor<512xf32>
  }
}

// REMARK: remark: selected transaction ticket 1 with cost
// IR-LABEL: func.func private @elementwise_fused_0
// IR-SAME: attributes {hivm.vector_function, no_inline}
// IR: scf.for
// IR: vector.transfer_read
// IR: arith.addf {{.*}} : vector<64xf32>
// IR-NOT: linalg.
// IR-LABEL: func.func @elementwise(
// IR: call @elementwise_fused_0
// IR-NOT: hivm.hir.vadd

// -----

// The tensor phase also materializes static sliced/reshaped call operands so
// the vector function receives a stable buffer.

module {
  func.func private @vf(%src: tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0], %cst {in_bounds = [true]}
        : tensor<8xf32>, vector<8xf32>
    %r = vector.transfer_write %v, %src[%c0] {in_bounds = [true]}
        : vector<8xf32>, tensor<8xf32>
    return %r : tensor<8xf32>
  }
  func.func @caller(%arg: tensor<32xf32>) -> tensor<8xf32> {
    %slice = tensor.extract_slice %arg[0] [8] [1]
        : tensor<32xf32> to tensor<8xf32>
    %r = call @vf(%slice) {hivm.vector_function, no_inline}
        : (tensor<8xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}

// REMARK: remark: no transaction candidate published
// IR-LABEL: func.func @caller(
// IR: %[[SLICE:.*]] = tensor.extract_slice
// IR-NEXT: %[[COPY:.*]] = bufferization.alloc_tensor() copy(%[[SLICE]])
// IR-NEXT: call @vf(%[[COPY]])
