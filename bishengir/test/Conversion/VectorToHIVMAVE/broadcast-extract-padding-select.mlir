// RUN: bishengir-opt %s -hacc-append-device-spec -remove-redundant-write-and-read-pair \
// RUN: -one-shot-bufferize -canonicalize -normalize-arith -normalize-vector \
// RUN: -arith-vector-mask-analyze -convert-vector-to-hivmave \
// RUN: -convert-arith-to-hivmave | FileCheck %s

// A padding scalar lifted from a 2-D vector must not be forwarded: the
// resulting broadcast would be rewritten into a broadcast of the 2-D source
// vector, which does not lower.

// CHECK-LABEL: func.func @pad_from_first_lane(
// CHECK-NOT: ave.hir.vector_broadcast
// CHECK: ave.hir.vload
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pad_from_first_lane(%d: memref<64xf32, #hivm.address_space<ub>>, %v: vector<64xf32>, %pv: vector<8x8xf32>, %out: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %dst = bufferization.to_tensor %d restrict writable : memref<64xf32, #hivm.address_space<ub>>
    %c0 = arith.constant 0 : index
    %pad = vector.extract %pv[0, 0] : f32 from vector<8x8xf32>
    %wm = vector.constant_mask [32] : vector<64xi1>
    %w = vector.transfer_write %v, %dst[%c0], %wm {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %e = tensor.expand_shape %w [[0, 1]] output_shape [8, 8] : tensor<64xf32> into tensor<8x8xf32>
    %rm = vector.constant_mask [4, 8] : vector<8x8xi1>
    %r = vector.transfer_read %e[%c0, %c0], %pad, %rm {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
    %flat = vector.shape_cast %r : vector<8x8xf32> to vector<64xf32>
    vector.transfer_write %flat, %out[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}
