// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" \
// RUN:   2>&1 | FileCheck %s

// The result-check stage validates that every selected function is an
// attributed vector function whose body no longer contains vectorization
// compute or HIVM operations. The verified function count is the cost that
// the transaction selects on.

module attributes {transform.target_tag = "payload"} {
  func.func @vf(%src: memref<64xf32>, %dst: memref<64xf32>)
      attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %v = vector.transfer_read %src[%c0], %cst {in_bounds = [true]}
        : memref<64xf32>, vector<64xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<64xf32>, memref<64xf32>
    return
  }

  func.func @caller(%src: memref<64xf32>, %dst: memref<64xf32>) {
    call @vf(%src, %dst) {hivm.vector_function, no_inline}
        : (memref<64xf32>, memref<64xf32>) -> ()
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %vfs = transform.structured.match ops{["func.func"]}
        attributes {hivm.vector_function} in %root
        : (!transform.any_op) -> !transform.any_op
    %cost = transform.hivm.check_result %vfs
        : (!transform.any_op) -> !transform.any_param
    transform.debug.emit_param_as_remark %cost, "cost" : !transform.any_param
    transform.yield
  }
}

// CHECK: remark: cost 1 : i64
