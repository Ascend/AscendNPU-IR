// REQUIRES: execution-engine
//
// Compare the rewritten tensor quotient/remainder with scalar integer div/rem.
// Cover both signed correction directions, unsigned IsAdd and PreShift,
// high-bit unsigned divisors, INT_MIN/MAX, exact multiples and their neighbors.
// Dynamic tensor dimensions also exercise extended-multiply generalization.
//
// RUN: bishengir-opt %s --hfusion-normalize-ops="use-regbase=true" \
// RUN:   --hfusion-pre-vectorization-fusion \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm --finalize-memref-to-llvm \
// RUN:   --convert-cf-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: sed -e "s/module attributes {[^}]*}/module/" | \
// RUN: mlir-cpu-runner -e main -entry-point-result=i32 | FileCheck %s
// CHECK: {{^0$}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func private @check_signed(%x: tensor<?xi32>, %q: tensor<?xi32>,
                                  %r: tensor<?xi32>, %d: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    scf.for %i = %c0 to %n step %c1 {
      %v = tensor.extract %x[%i] : tensor<?xi32>
      %actual_q = tensor.extract %q[%i] : tensor<?xi32>
      %actual_r = tensor.extract %r[%i] : tensor<?xi32>
      %expected_q = arith.divsi %v, %d : i32
      %expected_r = arith.remsi %v, %d : i32
      %q_ok = arith.cmpi eq, %actual_q, %expected_q : i32
      %r_ok = arith.cmpi eq, %actual_r, %expected_r : i32
      cf.assert %q_ok, "signed quotient mismatch"
      cf.assert %r_ok, "signed remainder mismatch"
    }
    return
  }

  func.func private @signed_3(%x: tensor<?xi32>) {
    %d = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @signed_7(%x: tensor<?xi32>) {
    %d = arith.constant 7 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @signed_neg_7(%x: tensor<?xi32>) {
    %d = arith.constant -7 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @signed_768(%x: tensor<?xi32>) {
    %d = arith.constant 768 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @signed_neg_768(%x: tensor<?xi32>) {
    %d = arith.constant -768 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @signed_2147483647(%x: tensor<?xi32>) {
    %d = arith.constant 2147483647 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @signed_neg_2147483647(%x: tensor<?xi32>) {
    %d = arith.constant -2147483647 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_signed(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @check_unsigned(%x: tensor<?xi32>, %q: tensor<?xi32>,
                                    %r: tensor<?xi32>, %d: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    scf.for %i = %c0 to %n step %c1 {
      %v = tensor.extract %x[%i] : tensor<?xi32>
      %actual_q = tensor.extract %q[%i] : tensor<?xi32>
      %actual_r = tensor.extract %r[%i] : tensor<?xi32>
      %expected_q = arith.divui %v, %d : i32
      %expected_r = arith.remui %v, %d : i32
      %q_ok = arith.cmpi eq, %actual_q, %expected_q : i32
      %r_ok = arith.cmpi eq, %actual_r, %expected_r : i32
      cf.assert %q_ok, "unsigned quotient mismatch"
      cf.assert %r_ok, "unsigned remainder mismatch"
    }
    return
  }

  func.func private @unsigned_3(%x: tensor<?xi32>) {
    %d = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @unsigned_7(%x: tensor<?xi32>) {
    %d = arith.constant 7 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @unsigned_14(%x: tensor<?xi32>) {
    %d = arith.constant 14 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  // This divisor exercises an unsigned pre-shift with no post-shift.
  func.func private @unsigned_28(%x: tensor<?xi32>) {
    %d = arith.constant 28 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @unsigned_768(%x: tensor<?xi32>) {
    %d = arith.constant 768 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @unsigned_2147483649(%x: tensor<?xi32>) {
    %d = arith.constant 2147483649 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func private @unsigned_4294967295(%x: tensor<?xi32>) {
    %d = arith.constant 4294967295 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
      ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    func.call @check_unsigned(%x, %q, %r, %d)
      : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32) -> ()
    return
  }

  func.func @main() -> i32 {
    %input = arith.constant dense<[
      -2147483648, -2147483647, -1536, -1000, -769, -768, -767, -7,
      -1, 0, 1, 2, 3, 6, 7, 8, 767, 768, 769, 1536,
      2147483646, 2147483647, 2147483903, 4294967295
    ]> : tensor<24xi32>
    %x = tensor.cast %input : tensor<24xi32> to tensor<?xi32>
    func.call @signed_3(%x) : (tensor<?xi32>) -> ()
    func.call @signed_7(%x) : (tensor<?xi32>) -> ()
    func.call @signed_neg_7(%x) : (tensor<?xi32>) -> ()
    func.call @signed_768(%x) : (tensor<?xi32>) -> ()
    func.call @signed_neg_768(%x) : (tensor<?xi32>) -> ()
    func.call @signed_2147483647(%x) : (tensor<?xi32>) -> ()
    func.call @signed_neg_2147483647(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_3(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_7(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_14(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_28(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_768(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_2147483649(%x) : (tensor<?xi32>) -> ()
    func.call @unsigned_4294967295(%x) : (tensor<?xi32>) -> ()
    %success = arith.constant 0 : i32
    return %success : i32
  }
}
