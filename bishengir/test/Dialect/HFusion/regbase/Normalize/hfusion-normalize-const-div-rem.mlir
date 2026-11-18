// RUN: bishengir-opt %s -hfusion-normalize-ops="use-regbase=true" -canonicalize -cse -split-input-file -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: FileCheck %s --check-prefix=SHARE --input-file=%t
// RUN: bishengir-opt %s -hfusion-normalize-ops="use-regbase=true enable-div-rem-pow2=false" -split-input-file | FileCheck %s --check-prefix=NOPOW2
// RUN: bishengir-opt %s -hfusion-normalize-ops="use-regbase=true" -hfusion-pre-vectorization-fusion -split-input-file | FileCheck %s --check-prefix=LOWER

// CHECK-LABEL: func.func @signed_divrem_768(
// CHECK: 715827883
// CHECK: hfusion.mulext
// CHECK: binary_fn<shrsi>
// CHECK: binary_fn<shrui>
// CHECK: binary_fn<add>
// CHECK: binary_fn<mul>
// CHECK: binary_fn<sub>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @signed_divrem_768(%x: tensor<64xi32>) -> (tensor<64xi32>, tensor<64xi32>) {
    %d = arith.constant 768 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q, %r : tensor<64xi32>, tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @unsigned_divrem_7(
// CHECK: 613566757
// CHECK: hfusion.mulextui
// CHECK: binary_fn<sub>
// CHECK: binary_fn<shrui>
// CHECK: binary_fn<add>
// CHECK: binary_fn<shrui>
// CHECK: binary_fn<mul>
// CHECK: binary_fn<sub>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsigned_divrem_7(%x: tensor<64xi32>) -> (tensor<64xi32>, tensor<64xi32>) {
    %d = arith.constant 7 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q, %r : tensor<64xi32>, tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @unsigned_even_14(
// CHECK: binary_fn<shrui>
// CHECK: hfusion.mulextui
// CHECK: binary_fn<shrui>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsigned_even_14(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 14 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @signed_negative_7(
// CHECK: hfusion.mulext
// CHECK: binary_fn<sub>
// CHECK: binary_fn<shrsi>
// CHECK: binary_fn<shrui>
// CHECK: binary_fn<add>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @signed_negative_7(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant -7 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @signed_positive_7(
// CHECK: hfusion.mulext
// CHECK: binary_fn<add>
// CHECK: binary_fn<shrsi>
// CHECK: binary_fn<shrui>
// CHECK: binary_fn<add>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @signed_positive_7(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 7 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @unsigned_1200_fill(
// CHECK: hfusion.mulextui
// CHECK: binary_fn<shrui>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsigned_1200_fill(%x: tensor<64xi32>) -> tensor<64xi32> {
    %c = arith.constant 1200 : i32
    %e = tensor.empty() : tensor<64xi32>
    %d = linalg.fill ins(%c : i32) outs(%e : tensor<64xi32>) -> tensor<64xi32>
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%x, %d : tensor<64xi32>, tensor<64xi32>) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @signed_1536_splat(
// CHECK: hfusion.mulext
// CHECK: binary_fn<mul>
// CHECK: binary_fn<sub>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @signed_1536_splat(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant dense<1536> : tensor<64xi32>
    %init = tensor.empty() : tensor<64xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%x, %d : tensor<64xi32>, tensor<64xi32>) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @unsigned_2560(
// CHECK: -858993459
// CHECK: hfusion.mulextui
// CHECK: binary_fn<shrui>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsigned_2560(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 2560 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @unsigned_high_bit(
// CHECK: hfusion.mulextui
// CHECK: binary_fn<shrui>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsigned_high_bit(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 4294967295 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @dynamic_shape(
// CHECK: tensor.dim
// CHECK: hfusion.mulext
// CHECK: binary_fn<shrsi>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dynamic_shape(%x: tensor<?xi32>) -> tensor<?xi32> {
    %d = arith.constant 768 : i32
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<?xi32>, i32) outs(%init : tensor<?xi32>) -> tensor<?xi32>
    return %q : tensor<?xi32>
  }
}

// -----

// CHECK-LABEL: func.func @dynamic_divisor(
// CHECK: binary_fn<div>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @dynamic_divisor(%x: tensor<64xi32>, %d: tensor<64xi32>) -> tensor<64xi32> {
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, tensor<64xi32>) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @non_splat_divisor(
// CHECK: binary_fn<div>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @non_splat_divisor(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant dense<[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]> : tensor<64xi32>
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, tensor<64xi32>) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @zero_divisor(
// CHECK: binary_fn<div>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @zero_divisor(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 0 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @power_of_two(
// CHECK: binary_fn<shrsi>
// CHECK: binary_fn<vand>
// CHECK: binary_fn<add>
// CHECK: binary_fn<shrsi>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @power_of_two(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 8 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @i64_fallback(
// CHECK: binary_fn<div>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @i64_fallback(%x: tensor<64xi64>) -> tensor<64xi64> {
    %d = arith.constant 7 : i64
    %init = tensor.empty() : tensor<64xi64>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi64>, i64) outs(%init : tensor<64xi64>) -> tensor<64xi64>
    return %q : tensor<64xi64>
  }
}

// -----

// CHECK-LABEL: func.func @no_target(
// CHECK-NOT: hfusion.mulext
// CHECK: hfusion.cast
// CHECK: tensor<64xf32>
// CHECK: binary_fn<mul>
// CHECK-NOT: hfusion.mulext
// CHECK: return
module {
  func.func @no_target(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 7 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// -----

// CHECK-LABEL: func.func @a3_fallback(
// CHECK-NOT: hfusion.mulext
// CHECK: hfusion.cast
// CHECK: tensor<64xf32>
// CHECK: binary_fn<mul>
// CHECK-NOT: hfusion.mulext
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @a3_fallback(%x: tensor<64xi32>) -> tensor<64xi32> {
    %d = arith.constant 7 : i32
    %init = tensor.empty() : tensor<64xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%x, %d : tensor<64xi32>, i32) outs(%init : tensor<64xi32>) -> tensor<64xi32>
    return %q : tensor<64xi32>
  }
}

// SHARE-LABEL: func.func @signed_divrem_768(
// SHARE: hfusion.mulext
// SHARE-NOT: hfusion.mulext
// SHARE: return
// SHARE-LABEL: func.func @unsigned_divrem_7(
// SHARE: hfusion.mulextui
// SHARE-NOT: hfusion.mulextui
// SHARE: return
// LOWER-LABEL: func.func @signed_divrem_768(
// LOWER: arith.mulsi_extended
// LOWER-NOT: i64
// LOWER: return
// LOWER-LABEL: func.func @unsigned_divrem_7(
// LOWER: arith.mului_extended
// LOWER-NOT: i64
// LOWER: return
// LOWER-LABEL: func.func @dynamic_shape(
// LOWER: tensor.dim
// LOWER: arith.mulsi_extended
// LOWER: return

// NOPOW2-LABEL: func.func @signed_divrem_768(
// NOPOW2: hfusion.mulext
// NOPOW2: return
// NOPOW2-LABEL: func.func @power_of_two(
// NOPOW2-NOT: hfusion.mulext
// NOPOW2: binary_fn<div>
// NOPOW2-NOT: binary_fn<shr
// NOPOW2: return

// -----

// The divisor remains constant through both a fill and a broadcast.
// CHECK-LABEL: func.func @unsigned_768_broadcast(
// CHECK: hfusion.mulextui
// CHECK: binary_fn<shrui>
// CHECK-NOT: binary_fn<div_unsigned>
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @unsigned_768_broadcast(%x: tensor<4x8xi32>) -> tensor<4x8xi32> {
    %c = arith.constant 768 : i32
    %row = tensor.empty() : tensor<8xi32>
    %filled = linalg.fill ins(%c : i32) outs(%row : tensor<8xi32>) -> tensor<8xi32>
    %init = tensor.empty() : tensor<4x8xi32>
    %d = linalg.broadcast ins(%filled : tensor<8xi32>)
      outs(%init : tensor<4x8xi32>) dimensions = [0]
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
      ins(%x, %d : tensor<4x8xi32>, tensor<4x8xi32>)
      outs(%init : tensor<4x8xi32>) -> tensor<4x8xi32>
    return %q : tensor<4x8xi32>
  }
}
