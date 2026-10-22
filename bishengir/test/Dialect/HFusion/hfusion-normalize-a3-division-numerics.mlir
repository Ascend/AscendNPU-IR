// REQUIRES: execution-engine
// RUN: bishengir-opt %s --hfusion-normalize-ops \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops --convert-scf-to-cf \
// RUN:   --convert-math-to-llvm --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts > %t
// RUN: FileCheck %s --check-prefix=CPU --implicit-check-not=llvm.fptosi < %t
// RUN: sed 's/ {round_mode = #hfusion.round_mode<rint>}//g' %t | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils -shared-libs=%mlir_c_runner_utils | FileCheck %s

// Execute the constant-divisor integer fast paths with exact integer oracles.
// The CPU lowering does not implement hfusion RINT casts, so floating-point
// compensation is covered structurally and by separate device tests.
// Select lowering may convert i1 to f16: remove its redundant rounding attr
// before CPU parsing. Reject float-to-int casts, where removal is unsound.
// INT_MIN / -1 intentionally checks this implementation's modulo-i32 result;
// it is not a claim that upstream arith.divsi defines signed overflow.

module {
// CPU: llvm.func @constant_one_divisions
  func.func @constant_one_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant 1 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @constant_minus_one_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant -1 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @constant_min_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant -2147483648 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @constant_two_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant 2 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @constant_negative_eight_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant -8 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @constant_positive_large_power_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant 1073741824 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @constant_negative_large_power_divisions(%a: tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>) {
    %b = arith.constant -1073741824 : i32
    %out = tensor.empty() : tensor<16xi32>
    %trunc = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %floor = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    %ceil = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%a, %b : tensor<16xi32>, i32) outs(%out : tensor<16xi32>) -> tensor<16xi32>
    return %trunc, %floor, %ceil : tensor<16xi32>, tensor<16xi32>, tensor<16xi32>
  }

  func.func @print_result(%x: tensor<16xi32>) {
    %m = bufferization.to_memref %x : memref<16xi32>
    %u = memref.cast %m : memref<16xi32> to memref<*xi32>
    func.call @printMemrefI32(%u) : (memref<*xi32>) -> ()
    return
  }
  func.func private @printMemrefI32(memref<*xi32>)
  func.func @main() {
    %a = arith.constant dense<[-2147483648, -2147483647, 2147483647, 2147483646, 16777217, -16777217, 16777219, -16777219, 7, 7, -7, -7, 0, 1, -1, 2147483647]> : tensor<16xi32>
    %b = arith.constant dense<[-1, 1, 3, -3, 2, -2, 7, -7, 4, -4, 4, -4, 3, -2147483648, -2147483648, 1]> : tensor<16xi32>
    %constant_one_divisions:3 = func.call @constant_one_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_one_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_one_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_one_divisions#2) : (tensor<16xi32>) -> ()
    %constant_minus_one_divisions:3 = func.call @constant_minus_one_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_minus_one_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_minus_one_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_minus_one_divisions#2) : (tensor<16xi32>) -> ()
    %constant_min_divisions:3 = func.call @constant_min_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_min_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_min_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_min_divisions#2) : (tensor<16xi32>) -> ()
    %constant_two_divisions:3 = func.call @constant_two_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_two_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_two_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_two_divisions#2) : (tensor<16xi32>) -> ()
    %constant_negative_eight_divisions:3 = func.call @constant_negative_eight_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_negative_eight_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_negative_eight_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_negative_eight_divisions#2) : (tensor<16xi32>) -> ()
    %constant_positive_large_power_divisions:3 = func.call @constant_positive_large_power_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_positive_large_power_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_positive_large_power_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_positive_large_power_divisions#2) : (tensor<16xi32>) -> ()
    %constant_negative_large_power_divisions:3 = func.call @constant_negative_large_power_divisions(%a) : (tensor<16xi32>) -> (tensor<16xi32>, tensor<16xi32>, tensor<16xi32>)
    func.call @print_result(%constant_negative_large_power_divisions#0) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_negative_large_power_divisions#1) : (tensor<16xi32>) -> ()
    func.call @print_result(%constant_negative_large_power_divisions#2) : (tensor<16xi32>) -> ()
    return
  }
}

// constant_one_divisions: trunc
// CHECK: [-2147483648, -2147483647, 2147483647, 2147483646, 16777217, -16777217, 16777219, -16777219, 7, 7, -7, -7, 0, 1, -1, 2147483647]
// constant_one_divisions: floor
// CHECK: [-2147483648, -2147483647, 2147483647, 2147483646, 16777217, -16777217, 16777219, -16777219, 7, 7, -7, -7, 0, 1, -1, 2147483647]
// constant_one_divisions: ceil
// CHECK: [-2147483648, -2147483647, 2147483647, 2147483646, 16777217, -16777217, 16777219, -16777219, 7, 7, -7, -7, 0, 1, -1, 2147483647]
// constant_minus_one_divisions: trunc
// CHECK: [-2147483648, 2147483647, -2147483647, -2147483646, -16777217, 16777217, -16777219, 16777219, -7, -7, 7, 7, 0, -1, 1, -2147483647]
// constant_minus_one_divisions: floor
// CHECK: [-2147483648, 2147483647, -2147483647, -2147483646, -16777217, 16777217, -16777219, 16777219, -7, -7, 7, 7, 0, -1, 1, -2147483647]
// constant_minus_one_divisions: ceil
// CHECK: [-2147483648, 2147483647, -2147483647, -2147483646, -16777217, 16777217, -16777219, 16777219, -7, -7, 7, 7, 0, -1, 1, -2147483647]
// constant_min_divisions: trunc
// CHECK: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
// constant_min_divisions: floor
// CHECK: [1, 0, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, -1]
// constant_min_divisions: ceil
// CHECK: [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
// constant_two_divisions: trunc
// CHECK: [-1073741824, -1073741823, 1073741823, 1073741823, 8388608, -8388608, 8388609, -8388609, 3, 3, -3, -3, 0, 0, 0, 1073741823]
// constant_two_divisions: floor
// CHECK: [-1073741824, -1073741824, 1073741823, 1073741823, 8388608, -8388609, 8388609, -8388610, 3, 3, -4, -4, 0, 0, -1, 1073741823]
// constant_two_divisions: ceil
// CHECK: [-1073741824, -1073741823, 1073741824, 1073741823, 8388609, -8388608, 8388610, -8388609, 4, 4, -3, -3, 0, 1, 0, 1073741824]
// constant_negative_eight_divisions: trunc
// CHECK: [268435456, 268435455, -268435455, -268435455, -2097152, 2097152, -2097152, 2097152, 0, 0, 0, 0, 0, 0, 0, -268435455]
// constant_negative_eight_divisions: floor
// CHECK: [268435456, 268435455, -268435456, -268435456, -2097153, 2097152, -2097153, 2097152, -1, -1, 0, 0, 0, -1, 0, -268435456]
// constant_negative_eight_divisions: ceil
// CHECK: [268435456, 268435456, -268435455, -268435455, -2097152, 2097153, -2097152, 2097153, 0, 0, 1, 1, 0, 0, 1, -268435455]
// constant_positive_large_power_divisions: trunc
// CHECK: [-2, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
// constant_positive_large_power_divisions: floor
// CHECK: [-2, -2, 1, 1, 0, -1, 0, -1, 0, 0, -1, -1, 0, 0, -1, 1]
// constant_positive_large_power_divisions: ceil
// CHECK: [-2, -1, 2, 2, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 2]
// constant_negative_large_power_divisions: trunc
// CHECK: [2, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
// constant_negative_large_power_divisions: floor
// CHECK: [2, 1, -2, -2, -1, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, -2]
// constant_negative_large_power_divisions: ceil
// CHECK: [2, 2, -1, -1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, -1]
