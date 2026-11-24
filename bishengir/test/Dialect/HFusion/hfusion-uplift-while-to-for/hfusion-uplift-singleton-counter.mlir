// RUN: bishengir-opt %s --hfusion-uplift-while-to-for | FileCheck %s

func.func private @update(tensor<16xf32>) -> tensor<16xf32>

// GEN-1749 carries a scalar loop counter alongside the tensor being computed.
// CHECK-LABEL: func.func @singleton_counter(
// CHECK-SAME:    %[[START:[^:]+]]: i32, %[[END:[^:]+]]: i32, %[[INPUT:[^:]+]]: tensor<16xf32>
// CHECK:       %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:       %[[OUT:.*]] = scf.for %{{.*}} = %[[START]] to %[[END]] step %[[ONE]] iter_args(%[[T:.*]] = %[[INPUT]])
// CHECK:         %[[NEXT:.*]] = {{.*}}call @update(%[[T]])
// CHECK:         scf.yield %[[NEXT]] : tensor<16xf32>
// CHECK:       return %[[OUT]] : tensor<16xf32>
func.func @singleton_counter(%start: i32, %end: i32, %input: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<1xi32>
  %init = linalg.fill ins(%start : i32) outs(%empty : tensor<1xi32>) -> tensor<1xi32>
  %step = linalg.fill ins(%c1 : i32) outs(%empty : tensor<1xi32>) -> tensor<1xi32>
  %r:2 = scf.while (%t = %input, %iv = %init) : (tensor<16xf32>, tensor<1xi32>) -> (tensor<16xf32>, tensor<1xi32>) {
    %i = tensor.extract %iv[%c0] : tensor<1xi32>
    %lt = arith.cmpi slt, %i, %end : i32
    scf.condition(%lt) %t, %iv : tensor<16xf32>, tensor<1xi32>
  } do {
  ^bb0(%t: tensor<16xf32>, %iv: tensor<1xi32>):
    %v = func.call @update(%t) : (tensor<16xf32>) -> tensor<16xf32>
    %next = arith.addi %step, %iv : tensor<1xi32>
    scf.yield %v, %next : tensor<16xf32>, tensor<1xi32>
  }
  return %r#0 : tensor<16xf32>
}

// A tensor step can be zero or negative.
// CHECK-LABEL: func.func @unknown_step(
// CHECK:       scf.while
// CHECK:         tensor.extract
// CHECK:         arith.addi {{.*}} : tensor<1xi32>
func.func @unknown_step(%end: i32, %step: tensor<1xi32>, %input: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant dense<0> : tensor<1xi32>
  %r:2 = scf.while (%iv = %zero, %t = %input) : (tensor<1xi32>, tensor<16xf32>) -> (tensor<1xi32>, tensor<16xf32>) {
    %i = tensor.extract %iv[%c0] : tensor<1xi32>
    %lt = arith.cmpi slt, %i, %end : i32
    scf.condition(%lt) %iv, %t : tensor<1xi32>, tensor<16xf32>
  } do {
  ^bb0(%iv: tensor<1xi32>, %t: tensor<16xf32>):
    %v = func.call @update(%t) : (tensor<16xf32>) -> tensor<16xf32>
    %next = arith.addi %iv, %step : tensor<1xi32>
    scf.yield %next, %v : tensor<1xi32>, tensor<16xf32>
  }
  return %r#1 : tensor<16xf32>
}

// linalg.fill casts its input to the output element type.
// CHECK-LABEL: func.func @widened_initial_value(
// CHECK:       linalg.fill ins(%{{.*}} : i16)
// CHECK:       scf.while
func.func @widened_initial_value(%start: i16, %end: i32, %input: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %step = arith.constant dense<1> : tensor<1xi32>
  %empty = tensor.empty() : tensor<1xi32>
  %init = linalg.fill ins(%start : i16) outs(%empty : tensor<1xi32>) -> tensor<1xi32>
  %r:2 = scf.while (%iv = %init, %t = %input) : (tensor<1xi32>, tensor<16xf32>) -> (tensor<1xi32>, tensor<16xf32>) {
    %i = tensor.extract %iv[%c0] : tensor<1xi32>
    %lt = arith.cmpi slt, %i, %end : i32
    scf.condition(%lt) %iv, %t : tensor<1xi32>, tensor<16xf32>
  } do {
  ^bb0(%iv: tensor<1xi32>, %t: tensor<16xf32>):
    %v = func.call @update(%t) : (tensor<16xf32>) -> tensor<16xf32>
    %next = arith.addi %iv, %step : tensor<1xi32>
    scf.yield %next, %v : tensor<1xi32>, tensor<16xf32>
  }
  return %r#1 : tensor<16xf32>
}
