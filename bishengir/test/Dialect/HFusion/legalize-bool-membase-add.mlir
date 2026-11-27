// RUN: bishengir-opt %s -hfusion-legalize-bool | FileCheck %s --check-prefixes=ADD,INT
// RUN: bishengir-opt %s -hfusion-legalize-bool="enable-clamp=true" | FileCheck %s --check-prefixes=CLAMP,INT

module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  // ADD-LABEL: func.func @bool_add
  // ADD: %[[OR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
  // ADD: return %[[OR]] : tensor<8xi1>
  // CLAMP-LABEL: func.func @bool_add
  // CLAMP: linalg.elemwise_binary {{.*}}#linalg.binary_fn<add>
  func.func @bool_add(%a: tensor<8xi1>, %b: tensor<8xi1>, %init: tensor<8xi1>) -> tensor<8xi1> {
    %r = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<8xi1>, tensor<8xi1>) outs(%init : tensor<8xi1>) -> tensor<8xi1>
    return %r : tensor<8xi1>
  }

  // INT-LABEL: func.func @widened_add
  // INT: %[[SUM:.*]] = linalg.elemwise_binary {{.*}}#linalg.binary_fn<add>
  // INT: return %[[SUM]] : tensor<8xi8>
  func.func @widened_add(%a: tensor<8xi1>, %b: tensor<8xi1>, %init: tensor<8xi8>) -> tensor<8xi8> {
    %r = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<8xi1>, tensor<8xi1>) outs(%init : tensor<8xi8>) -> tensor<8xi8>
    return %r : tensor<8xi8>
  }

  // INT-LABEL: func.func @mixed_add
  // INT: %[[SUM:.*]] = linalg.elemwise_binary {{.*}}#linalg.binary_fn<add>
  // INT: return %[[SUM]] : tensor<8xi8>
  func.func @mixed_add(%a: tensor<8xi8>, %b: tensor<8xi1>, %init: tensor<8xi8>) -> tensor<8xi8> {
    %b8 = arith.extui %b : tensor<8xi1> to tensor<8xi8>
    %r = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b8 : tensor<8xi8>, tensor<8xi8>) outs(%init : tensor<8xi8>) -> tensor<8xi8>
    return %r : tensor<8xi8>
  }
}
