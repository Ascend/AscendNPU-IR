// REQUIRES: execution-engine
// RUN: bishengir-opt --convert-hivm-to-hfusion="convert-to-named-op=true" %s | FileCheck %s --check-prefixes=COMMON
// RUN: bishengir-opt --convert-hivm-to-hfusion="convert-to-named-op=false" %s | FileCheck %s --check-prefixes=COMMON

// COMMON-LABEL: func.func @integer_tensor_bitwise_named_lowering
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @integer_tensor_bitwise_named_lowering(%a: tensor<16xi32>, %b: tensor<16xi32>, %dst: tensor<16xi32>) -> tensor<16xi32> {
      // COMMON: %[[AND:.*]] = linalg.map { arith.andi }
      %0 = hivm.hir.vand ins(%a, %b : tensor<16xi32>, tensor<16xi32>) outs(%dst : tensor<16xi32>) -> tensor<16xi32>

      // COMMON: %[[OR:.*]] = linalg.map { arith.ori }
      %1 = hivm.hir.vor ins(%0, %b : tensor<16xi32>, tensor<16xi32>) outs(%dst : tensor<16xi32>) -> tensor<16xi32>

      // COMMON: %[[XOR:.*]] = linalg.map { arith.xori }
      %2 = hivm.hir.vxor ins(%1, %b : tensor<16xi32>, tensor<16xi32>) outs(%dst : tensor<16xi32>) -> tensor<16xi32>

      func.return %2 : tensor<16xi32>
  }
}

// -----

// Multi-function module: the host entry and the AIC (cube) function must be
// skipped, while every collected AIV function must be rewritten with the same
// pattern set. @aiv_second is the regression guard: the pattern set must not
// be moved-from by the first greedy application, otherwise the second
// function would keep its hivm.hir.vadd.

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // COMMON-LABEL: func.func @host_entry
  // COMMON-NOT: linalg.add
  // COMMON: hivm.hir.vadd
  func.func @host_entry(%a: tensor<16xf32>, %b: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }

  // COMMON-LABEL: func.func @aic_func
  // COMMON-NOT: linalg.add
  // COMMON: hivm.hir.vadd
  func.func @aic_func(%a: tensor<16xf32>, %b: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }

  // COMMON-LABEL: func.func @aiv_first
  // COMMON: linalg.add
  func.func @aiv_first(%a: tensor<16xf32>, %b: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }

  // COMMON-LABEL: func.func @aiv_second
  // COMMON: linalg.add
  func.func @aiv_second(%a: tensor<16xf32>, %b: tensor<16xf32>, %dst: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %0 = hivm.hir.vadd ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%dst : tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
