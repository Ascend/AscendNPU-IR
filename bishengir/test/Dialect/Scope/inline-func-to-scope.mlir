// RUN: bishengir-opt -inline-func-to-scope %s | FileCheck %s
// RUN: bishengir-opt -inline-func-to-scope -outline-scope %s | FileCheck %s --check-prefix=ROUNDTRIP

// The VF callee is inlined into a scope.scope at the call site; callee
// arguments map to the call operands, function-level attributes are
// transferred to the scope, and call-site attributes are recorded in the
// `call_attrs_vec` dictionary.

// CHECK-LABEL: func.func @caller
func.func @caller(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2x2xf32> {
  // CHECK: %[[ADD:.*]] = arith.addf %arg0, %arg1 : tensor<2x2xf32>
  // CHECK: scope.return %[[ADD]] : tensor<2x2xf32>
  // CHECK: } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline, outline, scope.call_attrs_vec = #scope.call_attrs_vec<{hivm.vector_function, ptc_simdvf}>, scope.sym_func_name = #scope.sym_func_name<"vf_add">}
  %0 = func.call @vf_add(%arg0, %arg1) {hivm.vector_function, ptc_simdvf} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %[[SCOPE]] : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// The inlined callee is erased once it has no remaining symbol uses.
//
// CHECK-NOT: func.func @vf_add
func.func @vf_add(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %1 = arith.addf %arg0, %arg1 : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// The command-line pass accepts every callee, so callees without
// `hivm.vector_function` are inlined as well.
//
// CHECK-LABEL: func.func @plain_caller
func.func @plain_caller(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[SCOPE:.*]] = scope.scope : () -> tensor<2x2xf32> {
  // CHECK: scope.return %arg0 : tensor<2x2xf32>
  // CHECK: } {outline, scope.sym_func_name = #scope.sym_func_name<"plain">}
  %0 = func.call @plain(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %[[SCOPE]] : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-NOT: func.func @plain
func.func @plain(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  return %arg0 : tensor<2x2xf32>
}

// Round trip: outline-scope restores the function form under the original VF
// name recorded in `sym_func_name`, with an equivalent signature. Function
// attributes match the original callee; call-site attributes are restored on
// the call, and the bookkeeping attributes (`outline`, `sym_func_name`,
// `call_attrs_vec`) are dropped.
//
// ROUNDTRIP-LABEL: func.func @vf_add
// ROUNDTRIP-SAME: {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline}
// ROUNDTRIP: %[[ADD:.*]] = arith.addf %arg0, %arg1 : tensor<2x2xf32>
// ROUNDTRIP: return %[[ADD]] : tensor<2x2xf32>
// ROUNDTRIP-LABEL: func.func @caller
// ROUNDTRIP: %[[CALL:.*]] = call @vf_add(%arg0, %arg1) {hivm.vector_function, ptc_simdvf} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// ROUNDTRIP: return %[[CALL]] : tensor<2x2xf32>
// ROUNDTRIP-NOT: func.func @caller_scope
// ROUNDTRIP-NOT: scope.sym_func_name
// ROUNDTRIP-NOT: outline
