// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

// CHECK-LABEL: symbolic_int
func.func @test_symbol(%some_index: index) {
  // expected-error@below {{int symbol must be produced by valid operations}}
  %S1 = symbol.symbolic_int @S1 [%some_index], affine_map<(s0) -> (s0 + 1)> {max_val = 11 : i64, min_val = 1 : i64} : index
  return
}

// -----

// CHECK-LABEL: symbolic_int
func.func @test_symbol(%some_index: index) {
  // expected-error@below {{number of int symbols 0 doesn't match with affine map 1}}
  %S1 = symbol.symbolic_int @S1 [], affine_map<()[s0] -> (s0 + 1)> {max_val = 11 : i64, min_val = 1 : i64} : index
  return
}

// -----

// CHECK-LABEL: symbolic_int
func.func @test_symbol(%some_index: index) {
  // expected-error@below {{the affine map should only contain symbols}}
  %S1 = symbol.symbolic_int @S1 [], affine_map<(d0)[] -> (d0 + 1)> {max_val = 11 : i64, min_val = 1 : i64} : index
  return
}

// -----

// CHECK-LABEL: symbolic_int
func.func @test_symbol(%some_index: index) {
  %S1 = symbol.symbolic_int @S1 [], affine_map<()[] -> ()> {max_val = 11 : i64, min_val = 1 : i64} : index
  %S2 = symbol.symbolic_int @S2 [], affine_map<()[] -> ()> {max_val = 11 : i64, min_val = 1 : i64} : index
  // expected-error@below {{mapping must not produce more than one value}}
  %S3 = symbol.symbolic_int @S3 [%S1, %S2], affine_map<()[s0, s1] -> (s0 + 1, s1)> {max_val = 11 : i64, min_val = 1 : i64} : index
  return
}

// -----

// CHECK-LABEL: test_bind_symbolic_shape
func.func @test_bind_symbolic_shape() {
  %S0 = symbol.symbolic_int @S0 {max_val = 10 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 {max_val = 10 : i64, min_val = 0 : i64} : index
  %some_value = "some_op"() : () -> tensor<?x16xf32>
  // expected-error@below {{op number of shape symbols doesn't match the number of symbols in the affine.map}}
  symbol.bind_symbolic_shape %some_value, [%S0, %S1], affine_map<()[s0] -> (s0, 16)> : tensor<?x16xf32>
  return
}

// -----

// CHECK-LABEL: test_bind_symbolic_shape
func.func @test_bind_symbolic_shape() {
  %some_value = "some_op"() : () -> tensor<?x16xf32>
  // expected-error@below {{op requires non-empty shapeSymbols}}
  symbol.bind_symbolic_shape %some_value, [], affine_map<()[s0] -> (s0, 16)> : tensor<?x16xf32>
  return
}

