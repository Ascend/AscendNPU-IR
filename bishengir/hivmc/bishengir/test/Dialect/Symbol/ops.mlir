// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect -split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: test_symbolic_int
func.func @test_symbolic_int() {
  %S0 = symbol.symbolic_int @S0 [], affine_map<() -> ()> {max_val = 10 : i64, min_val = 0 : i64} : index
  %S1 = symbol.symbolic_int @S1 [%S0], affine_map<()[s0] -> (s0 + 1)> {max_val = 11 : i64, min_val = 1 : i64} : index
  %S2 = symbol.symbolic_int @S2 {max_val = 10 : i64, min_val = 0 : i64} : index
  return
}

// -----

// CHECK-LABEL: test_bind_symbolic_shape
func.func @test_bind_symbolic_shape() {
  %S0 = symbol.symbolic_int @S0 [], affine_map<() -> ()> {max_val = 10 : i64, min_val = 0 : i64} : index
  %some_value = "some_op"() : () -> tensor<?x16xf32>
  symbol.bind_symbolic_shape %some_value, [%S0], affine_map<()[s0] -> (s0, 16)> : tensor<?x16xf32>
  return
}
