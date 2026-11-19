// RUN: not bishengir-opt %s -convert-arith-to-hivmave 2>&1 | FileCheck %s
// Unsupported dense vectors must fail conversion, not become legal silently.
// CHECK: error: failed to legalize operation 'arith.constant'
func.func @unsupported_constant() -> vector<32xi32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c = arith.constant dense<[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131]> : vector<32xi32>
  return %c : vector<32xi32>
}
