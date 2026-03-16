// RUN: mkdir -p %t/impl/
// RUN: rm -f %t/impl/*
// RUN: echo "extern \"C\" {[aicore] void _mlir_ciface_foo() {}}" > %t/impl/foo.cpp
// RUN: ccec -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -c -emit-llvm %t/impl/foo.cpp -o %t/impl/foo.bc
// RUN: bishengir-compile %s --link-aicore-bitcode %t/impl/foo.bc | FileCheck %s

// Check custom bishengir-compile compilation succeed
func.func @custom_test(%arg0 : memref<?xf32>, %arg1 : tensor<3x3xi64>)
  attributes { hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  // CHECK: call @foo
  %0 = hivm.hir.custom
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
         func_name = "foo" }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}
