// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: void @vf_0(ptr noalias %[[arg0:.*]], ptr noalias %[[arg1:.*]], i64 %[[arg2:.*]])
llvm.func @vf_0( %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg23: i64) attributes {hivm.vector_function} {
  llvm.return
}
