// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_pge_b8(i32 %0, i32 %1) {
// CHECK: %[[CALL:.*]] = call <256 x i1> @llvm.hivm.pge.b8(i32 %0, i32 %1)
llvm.func @test_pge_b8(%scalar:i32, %scalar1:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.pge.b8"(%scalar, %scalar1) : (i32, i32) -> (vector<256 x i1>)
  llvm.return
}

// CHECK-LABEL: define void @test_pge_b16(i32 %0, i32 %1) {
// CHECK: %[[CALL:.*]] = call <256 x i1> @llvm.hivm.pge.b16(i32 %0, i32 %1)
llvm.func @test_pge_b16(%scalar:i32, %scalar1:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%scalar, %scalar1) : (i32, i32) -> (vector<256 x i1>)
  llvm.return
}

// CHECK-LABEL: define void @test_pge_b32(i32 %0, i32 %1) {
// CHECK: %[[CALL:.*]] = call <256 x i1> @llvm.hivm.pge.b32(i32 %0, i32 %1)
llvm.func @test_pge_b32(%scalar:i32, %scalar1:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%scalar, %scalar1) : (i32, i32) -> (vector<256 x i1>)
  llvm.return
}
