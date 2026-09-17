// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_plt_b8(i32 %0) {
// CHECK: %[[CALL:.*]] = call { <256 x i1>, i32 } @llvm.hivm.plt.b8.v300(i32 %0)
llvm.func @test_plt_b8(%scalar:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.plt.b8.v300"(%scalar) : (i32) -> !llvm.struct<(vector<256 x i1>, i32)>
  llvm.return
}

// CHECK-LABEL: define void @test_plt_b16(i32 %0) {
// CHECK: %[[CALL:.*]] = call { <256 x i1>, i32 } @llvm.hivm.plt.b16.v300(i32 %0)
llvm.func @test_plt_b16(%scalar:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.plt.b16.v300"(%scalar) : (i32) -> !llvm.struct<(vector<256 x i1>, i32)>
  llvm.return
}

// CHECK-LABEL: define void @test_plt_b32(i32 %0) {
// CHECK: %[[CALL:.*]] = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %0)
llvm.func @test_plt_b32(%scalar:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.plt.b32.v300"(%scalar) : (i32) -> !llvm.struct<(vector<256 x i1>, i32)>
  llvm.return
}
