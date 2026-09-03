// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @test_vdup_v64_s32(<64 x i32> %0, <64 x i32> %1, <256 x i1> %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vdup.m.v64i32(<64 x i32> [[V0:%.*]], <64 x i32> [[V1:%.*]], <256 x i1> [[V2:%.*]], i32 [[V3:%.*]])
llvm.func @test_vdup_v64_s32(%src0: vector<64 x i32>, %src1: vector<64 x i32>, %mask: vector<256 x i1>, %mode: i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vdup.m"(%src0, %src1, %mask, %mode) : (vector<64 x i32>, vector<64 x i32>, vector<256 x i1>, i32)-> (vector<64 x i32>)
  llvm.return
}
