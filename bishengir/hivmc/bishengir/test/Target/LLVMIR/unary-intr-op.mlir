// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_vabs_v256_s8_x(<256 x i8> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <256 x i8> @llvm.hivm.vabs.x.v256i8(<256 x i8> %0, <256 x i1> %1)
llvm.func @test_vabs_v256_s8_x(%src: vector<256 x i8>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vabs.x"(%src, %mask) : (vector<256 x i8>, vector<256 x i1>)-> (vector<256 x i8>)
  llvm.return
}

// CHECK-LABEL: define void @test_vabs_v128_s16_x(<128 x i16> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <128 x i16> @llvm.hivm.vabs.x.v128i16(<128 x i16> %0, <256 x i1> %1)
llvm.func @test_vabs_v128_s16_x(%src: vector<128 x i16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vabs.x"(%src, %mask) : (vector<128 x i16>, vector<256 x i1>)-> (vector<128 x i16>)
  llvm.return
}

// CHECK-LABEL: define void @test_vabs_v128_f16_x(<128 x half> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <128 x half> @llvm.hivm.vabs.x.v128f16(<128 x half> %0, <256 x i1> %1)
llvm.func @test_vabs_v128_f16_x(%src: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vabs.x"(%src, %mask) : (vector<128 x f16>, vector<256 x i1>)-> (vector<128 x f16>)
  llvm.return
}

// CHECK-LABEL: define void @test_vabs_v64_s32_x(<64 x i32> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vabs.x.v64i32(<64 x i32> %0, <256 x i1> %1)
llvm.func @test_vabs_v64_s32_x(%src: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vabs.x"(%src, %mask) : (vector<64 x i32>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vabs_v64_f32_x(<64 x float> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %0, <256 x i1> %1)
llvm.func @test_vabs_v64_f32_x(%src: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vabs.x"(%src, %mask) : (vector<64 x f32>, vector<256 x i1>)-> (vector<64 x f32>)
  llvm.return
}
