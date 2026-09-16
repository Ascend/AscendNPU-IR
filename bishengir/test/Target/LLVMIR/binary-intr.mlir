// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @test_vadd_v128_f16_x(<128 x half> %0, <128 x half> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <128 x half> @llvm.hivm.vadd.s.x.v128f16(<128 x half> %0, <128 x half> %1, <256 x i1> %2)
llvm.func @test_vadd_v128_f16_x(%a: vector<128 x f16>, %b: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%a, %b, %mask) : (vector<128 x f16>, vector<128 x f16>, vector<256 x i1>)-> (vector<128 x f16>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v128_s16_x(<128 x i16> %0, <128 x i16> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <128 x i16> @llvm.hivm.vadd.s.x.v128i16(<128 x i16> %0, <128 x i16> %1, <256 x i1> %2)
llvm.func @test_vadd_v128_s16_x(%a: vector<128 x i16>, %b: vector<128 x i16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%a, %b, %mask) : (vector<128 x i16>, vector<128 x i16>, vector<256 x i1>)-> (vector<128 x i16>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v128_u16_x(<128 x i16> %0, <128 x i16> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <128 x i16> @llvm.hivm.vadd.u.x.v128i16(<128 x i16> %0, <128 x i16> %1, <256 x i1> %2)
llvm.func @test_vadd_v128_u16_x(%a: vector<128 x i16>, %b: vector<128 x i16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.u.x"(%a, %b, %mask) : (vector<128 x i16>, vector<128 x i16>, vector<256 x i1>)-> (vector<128 x i16>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v256_s8_x(<256 x i8> %0, <256 x i8> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <256 x i8> @llvm.hivm.vadd.s.x.v256i8(<256 x i8> %0, <256 x i8> %1, <256 x i1> %2)
llvm.func @test_vadd_v256_s8_x(%a: vector<256 x i8>, %b: vector<256 x i8>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%a, %b, %mask) : (vector<256 x i8>, vector<256 x i8>, vector<256 x i1>)-> (vector<256 x i8>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v256_u8_x(<256 x i8> %0, <256 x i8> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <256 x i8> @llvm.hivm.vadd.u.x.v256i8(<256 x i8> %0, <256 x i8> %1, <256 x i1> %2)
llvm.func @test_vadd_v256_u8_x(%a: vector<256 x i8>, %b: vector<256 x i8>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.u.x"(%a, %b, %mask) : (vector<256 x i8>, vector<256 x i8>, vector<256 x i1>)-> (vector<256 x i8>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v64_f32_x(<64 x float> %0, <64 x float> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %0, <64 x float> %1, <256 x i1> %2)
llvm.func @test_vadd_v64_f32_x(%a: vector<64 x f32>, %b: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%a, %b, %mask) : (vector<64 x f32>, vector<64 x f32>, vector<256 x i1>)-> (vector<64 x f32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v64_s32_x(<64 x i32> %0, <64 x i32> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vadd.s.x.v64i32(<64 x i32> %0, <64 x i32> %1, <256 x i1> %2)
llvm.func @test_vadd_v64_s32_x(%a: vector<64 x i32>, %b: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%a, %b, %mask) : (vector<64 x i32>, vector<64 x i32>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vadd_v64_u32_x(<64 x i32> %0, <64 x i32> %1, <256 x i1> %2) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32> %0, <64 x i32> %1, <256 x i1> %2)
llvm.func @test_vadd_v64_u32_x(%a: vector<64 x i32>, %b: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vadd.u.x"(%a, %b, %mask) : (vector<64 x i32>, vector<64 x i32>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}
