// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @test_vcadd_v128_f16_x(<128 x half> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <128 x half> @llvm.hivm.vcadd.s.x.v128f16.v128f16(<128 x half> [[V0:%.*]], <256 x i1> [[V1:%.*]])
llvm.func @test_vcadd_v128_f16_x(%a: vector<128 x f16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%a, %mask) : (vector<128 x f16>, vector<256 x i1>)-> (vector<128 x f16>)
  llvm.return
}

// CHECK-LABEL: define void @test_vcadd_v128_s16_x(<128 x i16> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vcadd.s.x.v64i32.v128i16(<128 x i16> [[V0:%.*]], <256 x i1> [[V1:%.*]])
llvm.func @test_vcadd_v128_s16_x(%a: vector<128 x i16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%a, %mask) : (vector<128 x i16>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vcadd_v128_u16_x(<128 x i16> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vcadd.u.x.v64i32.v128i16(<128 x i16> [[V0:%.*]], <256 x i1> [[V1:%.*]])
llvm.func @test_vcadd_v128_u16_x(%a: vector<128 x i16>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vcadd.u.x"(%a, %mask) : (vector<128 x i16>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vcadd_v64_f32_x(<64 x float> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> [[V0:%.*]], <256 x i1> [[V1:%.*]])
llvm.func @test_vcadd_v64_f32_x(%a: vector<64 x f32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%a, %mask) : (vector<64 x f32>, vector<256 x i1>)-> (vector<64 x f32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vcadd_v64_s32_x(<64 x i32> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vcadd.s.x.v64i32.v64i32(<64 x i32> [[V0:%.*]], <256 x i1> [[V1:%.*]])
llvm.func @test_vcadd_v64_s32_x(%a: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%a, %mask) : (vector<64 x i32>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}

// CHECK-LABEL: define void @test_vcadd_v64_u32_x(<64 x i32> %0, <256 x i1> %1) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vcadd.u.x.v64i32.v64i32(<64 x i32> [[V0:%.*]], <256 x i1> [[V1:%.*]])
llvm.func @test_vcadd_v64_u32_x(%a: vector<64 x i32>, %mask: vector<256 x i1>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vcadd.u.x"(%a, %mask) : (vector<64 x i32>, vector<256 x i1>)-> (vector<64 x i32>)
  llvm.return
}
