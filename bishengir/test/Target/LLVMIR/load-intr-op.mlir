// RUN: bishengir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: define void @test_load(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <32 x i64> @llvm.hivm.vldsx1.v32u64(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v32u64"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load1(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <32 x i64> @llvm.hivm.vldsx1.v32s64(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load1(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v32s64"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load2(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vldsx1.v64s32(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load2(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64s32"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load3(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <64 x i32> @llvm.hivm.vldsx1.v64u32(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load3(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64u32"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load4(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <128 x i16> @llvm.hivm.vldsx1.v128s16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load4(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128s16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load5(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <128 x i16> @llvm.hivm.vldsx1.v128u16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load5(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128u16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load6(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <256 x i8> @llvm.hivm.vldsx1.v256s8(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load6(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load7(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <256 x i8> @llvm.hivm.vldsx1.v256u8(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load7(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v256u8"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load8(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load8(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load9(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <128 x half> @llvm.hivm.vldsx1.v128f16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load9(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_load10(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_load10(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128bf16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> (vector<32 x i64>)
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <32 x i64>, <32 x i64> } @llvm.hivm.vldsx2.v32u64(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v32u64"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<32xi64>, vector<32xi64>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_2(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <32 x i64>, <32 x i64> } @llvm.hivm.vldsx2.v32s64(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_2(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v32s64"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<32xi64>, vector<32xi64>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_3(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <64 x i32>, <64 x i32> } @llvm.hivm.vldsx2.v64s32(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_3(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v64s32"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<64xi32>, vector<64xi32>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_4(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <64 x i32>, <64 x i32> } @llvm.hivm.vldsx2.v64u32(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_4(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v64u32"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<64xi32>, vector<64xi32>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_5(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <128 x i16>, <128 x i16> } @llvm.hivm.vldsx2.v128s16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_5(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v128s16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_6(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <128 x i16>, <128 x i16> } @llvm.hivm.vldsx2.v128u16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_6(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v128u16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<128xi16>, vector<128xi16>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_7(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <256 x i8>, <256 x i8> } @llvm.hivm.vldsx2.v256s8(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_7(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v256s8"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_8(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <256 x i8>, <256 x i8> } @llvm.hivm.vldsx2.v256u8(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_8(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v256u8"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<256xi8>, vector<256xi8>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_9(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <64 x float>, <64 x float> } @llvm.hivm.vldsx2.v64f32(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_9(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v64f32"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<64xf32>, vector<64xf32>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_10(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <128 x half>, <128 x half> } @llvm.hivm.vldsx2.v128f16(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_10(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.v128f16"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX1_Post(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <32 x i64>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v32u64(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX1_Post(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx1.post.v32u64"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<64xi64>, !llvm.ptr)>
  llvm.return
}

// CHECK-LABEL: define void @test_loadX2_Post(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3) {
// CHECK: %[[CALL:.*]] = call { <32 x i64>, <32 x i64>, ptr addrspace(6) } @llvm.hivm.vldsx2.post.v32u64(ptr addrspace(6) %0, i32 %1, i32 %2, i32 %3)
llvm.func @test_loadX2_Post(%base: !llvm.ptr<6>, %offset: i32, %dis: i32, %mode:i32) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vldsx2.post.v32u64"(%base, %offset, %dis, %mode) : (!llvm.ptr<6>, i32, i32, i32) -> !llvm.struct<(vector<32xi64>, vector<32xi64>, !llvm.ptr)>
  llvm.return
}
