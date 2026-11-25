// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// Ascend960 (dav-920r1) only registers the typed, unsigned-amount shift
// intrinsics (`vshrs.v<vl><elem><amount>.z`). When the module carries the v2
// intrinsics attribute, the legacy shift ops must be rewritten to those forms.

// CHECK-LABEL: define void @test_vshrs_s16
// CHECK: call <128 x i16> @llvm.hivm.vshrs.v128s16u16.z(<128 x i16> %{{.*}}, i16 4, <256 x i1> %{{.*}})
// CHECK-LABEL: define void @test_vshls_u8
// CHECK: call <256 x i8> @llvm.hivm.vshls.v256u8u8.z(<256 x i8> %{{.*}}, i16 2, <256 x i1> %{{.*}})
// CHECK-LABEL: define void @test_vshrs_u32
// CHECK: call <64 x i32> @llvm.hivm.vshrs.v64u32u32.z(<64 x i32> %{{.*}}, i32 8, <256 x i1> %{{.*}})
// CHECK-LABEL: define void @test_vshr_s16
// CHECK: call <128 x i16> @llvm.hivm.vshr.v128s16u16.z(<128 x i16> %{{.*}}, <128 x i16> %{{.*}}, <256 x i1> %{{.*}})

module attributes {hivm_regbaseintrins.v2intrins} {
  llvm.func @test_vshrs_s16(%a: vector<128xi16>, %m: vector<256xi1>) -> () {
    %c4 = llvm.mlir.constant(4 : i16) : i16
    %0 = "hivm_regbaseintrins.intr.hivm.vshrs.s.x"(%a, %c4, %m) : (vector<128xi16>, i16, vector<256xi1>) -> vector<128xi16>
    llvm.return
  }
  llvm.func @test_vshls_u8(%a: vector<256xi8>, %m: vector<256xi1>) -> () {
    %c2 = llvm.mlir.constant(2 : i16) : i16
    %0 = "hivm_regbaseintrins.intr.hivm.vshls.u.x"(%a, %c2, %m) : (vector<256xi8>, i16, vector<256xi1>) -> vector<256xi8>
    llvm.return
  }
  llvm.func @test_vshrs_u32(%a: vector<64xi32>, %m: vector<256xi1>) -> () {
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %0 = "hivm_regbaseintrins.intr.hivm.vshrs.u.x"(%a, %c8, %m) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    llvm.return
  }
  llvm.func @test_vshr_s16(%a: vector<128xi16>, %b: vector<128xi16>, %m: vector<256xi1>) -> () {
    %0 = "hivm_regbaseintrins.intr.hivm.vshr.s.x"(%a, %b, %m) : (vector<128xi16>, vector<128xi16>, vector<256xi1>) -> vector<128xi16>
    llvm.return
  }
}
