// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// Without the v2 intrinsics attribute the dual-output predicate
// interleave/de-interleave ops keep their legacy, struct-returning intrinsic
// forms, as required by historical targets.

// CHECK-LABEL: define void @test_pintlv_b8
// CHECK: call { <256 x i1>, <256 x i1> } @llvm.hivm.pintlv.b8(<256 x i1> %0, <256 x i1> %1)

// CHECK-LABEL: define void @test_pdintlv_b16
// CHECK: call { <256 x i1>, <256 x i1> } @llvm.hivm.pdintlv.b16(<256 x i1> %0, <256 x i1> %1)

// CHECK-LABEL: define void @test_pdintlv_b32
// CHECK: call { <256 x i1>, <256 x i1> } @llvm.hivm.pdintlv.b32(<256 x i1> %0, <256 x i1> %1)

// CHECK-NOT: llvm.hivm.pintlvv2
// CHECK-NOT: llvm.hivm.pdintlvv2

module {
  llvm.func @test_pintlv_b8(%a: vector<256xi1>, %b: vector<256xi1>) -> () {
    %0 = "hivm_regbaseintrins.intr.hivm.pintlv.b8"(%a, %b) : (vector<256xi1>, vector<256xi1>)-> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
    llvm.return
  }
  llvm.func @test_pdintlv_b16(%a: vector<256xi1>, %b: vector<256xi1>) -> () {
    %0 = "hivm_regbaseintrins.intr.hivm.pdintlv.b16"(%a, %b) : (vector<256xi1>, vector<256xi1>)-> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
    llvm.return
  }
  llvm.func @test_pdintlv_b32(%a: vector<256xi1>, %b: vector<256xi1>) -> () {
    %0 = "hivm_regbaseintrins.intr.hivm.pdintlv.b32"(%a, %b) : (vector<256xi1>, vector<256xi1>)-> !llvm.struct<(vector<256xi1>, vector<256xi1>)>
    llvm.return
  }
}
