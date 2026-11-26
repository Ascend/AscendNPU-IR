// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// Ascend960 (V121) toolchains only provide the v2 part-select predicate
// interleave/de-interleave intrinsics (pintlvv2/pdintlvv2, part = 0/1 =
// LOWER/HIGHER). When the module carries the v2 intrinsics attribute, each
// dual-output predicate interleave op splits into two v2 calls and the struct
// result is reassembled.

// CHECK-LABEL: define void @test_pintlv_b8
// CHECK: call <256 x i1> @llvm.hivm.pintlvv2.b8(<256 x i1> %0, <256 x i1> %1, i32 0)
// CHECK: call <256 x i1> @llvm.hivm.pintlvv2.b8(<256 x i1> %0, <256 x i1> %1, i32 1)
// CHECK: insertvalue { <256 x i1>, <256 x i1> } poison, <256 x i1> %{{.*}}, 0
// CHECK: insertvalue { <256 x i1>, <256 x i1> } %{{.*}}, <256 x i1> %{{.*}}, 1

// CHECK-LABEL: define void @test_pdintlv_b16
// CHECK: call <256 x i1> @llvm.hivm.pdintlvv2.b16(<256 x i1> %0, <256 x i1> %1, i32 0)
// CHECK: call <256 x i1> @llvm.hivm.pdintlvv2.b16(<256 x i1> %0, <256 x i1> %1, i32 1)

// CHECK-LABEL: define void @test_pdintlv_b32
// CHECK: call <256 x i1> @llvm.hivm.pdintlvv2.b32(<256 x i1> %0, <256 x i1> %1, i32 0)
// CHECK: call <256 x i1> @llvm.hivm.pdintlvv2.b32(<256 x i1> %0, <256 x i1> %1, i32 1)

module attributes {hivm_regbaseintrins.v2intrins} {
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
