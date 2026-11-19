// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// Ascend960 (V121) vselrv2 is a b8 masked-move intrinsic lowered directly from
// the regbase op. Both the data and the u8/s8 data mask are 256-byte vectors
// and the intrinsic selects src0 lane-wise by the mask, else zero.

// CHECK-LABEL: define {{.*}} @test_vselrv2_b8
// CHECK: call <256 x i8> @llvm.hivm.vselrv2.v256i8(<256 x i8> %0, <256 x i8> %1)

llvm.func @test_vselrv2_b8(%a: vector<256xi8>, %m: vector<256xi8>) -> () {
  %0 = "hivm_regbaseintrins.intr.hivm.vselrv2"(%a, %m) : (vector<256xi8>, vector<256xi8>)-> vector<256xi8>
  llvm.return
}
