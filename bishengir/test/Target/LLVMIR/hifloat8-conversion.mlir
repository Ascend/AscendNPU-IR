// RUN: bishengir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define <256 x hifloat8> @encode_hybrid(
// CHECK: call <256 x hifloat8> @llvm.hivm.vcvtff.f322hif8.x({{.*}}i32 6, i32 0, i32 0)
llvm.func @encode_hybrid(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> {
  %rnd = llvm.mlir.constant(6 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %y = "hivm_regbaseintrins.intr.hivm.vcvtff.f322hif8.x"(%x, %mask, %rnd, %zero, %zero) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8HIF>
  llvm.return %y : vector<256xf8HIF>
}

// CHECK-LABEL: define <64 x float> @decode(
// CHECK: call <64 x float> @llvm.hivm.vcvtff.hif82f32.x({{.*}}i32 0)
llvm.func @decode(%x: vector<256xf8HIF>, %mask: vector<256xi1>) -> vector<64xf32> {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %y = "hivm_regbaseintrins.intr.hivm.vcvtff.hif82f32.x"(%x, %mask, %zero) : (vector<256xf8HIF>, vector<256xi1>, i32) -> vector<64xf32>
  llvm.return %y : vector<64xf32>
}
