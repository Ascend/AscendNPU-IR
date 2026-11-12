// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910_9589 \
// RUN: -convert-hivmave-to-ave-intrin %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @vcpadd_f16
// CHECK: hivm_regbaseintrins.intr.hivm.vcpadd.x
func.func @vcpadd_f16(%src: vector<128xf16>, %mask: vector<128xi1>)
    -> vector<128xf16> {
  %res = ave.hir.vcpadd %src, %mask
      : vector<128xf16>, vector<128xi1>
  return %res : vector<128xf16>
}

// -----

// CHECK-LABEL: func.func @vcpadd_f32
// CHECK: hivm_regbaseintrins.intr.hivm.vcpadd.x
func.func @vcpadd_f32(%src: vector<64xf32>, %mask: vector<64xi1>)
    -> vector<64xf32> {
  %res = ave.hir.vcpadd %src, %mask
      : vector<64xf32>, vector<64xi1>
  return %res : vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @vsqz_stored
// CHECK: %[[STORE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vsqz.x"({{.*}}, {{.*}}, %[[STORE]])
func.func @vsqz_stored(%src: vector<256xi8>, %mask: vector<256xi1>)
    -> vector<256xi8> {
  %res = ave.hir.vsqz %src, %mask : vector<256xi8>, vector<256xi1>
  return %res : vector<256xi8>
}

// -----

// CHECK-LABEL: func.func @vsqz_no_stored
// CHECK: %[[STORE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vsqz.x"({{.*}}, {{.*}}, %[[STORE]])
func.func @vsqz_no_stored(%src: vector<256xi8>, %mask: vector<256xi1>)
    -> vector<256xi8> {
  %res = ave.hir.vsqz %src, %mask
      {squeeze_store = #ave.squeeze_store<no_stored>}
      : vector<256xi8>, vector<256xi1>
  return %res : vector<256xi8>
}

// -----

// CHECK-LABEL: func.func @vusqz_i8
// CHECK-NOT: dense<0>
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK: %[[INIT:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[ZERO]])
// CHECK: "hivm_regbaseintrins.intr.hivm.vusqz.m"(%[[INIT]],
func.func @vusqz_i8(%mask: vector<256xi1>) -> vector<256xi8> {
  %res = ave.hir.vusqz %mask : vector<256xi1> -> vector<256xi8>
  return %res : vector<256xi8>
}

// -----

// CHECK-LABEL: func.func @vusqz_i16
// CHECK-NOT: dense<0>
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i16) : i16
// CHECK: %[[INIT:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[ZERO]])
// CHECK: "hivm_regbaseintrins.intr.hivm.vusqz.m"(%[[INIT]],
func.func @vusqz_i16(%mask: vector<128xi1>) -> vector<128xi16> {
  %res = ave.hir.vusqz %mask : vector<128xi1> -> vector<128xi16>
  return %res : vector<128xi16>
}

// -----

// CHECK-LABEL: func.func @vusqz_i32
// CHECK-NOT: dense<0>
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[INIT:.*]] = "hivm_regbaseintrins.intr.hivm.vbr"(%[[ZERO]])
// CHECK: "hivm_regbaseintrins.intr.hivm.vusqz.m"(%[[INIT]],
func.func @vusqz_i32(%mask: vector<64xi1>) -> vector<64xi32> {
  %res = ave.hir.vusqz %mask : vector<64xi1> -> vector<64xi32>
  return %res : vector<64xi32>
}
