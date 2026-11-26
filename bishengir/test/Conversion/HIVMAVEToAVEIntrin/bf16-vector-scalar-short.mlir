// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivmave-to-ave-intrin | FileCheck %s

// CHECK-LABEL: func.func @vmaxs_bf16_n1
// CHECK: "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// CHECK: vector<128xbf16> to vector<1xbf16>
func.func @vmaxs_bf16_n1(%vec: vector<1xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> vector<1xbf16> {
  %res = ave.hir.vmaxs %vec, %scalar, %mask : vector<1xbf16>, bf16, vector<128xi1>
  return %res : vector<1xbf16>
}

// CHECK-LABEL: func.func @vmins_bf16_n1
// CHECK: "hivm_regbaseintrins.intr.hivm.vmins.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// CHECK: vector<128xbf16> to vector<1xbf16>
func.func @vmins_bf16_n1(%vec: vector<1xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> vector<1xbf16> {
  %res = ave.hir.vmins %vec, %scalar, %mask : vector<1xbf16>, bf16, vector<128xi1>
  return %res : vector<1xbf16>
}

// CHECK-LABEL: func.func @vmaxs_bf16_n64
// CHECK: "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// CHECK: vector<128xbf16> to vector<64xbf16>
func.func @vmaxs_bf16_n64(%vec: vector<64xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> vector<64xbf16> {
  %res = ave.hir.vmaxs %vec, %scalar, %mask : vector<64xbf16>, bf16, vector<128xi1>
  return %res : vector<64xbf16>
}

// CHECK-LABEL: func.func @vmins_bf16_n64
// CHECK: "hivm_regbaseintrins.intr.hivm.vmins.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// CHECK: vector<128xbf16> to vector<64xbf16>
func.func @vmins_bf16_n64(%vec: vector<64xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> vector<64xbf16> {
  %res = ave.hir.vmins %vec, %scalar, %mask : vector<64xbf16>, bf16, vector<128xi1>
  return %res : vector<64xbf16>
}
