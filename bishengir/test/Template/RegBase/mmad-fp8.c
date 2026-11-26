// REQUIRES: enable-lir-compile
// RUN: %bisheng_compiler -x cce %s --cce-aicore-only \
// RUN:   --cce-generic-addrspace=off --cce-auto-sync=off \
// RUN:   --cce-aicore-arch=dav-920r1-cube -std=c++17 -O2 -c -emit-llvm \
// RUN:   -I %bishengir_src_root/lib/Template/lib/RegBase/Cube/include \
// RUN:   -I %bishengir_src_root/lib/Template/include -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s

#include "ascendc_bisheng/ascendc_bisheng.hpp"

using namespace AscendCBisheng;

// CHECK-LABEL: define {{.*}} @regular(
// CHECK: call void @llvm.hivm.MAD.e4m3e4m3
extern "C" __aicore__ void regular(__cc__ float *c, __ca__ float8_e4m3_t *a,
                                   __cb__ float8_e4m3_t *b) {
  MmadCal(c, a, b, MmadParams(16, 16, 32, 0, false, true));
}

// CHECK-LABEL: define {{.*}} @regular_bias(
// CHECK: call void @llvm.hivm.MAD.e5m2e4m3
extern "C" __aicore__ void regular_bias(__cc__ float *c,
                                        __ca__ float8_e5m2_t *a,
                                        __cb__ float8_e4m3_t *b,
                                        uint64_t bias) {
  MmadCal(c, a, b, bias, MmadParams(16, 16, 32, 0, true, false), true);
}
