// REQUIRES: enable-lir-compile, bishengir-hifloat8
// RUN: split-file %s %t
// RUN: %bisheng_compiler -x cce %t/probe.cpp --cce-aicore-only --cce-generic-addrspace=off --cce-auto-sync=off --cce-aicore-arch=dav-920r1-vec -DBSPUB_DAVINCI_BISHENGIR=1 -std=c++17 -O2 -c -emit-llvm -I %bishengir_src_root/lib/Template/lib/RegBase/Vector/include -I %bishengir_src_root/lib/Template/include -I %bishengir_src_root/lib/Template/lib -o %t/hifloat8.bc
// RUN: llvm-dis %t/hifloat8.bc -o - | FileCheck %s
// The shared NDDMA emitter must use byte movement for HiFloat8 too.
// Exercise both ranks and padding modes without a numeric HiFloat8 cast.
// CHECK-LABEL: define {{.*}}@hifloat8_nddma_2d(
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8
// CHECK-LABEL: define {{.*}}@hifloat8_nddma_3d(
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8
// CHECK-LABEL: define {{.*}}@hifloat8_nddma_nearest(
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8

//--- probe.cpp
#include "RegBase/DMAUtils.h"

extern "C" __aiv__ void hifloat8_nddma_2d(__gm__ hifloat8_t *src,
                                          __ubuf__ hifloat8_t *dst,
                                          uint32_t pad) {
  memref_t<__gm__ hifloat8_t, 2> input{src, src, 0, {2, 32}, {32, 1}};
  memref_t<__ubuf__ hifloat8_t, 2> output{dst, dst, 0, {2, 32}, {32, 1}};
  load_gm_to_ubuf_by_nddma<hifloat8_t, 2>(&input, &output, 0, 0, pad, 0, 0,
                                          false, false);
}

extern "C" __aiv__ void hifloat8_nddma_3d(__gm__ hifloat8_t *src,
                                          __ubuf__ hifloat8_t *dst,
                                          uint32_t pad) {
  memref_t<__gm__ hifloat8_t, 3> input{src, src, 0, {2, 2, 32}, {64, 32, 1}};
  memref_t<__ubuf__ hifloat8_t, 3> output{dst, dst, 0, {2, 2, 32}, {64, 32, 1}};
  load_gm_to_ubuf_by_nddma<hifloat8_t, 3>(&input, &output, 0, 0, pad, 0, 0,
                                          false, false);
}

extern "C" __aiv__ void hifloat8_nddma_nearest(__gm__ hifloat8_t *src,
                                               __ubuf__ hifloat8_t *dst) {
  memref_t<__gm__ hifloat8_t, 2> input{src, src, 0, {2, 32}, {32, 1}};
  memref_t<__ubuf__ hifloat8_t, 2> output{dst, dst, 0, {2, 32}, {32, 1}};
  load_gm_to_ubuf_by_nddma<hifloat8_t, 2>(&input, &output, 0, 0, 0, 0, 0, false,
                                          true);
}
