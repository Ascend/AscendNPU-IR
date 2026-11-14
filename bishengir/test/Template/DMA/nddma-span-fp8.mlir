// REQUIRES: enable-lir-compile
// RUN: split-file %s %t
// RUN: %bisheng_compiler -x cce %t/probe.cpp --cce-aicore-only --cce-generic-addrspace=off --cce-auto-sync=off --cce-aicore-arch=dav-920r1-vec -std=c++17 -O2 -c -emit-llvm -I %bishengir_src_root/lib/Template/lib/RegBase/Vector/include -I %bishengir_src_root/lib/Template/include -I %bishengir_src_root/lib/Template/lib -o %t/fp8.bc
// RUN: llvm-dis %t/fp8.bc -o %t/fp8.ll
// RUN: FileCheck %s < %t/fp8.ll
// Numeric 1.5 is encoded as 0x3c (e4m3) and 0x3e (e5m2).
// The f32 ABI carries zero-extended FP8 bits, NOT a numeric f32 value.
// CHECK-LABEL: define {{.*}}@span_e4m3_2_dma(
// CHECK: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 60)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e4m3_2_zero(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: store i8 60,
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e4m3_2_wide(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: %[[PAD:.*]] = phi i8 {{.*}}[ 60,
// CHECK: store i8 %[[PAD]],
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e4m3_3_dma(
// CHECK: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 60)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e4m3_3_zero(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: store i8 60,
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e4m3_3_wide(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: %[[PAD:.*]] = phi i8 {{.*}}[ 60,
// CHECK: store i8 %[[PAD]],
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e5m2_2_dma(
// CHECK: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 62)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e5m2_2_zero(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: store i8 62,
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e5m2_2_wide(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: %[[PAD:.*]] = phi i8 {{.*}}[ 62,
// CHECK: store i8 %[[PAD]],
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e5m2_3_dma(
// CHECK: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 62)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB.V920.b8
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e5m2_3_zero(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: store i8 62,
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@span_e5m2_3_wide(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: %[[PAD:.*]] = phi i8 {{.*}}[ 62,
// CHECK: store i8 %[[PAD]],
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK-NOT: @llvm.hivm.trap
// CHECK: ret void

//--- probe.cpp
#include "RegBase/DMAUtils.h"
extern "C" __aiv__ void span_e4m3_2_dma(__gm__ float8_e4m3_t *s, __ubuf__ float8_e4m3_t *d) {
 memref_t<__gm__ float8_e4m3_t,2> a{s,s,0,{1,4},{4,1}};
 memref_t<__ubuf__ float8_e4m3_t,2> b{d,d,0,{2,9},{16,1}};
 uint32_t raw = 60;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e4m3_t,2>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,1);
}
extern "C" __aiv__ void span_e4m3_2_zero(__gm__ float8_e4m3_t *s, __ubuf__ float8_e4m3_t *d) {
 memref_t<__gm__ float8_e4m3_t,2> a{s,s,0,{1,0},{4,1}};
 memref_t<__ubuf__ float8_e4m3_t,2> b{d,d,0,{2,5},{12,1}};
 uint32_t raw = 60;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e4m3_t,2>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,1);
}
extern "C" __aiv__ void span_e4m3_2_wide(__gm__ float8_e4m3_t *s, __ubuf__ float8_e4m3_t *d) {
 memref_t<__gm__ float8_e4m3_t,2> a{s,s,0,{1,4},{4,1}};
 memref_t<__ubuf__ float8_e4m3_t,2> b{d,d,0,{2,307},{314,1}};
 uint32_t raw = 60;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e4m3_t,2>(&a,&b,PadMode::Value,carrier,300,3,EvictionPolicy::EvictLast,1);
}
extern "C" __aiv__ void span_e4m3_3_dma(__gm__ float8_e4m3_t *s, __ubuf__ float8_e4m3_t *d) {
 memref_t<__gm__ float8_e4m3_t,3> a{s,s,0,{1,1,4},{4,4,1}};
 memref_t<__ubuf__ float8_e4m3_t,3> b{d,d,0,{2,2,9},{32,16,1}};
 uint32_t raw = 60;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e4m3_t,3>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,3);
}
extern "C" __aiv__ void span_e4m3_3_zero(__gm__ float8_e4m3_t *s, __ubuf__ float8_e4m3_t *d) {
 memref_t<__gm__ float8_e4m3_t,3> a{s,s,0,{1,1,0},{4,4,1}};
 memref_t<__ubuf__ float8_e4m3_t,3> b{d,d,0,{2,2,5},{24,12,1}};
 uint32_t raw = 60;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e4m3_t,3>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,3);
}
extern "C" __aiv__ void span_e4m3_3_wide(__gm__ float8_e4m3_t *s, __ubuf__ float8_e4m3_t *d) {
 memref_t<__gm__ float8_e4m3_t,3> a{s,s,0,{1,1,4},{4,4,1}};
 memref_t<__ubuf__ float8_e4m3_t,3> b{d,d,0,{2,2,307},{628,314,1}};
 uint32_t raw = 60;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e4m3_t,3>(&a,&b,PadMode::Value,carrier,300,3,EvictionPolicy::EvictLast,3);
}
extern "C" __aiv__ void span_e5m2_2_dma(__gm__ float8_e5m2_t *s, __ubuf__ float8_e5m2_t *d) {
 memref_t<__gm__ float8_e5m2_t,2> a{s,s,0,{1,4},{4,1}};
 memref_t<__ubuf__ float8_e5m2_t,2> b{d,d,0,{2,9},{16,1}};
 uint32_t raw = 62;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e5m2_t,2>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,1);
}
extern "C" __aiv__ void span_e5m2_2_zero(__gm__ float8_e5m2_t *s, __ubuf__ float8_e5m2_t *d) {
 memref_t<__gm__ float8_e5m2_t,2> a{s,s,0,{1,0},{4,1}};
 memref_t<__ubuf__ float8_e5m2_t,2> b{d,d,0,{2,5},{12,1}};
 uint32_t raw = 62;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e5m2_t,2>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,1);
}
extern "C" __aiv__ void span_e5m2_2_wide(__gm__ float8_e5m2_t *s, __ubuf__ float8_e5m2_t *d) {
 memref_t<__gm__ float8_e5m2_t,2> a{s,s,0,{1,4},{4,1}};
 memref_t<__ubuf__ float8_e5m2_t,2> b{d,d,0,{2,307},{314,1}};
 uint32_t raw = 62;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e5m2_t,2>(&a,&b,PadMode::Value,carrier,300,3,EvictionPolicy::EvictLast,1);
}
extern "C" __aiv__ void span_e5m2_3_dma(__gm__ float8_e5m2_t *s, __ubuf__ float8_e5m2_t *d) {
 memref_t<__gm__ float8_e5m2_t,3> a{s,s,0,{1,1,4},{4,4,1}};
 memref_t<__ubuf__ float8_e5m2_t,3> b{d,d,0,{2,2,9},{32,16,1}};
 uint32_t raw = 62;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e5m2_t,3>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,3);
}
extern "C" __aiv__ void span_e5m2_3_zero(__gm__ float8_e5m2_t *s, __ubuf__ float8_e5m2_t *d) {
 memref_t<__gm__ float8_e5m2_t,3> a{s,s,0,{1,1,0},{4,4,1}};
 memref_t<__ubuf__ float8_e5m2_t,3> b{d,d,0,{2,2,5},{24,12,1}};
 uint32_t raw = 62;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e5m2_t,3>(&a,&b,PadMode::Value,carrier,2,3,EvictionPolicy::EvictLast,3);
}
extern "C" __aiv__ void span_e5m2_3_wide(__gm__ float8_e5m2_t *s, __ubuf__ float8_e5m2_t *d) {
 memref_t<__gm__ float8_e5m2_t,3> a{s,s,0,{1,1,4},{4,4,1}};
 memref_t<__ubuf__ float8_e5m2_t,3> b{d,d,0,{2,2,307},{628,314,1}};
 uint32_t raw = 62;
 float carrier = *reinterpret_cast<float *>(&raw);
 load_gm_to_ubuf_span_core<float8_e5m2_t,3>(&a,&b,PadMode::Value,carrier,300,3,EvictionPolicy::EvictLast,3);
}
