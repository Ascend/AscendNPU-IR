// REQUIRES: enable-lir-compile
// RUN: split-file %s %t
// RUN: %bisheng_compiler -x cce %t/probe.cpp --cce-aicore-only --cce-generic-addrspace=off --cce-auto-sync=off --cce-aicore-arch=dav-920r1-vec -std=c++17 -O2 -c -emit-llvm -I %bishengir_src_root -I %bishengir_src_root/lib/Template/lib/RegBase/Vector/include -I %bishengir_src_root/lib/Template/include -I %bishengir_src_root/lib/Template/lib -o %t/probe.bc
// RUN: llvm-dis %t/probe.bc -o %t/probe.ll
// RUN: FileCheck %s < %t/probe.ll
//
// Issue #1014 regression: the load_gm_to_ubuf_2d_core alignment gate must
// check the left-padding pull-back destination (dst + offset - lp*stride1),
// because load_gm_to_ubuf_intrin_core starts the DMA at the pulled-back
// address. The pointwise-cat large_4096 second input lands at UB offset 257
// (fp16, 514B) with left_padding_num=1; the pulled-back start (512B) is
// 32B aligned and must stay on the MOV DMA path instead of the scalar loop.
// The N=256/272 neighbours keep lp=0 with an already aligned offset, and a
// pull-back that still misses 32B alignment must keep the scalar fallback.

// The UB base pointer is a runtime value, so the admission gate stays a
// dynamic branch and every probe body statically contains both the DMA and
// the scalar-fallback blocks. The regression signal is SSA-level: the gate
// must check exactly the address the DMA uses, i.e. the pulled-back
// getelementptr (offset - left_padding_num) whose result feeds BOTH the
// alignment ptrtoint and the ALIGN.V2 destination. On the buggy gate the
// checked gep is the raw offset and no ptrtoint binds to the pulled-back
// gep, so these checks fail.

// CHECK-LABEL: define {{.*}}@lp1_dma(
// CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds half, ptr addrspace(6) {{.*}}, i64 256
// CHECK: [[PTI:%[0-9]+]] = ptrtoint ptr addrspace(6) [[GEP]] to i64
// CHECK: and i64 [[PTI]], 31
// CHECK: @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.{{[^ (]*}}(ptr addrspace(6) [[GEP]],
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@lp2_scalar(
// CHECK: [[GEP2:%[0-9]+]] = getelementptr inbounds half, ptr addrspace(6) {{.*}}, i64 255
// CHECK: ptrtoint ptr addrspace(6) [[GEP2]] to i64
// CHECK: load i16
// CHECK: store i16
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@n256_dma(
// CHECK: [[GEP3:%[0-9]+]] = getelementptr inbounds half, ptr addrspace(6) {{.*}}, i64 256
// CHECK: ptrtoint ptr addrspace(6) [[GEP3]] to i64
// CHECK: @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.{{[^ (]*}}(ptr addrspace(6) [[GEP3]],
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@n272_dma(
// CHECK: [[GEP4:%[0-9]+]] = getelementptr inbounds half, ptr addrspace(6) {{.*}}, i64 272
// CHECK: ptrtoint ptr addrspace(6) [[GEP4]] to i64
// CHECK: @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.{{[^ (]*}}(ptr addrspace(6) [[GEP4]],
// CHECK: ret void

//--- probe.cpp
#include "ArchDefs.h"
#include "RegBase/DMAUtils.h"
// Compile only the cores under test from Copy2D.cpp: neutralize the
// extern "C" registration block so the emitted bitcode keeps just the
// half instantiations below (the all-dtype registration bulk trips the
// host llvm-dis bitcode reader).
#undef REGISTE_DMA_SPAN_LOAD
#undef REGISTE_DMA_LOAD
#undef REGISTE_DMA_LOAD_FP8
#undef REGISTE_DMA_STORE
#undef REGISTE_DMA_UB_COPY
#define REGISTE_DMA_SPAN_LOAD(dim, type)
#define REGISTE_DMA_LOAD(dim, type)
#define REGISTE_DMA_LOAD_FP8(dim, type)
#define REGISTE_DMA_STORE(dim, type)
#define REGISTE_DMA_UB_COPY(src_scope, dst_scope, dim, type)
#include "lib/Template/lib/RegBase/Vector/Copy2D.cpp"

// Issue #1014 hit case: dst start 257 (514B, unaligned), lp=1 pulls the DMA
// start back to 512B. The gate must admit the MOV DMA path.
extern "C" __aiv__ void lp1_dma(__gm__ half *s, __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 257}, {257, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 257, {15, 257}, {1024, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 1,
                                EvictionPolicy::EvictFirst);
}

// Pull-back still misses 32B alignment (257 - 2 = 255 elems, 510B): the
// scalar fallback must be kept for correctness.
extern "C" __aiv__ void lp2_scalar(__gm__ half *s, __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 257}, {257, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 257, {15, 257}, {1024, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 2,
                                EvictionPolicy::EvictFirst);
}

// Neighbour N=256: offset 256 (512B) aligned, lp=0. DMA, unchanged.
extern "C" __aiv__ void n256_dma(__gm__ half *s, __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 256}, {256, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 256, {15, 256}, {1024, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 0,
                                EvictionPolicy::EvictFirst);
}

// Neighbour N=272: offset 272 (544B) aligned, lp=0. DMA, unchanged.
extern "C" __aiv__ void n272_dma(__gm__ half *s, __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 272}, {272, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 272, {15, 272}, {1024, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 0,
                                EvictionPolicy::EvictFirst);
}
