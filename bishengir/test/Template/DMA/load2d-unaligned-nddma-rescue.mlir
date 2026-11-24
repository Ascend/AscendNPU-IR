// REQUIRES: enable-lir-compile
// RUN: split-file %s %t
// RUN: %bisheng_compiler -x cce %t/probe.cpp --cce-aicore-only --cce-generic-addrspace=off --cce-auto-sync=off --cce-aicore-arch=dav-920r1-vec -std=c++17 -O2 -c -emit-llvm -I %bishengir_src_root -I %bishengir_src_root/lib/Template/lib/RegBase/Vector/include -I %bishengir_src_root/lib/Template/include -I %bishengir_src_root/lib/Template/lib -o %t/probe.bc
// RUN: llvm-dis %t/probe.bc -o %t/probe.ll
// RUN: FileCheck %s < %t/probe.ll
// Pin the real-machine arch as well: the A5/950 template is meta_op.aiv.c310,
// and the admission lives in BISHENG_TPL_REGBASE scope so both archs must
// emit the same routing.
// RUN: %bisheng_compiler -x cce %t/probe.cpp --cce-aicore-only --cce-generic-addrspace=off --cce-auto-sync=off --cce-aicore-arch=dav-c310-vec -std=c++17 -O2 -c -emit-llvm -I %bishengir_src_root -I %bishengir_src_root/lib/Template/lib/RegBase/Vector/include -I %bishengir_src_root/lib/Template/include -I %bishengir_src_root/lib/Template/lib -o %t/probe_c310.bc
// RUN: llvm-dis %t/probe_c310.bc -o %t/probe_c310.ll
// RUN: FileCheck %s < %t/probe_c310.ll
//
// A GM->UB 2D load whose UB destination is not 32B aligned cannot start a
// copy_gm_to_ubuf_align_v2 burst and falls back to a per-element scalar copy.
// nddma_out_to_ub addresses UB at element granularity, so the window moves as
// one descriptor instead, carrying no padding counts and no pad value: it
// writes exactly the valid window, as the scalar copy does.
//
// The pad value is the discriminator these probes pin: a PadMode::Value load
// with pad 0.5 must reach the NDDMA with SET.PAD.VAL.NDDMA(0), never with the
// forwarded pattern (half 0.5 = 14336, float 0.5 = 1056964608).
//
// Note on the negative probes: the UB alignment test itself is dynamic, so for
// an *admitted* shape both the descriptor and the aligned ALIGN.V2 burst are
// emitted, whichever one a given pointer takes at run time. The discriminating
// signal is the admission predicate, which folds at compile time: a rejected
// shape emits no ND.DMA at all.

// CHECK-LABEL: define {{.*}}@issue1014_f16_nddma(
// CHECK-NOT: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 14336)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB{{.*}}b16
// CHECK-NOT: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 14336)
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@v11_null_padmode_nddma(
// CHECK: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 0)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB{{.*}}b32
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@f32_value_nddma(
// CHECK-NOT: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 1056964608)
// CHECK: @llvm.hivm.ND.DMA.OUT.TO.UB{{.*}}b32
// CHECK-NOT: @llvm.hivm.SET.PAD.VAL.NDDMA(i64 1056964608)
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@strided_inner_scalar(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: load i16
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@b64_scalar(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@wide_dst_stride_scalar(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: load i16
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@overlapping_dst_rows_scalar(
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: load i16
// CHECK-NOT: @llvm.hivm.ND.DMA
// CHECK: ret void
// CHECK-LABEL: define {{.*}}@aligned_value_mov_unchanged(
// CHECK: @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2
// CHECK: ret void

//--- probe.cpp
#include "ArchDefs.h"
#include "RegBase/DMAUtils.h"
// Compile only the cores under test from Copy2D.cpp: neutralize the
// extern "C" registration block so the emitted bitcode keeps just the
// instantiations below (the all-dtype registration bulk trips the host
// llvm-dis bitcode reader).
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

// f16, dst start 257 elements = 514B (not 32B aligned), lp = 1,
// PadMode::Value: moved by NDDMA, without the pad value, so the lp slot in
// front of dst->offset keeps the contents the scalar copy left there.
extern "C" __aiv__ void issue1014_f16_nddma(__gm__ half *s, __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 257}, {257, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 257, {15, 257}, {1024, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 1,
                                EvictionPolicy::EvictFirst);
}

// f32, dst start 148 elements = 592B, lp = 4, PadMode::Null: admitted as
// well, because the descriptor writes no padding - SET.PAD.VAL.NDDMA(0) and
// zero lp/rp mean the four elements in front of dst->offset, which may hold a
// neighbouring tile's live data, are not touched.
extern "C" __aiv__ void v11_null_padmode_nddma(__gm__ float *s,
                                               __ubuf__ float *d) {
  memref_t<__gm__ float, 2> a{s, s, 0, {15, 100}, {100, 1}};
  memref_t<__ubuf__ float, 2> b{d, d, 148, {15, 100}, {256, 1}};
  load_gm_to_ubuf_2d_core<float>(&a, &b, PadMode::Null, 0.0f, 4,
                                 EvictionPolicy::EvictFirst);
}

// Same geometry with PadMode::Value and a non-zero pad value: the pad value
// must not reach the descriptor either.
extern "C" __aiv__ void f32_value_nddma(__gm__ float *s, __ubuf__ float *d) {
  memref_t<__gm__ float, 2> a{s, s, 0, {15, 100}, {100, 1}};
  memref_t<__ubuf__ float, 2> b{d, d, 148, {15, 100}, {256, 1}};
  load_gm_to_ubuf_2d_core<float>(&a, &b, PadMode::Value, 0.5f, 4,
                                 EvictionPolicy::EvictFirst);
}

// Strided innermost axis on both sides: outside the admitted two-level
// descriptor family, scalar fallback.
extern "C" __aiv__ void strided_inner_scalar(__gm__ half *s,
                                             __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 257}, {514, 2}};
  memref_t<__ubuf__ half, 2> b{d, d, 257, {15, 257}, {2048, 2}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 1,
                                EvictionPolicy::EvictFirst);
}

// b64: nddma_out_to_ub has no 64-bit instance and the widened b32 view is
// owned by the existing padded paths, so b64 keeps the scalar fallback.
// PadMode::Null keeps the b64 pad fixup (apply_padding_b64) out of this
// translation unit: it is extracted into a VF function whose attributes the
// host llvm-dis cannot read.
extern "C" __aiv__ void b64_scalar(__gm__ int64_t *s, __ubuf__ int64_t *d) {
  memref_t<__gm__ int64_t, 2> a{s, s, 0, {15, 257}, {257, 1}};
  memref_t<__ubuf__ int64_t, 2> b{d, d, 257, {15, 257}, {1024, 1}};
  load_gm_to_ubuf_2d_core<int64_t>(&a, &b, PadMode::Null, (int64_t)0, 1,
                                   EvictionPolicy::EvictFirst);
}

// Destination row step beyond the 20-bit descriptor field (0x100000): the
// admission must reject it instead of truncating.
extern "C" __aiv__ void wide_dst_stride_scalar(__gm__ half *s,
                                               __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 257}, {257, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 257, {15, 257}, {0x100000, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 1,
                                EvictionPolicy::EvictFirst);
}

// Destination rows overlap (row step < row width): the descriptor would
// make the overlap order observable, so it stays scalar.
extern "C" __aiv__ void overlapping_dst_rows_scalar(__gm__ half *s,
                                                    __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 257}, {257, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 257, {15, 257}, {256, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 1,
                                EvictionPolicy::EvictFirst);
}

// Aligned destination (272 elements = 544B) with PadMode::Value and no left
// padding: unchanged, still the ALIGN.V2 MOV path. The rescue descriptor is
// also emitted for this shape - the alignment test is dynamic - but an
// aligned pointer never takes it.
extern "C" __aiv__ void aligned_value_mov_unchanged(__gm__ half *s,
                                                    __ubuf__ half *d) {
  memref_t<__gm__ half, 2> a{s, s, 0, {15, 272}, {272, 1}};
  memref_t<__ubuf__ half, 2> b{d, d, 272, {15, 272}, {1024, 1}};
  load_gm_to_ubuf_2d_core<half>(&a, &b, PadMode::Value, (half)0.5, 0,
                                EvictionPolicy::EvictFirst);
}
