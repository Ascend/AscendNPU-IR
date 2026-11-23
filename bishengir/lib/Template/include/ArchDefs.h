/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BISHENG_TPL_ARCHDEFS_H
#define BISHENG_TPL_ARCHDEFS_H

// Single point of convergence for dav architecture family and capability
// macros used across the Template library.
//
// The dav macros are predefined by ccec according to --cce-aicore-arch:
//   __DAV_C220__    Ascend910B/910C (c220, mem-based)
//   __DAV_C310__    Ascend910D       (c310, reg-based)
//   __DAV_M300__    (m300, reg-based)
//   __DAV_920R1__   Ascend960        (920r1, reg-based)
//
// Rules:
//   * Template sources must not spell __DAV_* directly; use the macros in
//     this header instead.
//   * Every macro in this header is always defined to 0 or 1, so it must be
//     tested with `#if BISHENG_TPL_*` / `#if !BISHENG_TPL_*`, never `#ifdef`.
//   * Prefer the family / capability macros. Use the per-arch membership
//     macros only for genuine per-arch intrinsic facts (e.g. an intrinsic
//     that exists on one arch only).
//   * When a new architecture is added, register it in all three sections
//     below.

// ---------------------------------------------------------------------------
// Architecture membership macros (1 when the current target is that arch).
// ---------------------------------------------------------------------------
#if defined(__DAV_C220__)
#define BISHENG_TPL_C220 1
#else
#define BISHENG_TPL_C220 0
#endif

#if defined(__DAV_C310__)
#define BISHENG_TPL_C310 1
#else
#define BISHENG_TPL_C310 0
#endif

#if defined(__DAV_M300__)
#define BISHENG_TPL_M300 1
#else
#define BISHENG_TPL_M300 0
#endif

#if defined(__DAV_920R1__)
#define BISHENG_TPL_920R1 1
#else
#define BISHENG_TPL_920R1 0
#endif

// ---------------------------------------------------------------------------
// Family macros
// ---------------------------------------------------------------------------

// Reg-based programming-model family: Ascend910D, Ascend960 and future
// reg-based backends. Mem-based architectures (c220) are not part of it.
#if BISHENG_TPL_C310 || BISHENG_TPL_920R1
#define BISHENG_TPL_REGBASE 1
#else
#define BISHENG_TPL_REGBASE 0
#endif

// Reg-based family plus the m300 reg-based arch. m300 shares the original
// RegBase (fixpipe / macro-gemm / DMA) forms with c310 but predates the
// newer c310/920r1 vector extensions, so it is kept out of
// BISHENG_TPL_REGBASE; this shortcut covers the paths the two share.
#if BISHENG_TPL_REGBASE || BISHENG_TPL_M300
#define BISHENG_TPL_REGBASE_OR_M300 1
#else
#define BISHENG_TPL_REGBASE_OR_M300 0
#endif

// Mem-based programming-model family (c220).
#if BISHENG_TPL_C220
#define BISHENG_TPL_MEMBASE 1
#else
#define BISHENG_TPL_MEMBASE 0
#endif

// ---------------------------------------------------------------------------
// Capability macros. Each macro is an independently extensible list of the
// architectures that provide the capability; register new backends here.
// ---------------------------------------------------------------------------

// MTE2 multi-ND2NZ / multi-DN2NZ / copy_gm_to_cbuf_align_v2 intrinsics take a
// trailing pre_allocation parameter (10-arg nd2nz/dn2nz, 12-arg align_v2).
#if BISHENG_TPL_920R1
#define BISHENG_TPL_HAS_DMA_PRE_ALLOCATION 1
#else
#define BISHENG_TPL_HAS_DMA_PRE_ALLOCATION 0
#endif

// Fixpipe intrinsics (copy_matrix_cc_to_gm/cbuf/ub) use the expanded
// overloads with quant_pre_rnd / l2_cache_ctl / dual_dst_ctl parameters; see
// the per-overload comments in FixpipeUtils.h.
#if BISHENG_TPL_920R1
#define BISHENG_TPL_HAS_FIXPIPE_EXPANDED_OVERLOAD 1
#else
#define BISHENG_TPL_HAS_FIXPIPE_EXPANDED_OVERLOAD 0
#endif

// The AVE shift-amount operand accepts a vector type (vector_u32).
#if BISHENG_TPL_920R1
#define BISHENG_TPL_AVE_SHIFT_AMOUNT_IS_VECTOR 1
#else
#define BISHENG_TPL_AVE_SHIFT_AMOUNT_IS_VECTOR 0
#endif

// The runtime does not pull in ccelib print by itself; the template has to
// bootstrap it (define __CCE_ENABLE_PRINT_AICORE__ and include
// ccelib/print/print.h).
#if BISHENG_TPL_920R1
#define BISHENG_TPL_NEEDS_PRINT_BOOTSTRAP 1
#else
#define BISHENG_TPL_NEEDS_PRINT_BOOTSTRAP 0
#endif

// The reg-family MTE2 multi-ND2NZ / multi-DN2NZ path uses the
// set_mte2_nz_para config encoder plus a copy_gm_to_cbuf_multi_nd2nz
// overload (c310: 9-arg; 920r1: 10-arg with pre_allocation). Other
// architectures use the older typed multi_nd2nz_b8/b16/b32s form.
#if BISHENG_TPL_C310 || BISHENG_TPL_920R1
#define BISHENG_TPL_HAS_MTE2_NZ_PARA_MULTI_ND2NZ 1
#else
#define BISHENG_TPL_HAS_MTE2_NZ_PARA_MULTI_ND2NZ 0
#endif

// Fixpipe batch drains are driven by the sticky set_nd_para register on
// c310/m300; dav-920r1 has no fixpipe ND_PARA and must drain batch rows one
// at a time.
#if BISHENG_TPL_C310 || BISHENG_TPL_M300
#define BISHENG_TPL_HAS_FIXPIPE_ND_PARA 1
#else
#define BISHENG_TPL_HAS_FIXPIPE_ND_PARA 0
#endif

// Fixpipe intrinsics take the XT1 extended arg list (dual_dst_ctl /
// sub_blockid / l2_cache_ctl / clip_relu_pre ...) rather than the plain
// pre-quant form. dav-920r1 keeps the XT1 list but through the expanded
// overload (see BISHENG_TPL_HAS_FIXPIPE_EXPANDED_OVERLOAD).
#if BISHENG_TPL_C310 || BISHENG_TPL_M300
#define BISHENG_TPL_HAS_FIXPIPE_XT1_ARGS 1
#else
#define BISHENG_TPL_HAS_FIXPIPE_XT1_ARGS 0
#endif

#endif // BISHENG_TPL_ARCHDEFS_H
