#include "../../Fixpipe/Fixpipe.cpp"

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2nd
//===-------------------------------------------------------------------===//

REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, int8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, uint8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);

REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, uint8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);

#if defined(__DAV_C310__)
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, half, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, bfloat16_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, int8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, uint8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, float, float, nz2nd, TransformMode::NZ_2_ND);

REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, half, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, int8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, uint8_t, nz2nd, TransformMode::NZ_2_ND);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 4, 2, int32_t, int32_t, nz2nd, TransformMode::NZ_2_ND);

//===-------------------------------------------------------------------===//
// fixpipe, 2 dim to 2 dim, nz2nz normal
//===-------------------------------------------------------------------===//
REGISTE_FIXPIPE(cc, ubuf, 2, 2, float, half, normal, TransformMode::NORMAL);
REGISTE_FIXPIPE(cc, ubuf, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

REGISTE_FIXPIPE_DUAL(cc, ubuf, 2, 2, float, half, normal, TransformMode::NORMAL);
REGISTE_FIXPIPE_DUAL(cc, ubuf, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

REGISTE_FIXPIPE(cc, cbuf, 2, 2, float, half, normal, TransformMode::NORMAL);
REGISTE_FIXPIPE(cc, cbuf, 2, 2, float, bfloat16_t, normal, TransformMode::NORMAL);

//===-------------------------------------------------------------------===//
// fixpipe, 4 dim to 2 dim, nz2dn
//===-------------------------------------------------------------------===//
REGISTE_FIXPIPE(cc, gm, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, gm, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, gm, 4, 2, float, float, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, gm, 4, 2, int32_t, int32_t, nz2dn, TransformMode::NZ_2_DN);

REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, int8_t, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, uint8_t, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, float, float, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, half, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int8_t, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, uint8_t, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, ubuf, 4, 2, int32_t, int32_t, nz2dn, TransformMode::NZ_2_DN);

REGISTE_FIXPIPE(cc, cbuf, 4, 2, float, half, nz2dn, TransformMode::NZ_2_DN);
REGISTE_FIXPIPE(cc, cbuf, 4, 2, float, bfloat16_t, nz2dn, TransformMode::NZ_2_DN);
#endif // defined(__DAV_C310__)