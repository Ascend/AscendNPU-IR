#include "compat/DMA/Cbuf/nd2nz.cpp"

extern "C" {
#if BSPUB_DAVINCI_BISHENGIR
DECLARE_ND2NZ(gm, cbuf, 2, 4, hifloat8_t);
#endif
DECLARE_ND2NZ(gm, cbuf, 2, 4, float8_e4m3_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, float8_e5m2_t);
#if BSPUB_DAVINCI_BISHENGIR
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, hifloat8_t);
#endif
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e4m3_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e5m2_t);
#if BSPUB_DAVINCI_BISHENGIR
REGISTE_ND2NZ(gm, cbuf, 2, 4, hifloat8_t);
#endif
REGISTE_ND2NZ(gm, cbuf, 2, 4, float8_e4m3_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, float8_e5m2_t);
#if BSPUB_DAVINCI_BISHENGIR
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, hifloat8_t);
#endif
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e4m3_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e5m2_t);
}
