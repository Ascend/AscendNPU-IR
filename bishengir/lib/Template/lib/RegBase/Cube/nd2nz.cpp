#include "../../DMA/Cbuf/nd2nz.cpp"

extern "C" {
DECLARE_ND2NZ(gm, cbuf, 2, 4, float8_e4m3_t);
DECLARE_ND2NZ(gm, cbuf, 2, 4, float8_e5m2_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e4m3_t);
DECLARE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e5m2_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, float8_e4m3_t);
REGISTE_ND2NZ(gm, cbuf, 2, 4, float8_e5m2_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e4m3_t);
REGISTE_ND2NZ_FORBIAS(gm, cbuf, 2, 4, float8_e5m2_t);
}
