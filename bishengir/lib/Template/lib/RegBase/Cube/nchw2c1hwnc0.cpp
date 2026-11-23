#include "compat/DMA/Cbuf/nchw2c1hwnc0.cpp"

#if defined(__DAV_C310__)
extern "C" {
REGISTER_NCHW2C1HWNC0(half);
REGISTER_NCHW2C1HWNC0(float);
REGISTER_NCHW2C1HWNC0(bfloat16_t);
}
#endif
