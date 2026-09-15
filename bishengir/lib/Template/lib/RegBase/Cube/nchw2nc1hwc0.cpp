#include "compat/DMA/Cbuf/nchw2nc1hwc0.cpp"

#if defined(__DAV_C310__)
extern "C" {
REGISTER_NCHW2NC1HWC0(half);
REGISTER_NCHW2NC1HWC0(float);
REGISTER_NCHW2NC1HWC0(bfloat16_t);
}
#endif
