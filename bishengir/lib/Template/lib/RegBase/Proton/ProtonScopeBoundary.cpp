#include <cstdint>

namespace {
// Keep these control values aligned with HIVMProtonOps.h.
constexpr std::int32_t kOuterScopeDrain = 0xfff;
constexpr std::int32_t kKernelExitDrain = 0x10000;
} // namespace

extern "C" {
__aicore__ __attribute__((always_inline)) void
proton_scope_boundary(std::int32_t regionId) {
#ifndef ENABLE_CPU_TRACE_INTRINSIC
  pipe_barrier(PIPE_ALL);
  if (regionId != kKernelExitDrain && regionId != kOuterScopeDrain) {
    __dfx_region(static_cast<std::uint16_t>(regionId), PIPE_S);
    return;
  }
  // Experimental protection waits stay outside all measured user scopes.
  // This mitigates, but does not yet eliminate, incomplete captures on A5.
  asm volatile("bar.all\n\t.rept 3500\n\tNOP\n\t.endr" ::: "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xd88) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xd99) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xdaa) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xdbb) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xdcc) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xddd) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xdee) : "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xdff) : "memory");
  asm volatile(".rept 3500\n\tNOP\n\t.endr" ::: "memory");
  asm volatile("DFX_REGION.S %0\n\tnop" ::"l"(0xdff) : "memory");
#endif
}
}
