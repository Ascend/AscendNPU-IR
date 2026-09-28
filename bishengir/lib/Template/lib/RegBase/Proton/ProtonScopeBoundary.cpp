#include <cstdint>

extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_proton_scope_boundary(std::int32_t regionId) {
#ifndef ENABLE_CPU_TRACE_INTRINSIC
  pipe_barrier(PIPE_ALL);
  __dfx_region(static_cast<std::uint16_t>(regionId), PIPE_S);
#endif
}
}
