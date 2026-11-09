// The drain sequence is adapted from Ascend/msopprof's instrprof_end.cpp.
// Copyright (c) 2025 Huawei Technologies Co.,Ltd.
// Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
// Provided on an "AS IS" basis, without warranties of any kind.
#include <cstdint>

extern "C" {
__aicore__ __attribute__((always_inline)) void
proton_scope_boundary(std::int32_t regionId) {
#ifndef ENABLE_CPU_TRACE_INTRINSIC
  pipe_barrier(PIPE_ALL);
  // 0xfff is the outer-scope drain; 0x10000 is a late LLVM epilogue call.
  if (regionId != 0x10000 && regionId != 0xfff)
    __dfx_region(static_cast<std::uint16_t>(regionId), PIPE_S);
  if (regionId == 0x10000 || regionId == 0xfff) {
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
  }
#endif
}
}
