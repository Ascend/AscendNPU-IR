// MLIR C interfaces for transport-specific SHMEM operations.

#ifndef BISHENGIR_TEMPLATE_SHMEM_TRANSPORT_H
#define BISHENGIR_TEMPLATE_SHMEM_TRANSPORT_H

// Included after the SHMEM device headers by Shmem.cpp. Keep these wrappers in
// the same translation unit because those headers define device globals.
extern "C" {

// RoCE uses the same runtime-configured UB workspace and event ID as generic
// SHMEM RMA. Transfer counts are in elements; payload memrefs retain offsets.
#define ACLSHMEM_ROCE_GET_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_roce_get_nbi_##NAME(memref_t<__gm__ TYPE, 1> *dst,    \
                                             memref_t<__gm__ TYPE, 1> *src,    \
                                             uint32_t elem_size, int32_t pe) { \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ TYPE *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_get_nbi<TYPE>(dst->aligned + dst->offset,                   \
                                 src->aligned + src->offset, buf, elem_size,   \
                                 pe, sync_id);                                 \
  }

#define ACLSHMEM_ROCE_PUT_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_roce_put_nbi_##NAME(memref_t<__gm__ TYPE, 1> *dst,    \
                                             memref_t<__gm__ TYPE, 1> *src,    \
                                             uint32_t elem_size, int32_t pe) { \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ TYPE *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_put_nbi<TYPE>(dst->aligned + dst->offset,                   \
                                 src->aligned + src->offset, buf, elem_size,   \
                                 pe, sync_id);                                 \
  }

#define ACLSHMEM_ROCE_QUIET_WRAPPER(NAME, TYPE)                                \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_roce_quiet_##NAME(uint32_t pe) {                      \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ TYPE *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_quiet<TYPE>(pe, buf, sync_id);                              \
  }

// Match the existing PIPE_S UDMA put ABI: no UB workspace or event is needed.
#define ACLSHMEM_UDMA_GET_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_udma_get_nbi_##NAME(memref_t<__gm__ TYPE, 1> *dst,    \
                                             memref_t<__gm__ TYPE, 1> *src,    \
                                             uint32_t elem_size, int32_t pe) { \
    aclshmemx_udma_get_nbi<TYPE, PIPE_S>(                                      \
        dst->aligned + dst->offset, src->aligned + src->offset,                \
        (__ubuf__ TYPE *)nullptr, elem_size, pe, 0);                           \
  }

#define ACLSHMEM_UDMA_PUT_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_udma_put_nbi_##NAME(memref_t<__gm__ TYPE, 1> *dst,    \
                                             memref_t<__gm__ TYPE, 1> *src,    \
                                             uint32_t elem_size, int32_t pe) { \
    aclshmemx_udma_put_nbi<TYPE, PIPE_S>(                                      \
        dst->aligned + dst->offset, src->aligned + src->offset,                \
        (__ubuf__ TYPE *)nullptr, elem_size, pe, 0);                           \
  }

// Relay wrappers are opt-in via BISHENGIR_SHMEM_RELAY_SUPPORT for C310 and
// require a matching relay-enabled runtime. C220 never enables these wrappers.
#ifdef ACLSHMEM_RELAY_SUPPORT
#define ACLSHMEM_UDMA_RELAY_GET_NBI_WRAPPER(NAME, TYPE)                        \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_udma_relay_get_nbi_##NAME(                            \
      memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,            \
      uint32_t elem_size, int32_t pe, int32_t relay_pe) {                      \
    aclshmemx_udma_relay_get_nbi<TYPE, PIPE_S>(                                \
        dst->aligned + dst->offset, src->aligned + src->offset,                \
        (__ubuf__ TYPE *)nullptr, elem_size, pe, relay_pe, 0);                 \
  }

#define ACLSHMEM_UDMA_RELAY_PUT_NBI_WRAPPER(NAME, TYPE)                        \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmemx_udma_relay_put_nbi_##NAME(                            \
      memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,            \
      uint32_t elem_size, int32_t pe, int32_t relay_pe) {                      \
    aclshmemx_udma_relay_put_nbi<TYPE, PIPE_S>(                                \
        dst->aligned + dst->offset, src->aligned + src->offset,                \
        (__ubuf__ TYPE *)nullptr, elem_size, pe, relay_pe, 0);                 \
  }

#define ACLSHMEM_UDMA_RELAY_WRAPPERS(NAME, TYPE)                               \
  ACLSHMEM_UDMA_RELAY_GET_NBI_WRAPPER(NAME, TYPE)                              \
  ACLSHMEM_UDMA_RELAY_PUT_NBI_WRAPPER(NAME, TYPE)
#else
#define ACLSHMEM_UDMA_RELAY_WRAPPERS(NAME, TYPE)
#endif

#define ACLSHMEM_TRANSPORT_WRAPPERS(NAME, TYPE)                                \
  ACLSHMEM_ROCE_GET_NBI_WRAPPER(NAME, TYPE)                                    \
  ACLSHMEM_ROCE_PUT_NBI_WRAPPER(NAME, TYPE)                                    \
  ACLSHMEM_ROCE_QUIET_WRAPPER(NAME, TYPE)                                      \
  ACLSHMEM_UDMA_GET_NBI_WRAPPER(NAME, TYPE)                                    \
  ACLSHMEM_UDMA_PUT_NBI_WRAPPER(NAME, TYPE)                                    \
  ACLSHMEM_UDMA_RELAY_WRAPPERS(NAME, TYPE)

ACLSHMEM_TRANSPORT_WRAPPERS(half, half)
ACLSHMEM_TRANSPORT_WRAPPERS(float, float)
ACLSHMEM_TRANSPORT_WRAPPERS(int8, int8_t)
ACLSHMEM_TRANSPORT_WRAPPERS(int16, int16_t)
ACLSHMEM_TRANSPORT_WRAPPERS(int32, int32_t)
ACLSHMEM_TRANSPORT_WRAPPERS(int64, int64_t)
ACLSHMEM_TRANSPORT_WRAPPERS(uint8, uint8_t)
ACLSHMEM_TRANSPORT_WRAPPERS(uint16, uint16_t)
ACLSHMEM_TRANSPORT_WRAPPERS(uint32, uint32_t)
ACLSHMEM_TRANSPORT_WRAPPERS(uint64, uint64_t)
ACLSHMEM_TRANSPORT_WRAPPERS(char, char)
ACLSHMEM_TRANSPORT_WRAPPERS(bfloat16, bfloat16_t)

#undef ACLSHMEM_UDMA_RELAY_WRAPPERS
#undef ACLSHMEM_TRANSPORT_WRAPPERS
#undef ACLSHMEM_ROCE_GET_NBI_WRAPPER
#undef ACLSHMEM_ROCE_PUT_NBI_WRAPPER
#undef ACLSHMEM_ROCE_QUIET_WRAPPER
#undef ACLSHMEM_UDMA_GET_NBI_WRAPPER
#undef ACLSHMEM_UDMA_PUT_NBI_WRAPPER
#undef ACLSHMEM_UDMA_RELAY_GET_NBI_WRAPPER
#undef ACLSHMEM_UDMA_RELAY_PUT_NBI_WRAPPER

// Use the SHMEM-configured workspace for the default collective overload.
__aicore__ __attribute__((always_inline)) int
_mlir_ciface_aclshmemx_roce_team_sync(aclshmem_team_t team) {
  return aclshmemx_roce_team_sync(aclshmemi_get_state()->team_pools[team]);
}

// Use the SHMEM-configured workspace for the default collective overload.
__aicore__ __attribute__((always_inline)) int
_mlir_ciface_aclshmemx_roce_barrier(aclshmem_team_t team) {
  return aclshmemx_roce_barrier(aclshmemi_get_state()->team_pools[team]);
}

// Use the SHMEM-configured workspace for the default collective overload.
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmemx_roce_sync_all(void) {
  aclshmemx_roce_sync_all();
}

// Use the SHMEM-configured workspace for the default collective overload.
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmemx_roce_barrier_all(void) {
  aclshmemx_roce_barrier_all();
}

} // extern "C"

#endif // BISHENGIR_TEMPLATE_SHMEM_TRANSPORT_H
