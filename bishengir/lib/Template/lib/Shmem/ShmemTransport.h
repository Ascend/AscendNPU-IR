// MLIR C interfaces for transport-specific SHMEM operations.

#ifndef BISHENGIR_TEMPLATE_SHMEM_TRANSPORT_H
#define BISHENGIR_TEMPLATE_SHMEM_TRANSPORT_H

extern "C" {

#define ACLSHMEM_MTE_GET_NBI_WRAPPER(NAME, TYPE)                               \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_mte_get_nbi_##NAME(memref_t<__gm__ TYPE, 1> *dst, \
                                                memref_t<__gm__ TYPE, 1> *src, \
                                                uint32_t nbytes, int32_t pe) { \
    auto *state = aclshmemi_get_state();                                       \
    aclshmemx_mte_get_nbi<char>(                                               \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        reinterpret_cast<__ubuf__ char *>(state->mte_config.aclshmem_ub),      \
        state->mte_config.ub_size, nbytes, pe, state->mte_config.sync_id);     \
  }

#define ACLSHMEM_MTE_PUT_NBI_WRAPPER(NAME, TYPE)                               \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_mte_put_nbi_##NAME(memref_t<__gm__ TYPE, 1> *dst, \
                                                memref_t<__gm__ TYPE, 1> *src, \
                                                uint32_t nbytes, int32_t pe) { \
    auto *state = aclshmemi_get_state();                                       \
    aclshmemx_mte_put_nbi<char>(                                               \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        reinterpret_cast<__ubuf__ char *>(state->mte_config.aclshmem_ub),      \
        state->mte_config.ub_size, nbytes, pe, state->mte_config.sync_id);     \
  }

ACLSHMEM_MTE_GET_NBI_WRAPPER(half, half)
ACLSHMEM_MTE_GET_NBI_WRAPPER(float, float)
ACLSHMEM_MTE_GET_NBI_WRAPPER(int8, int8_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(int16, int16_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(int32, int32_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(int64, int64_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(uint8, uint8_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(uint16, uint16_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(uint32, uint32_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(uint64, uint64_t)
ACLSHMEM_MTE_GET_NBI_WRAPPER(char, char)
ACLSHMEM_MTE_GET_NBI_WRAPPER(bfloat16, bfloat16_t)

#undef ACLSHMEM_MTE_GET_NBI_WRAPPER

ACLSHMEM_MTE_PUT_NBI_WRAPPER(half, half)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(float, float)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(int8, int8_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(int16, int16_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(int32, int32_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(int64, int64_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(uint8, uint8_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(uint16, uint16_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(uint32, uint32_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(uint64, uint64_t)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(char, char)
ACLSHMEM_MTE_PUT_NBI_WRAPPER(bfloat16, bfloat16_t)

#undef ACLSHMEM_MTE_PUT_NBI_WRAPPER

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmemx_mte_quiet() {
  aclshmemx_mte_quiet();
}

#if defined(__DAV_C310__)
#define ACLSHMEM_MTE_ATOMIC_FETCH(NAME, TYPE)                                  \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_mte_atomic_fetch_##NAME(                          \
          memref_t<__gm__ TYPE, 1> *dst, int32_t pe) {                         \
    return aclshmemx_mte_atomic_fetch<TYPE>(dst->aligned + dst->offset, pe);   \
  }

ACLSHMEM_MTE_ATOMIC_FETCH(float, float)
ACLSHMEM_MTE_ATOMIC_FETCH(int32, int32_t)
ACLSHMEM_MTE_ATOMIC_FETCH(int64, int64_t)
ACLSHMEM_MTE_ATOMIC_FETCH(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_FETCH(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_FETCH

#define ACLSHMEM_MTE_ATOMIC_SET(NAME, TYPE)                                    \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_mte_atomic_set_##NAME(                            \
          memref_t<__gm__ TYPE, 1> *dst, TYPE value, int32_t pe) {             \
    aclshmemx_mte_atomic_set<TYPE>(dst->aligned + dst->offset, value, pe);     \
  }

ACLSHMEM_MTE_ATOMIC_SET(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_SET(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_SET

#define ACLSHMEM_MTE_ATOMIC_SWAP(NAME, TYPE)                                   \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_mte_atomic_swap_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, TYPE value, int32_t pe) {             \
    return aclshmemx_mte_atomic_swap<TYPE>(dst->aligned + dst->offset, value,  \
                                           pe);                                \
  }

ACLSHMEM_MTE_ATOMIC_SWAP(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_SWAP(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_SWAP

#define ACLSHMEM_MTE_ATOMIC_COMPARE_SWAP(NAME, TYPE)                           \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_mte_atomic_compare_swap_##NAME(                   \
          memref_t<__gm__ TYPE, 1> *dst, TYPE cond, TYPE value, int32_t pe) {  \
    return aclshmemx_mte_atomic_compare_swap<TYPE>(dst->aligned + dst->offset, \
                                                   cond, value, pe);           \
  }

ACLSHMEM_MTE_ATOMIC_COMPARE_SWAP(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_COMPARE_SWAP(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_COMPARE_SWAP

#define ACLSHMEM_MTE_ATOMIC_INC(NAME, TYPE)                                    \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_mte_atomic_inc_##NAME(                            \
          memref_t<__gm__ TYPE, 1> *dst, int32_t pe) {                         \
    aclshmemx_mte_atomic_inc<TYPE>(dst->aligned + dst->offset, pe);            \
  }

ACLSHMEM_MTE_ATOMIC_INC(float, float)
ACLSHMEM_MTE_ATOMIC_INC(int32, int32_t)
ACLSHMEM_MTE_ATOMIC_INC(int64, int64_t)
ACLSHMEM_MTE_ATOMIC_INC(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_INC(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_INC
#endif // defined(__DAV_C310__)

#define ACLSHMEM_MTE_ATOMIC_ADD(NAME, TYPE)                                    \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_mte_atomic_add_##NAME(                            \
          memref_t<__gm__ TYPE, 1> *dst, TYPE value, int32_t pe) {             \
    aclshmemx_mte_atomic_add<TYPE>(dst->aligned + dst->offset, value, pe);     \
  }

ACLSHMEM_MTE_ATOMIC_ADD(float, float)
ACLSHMEM_MTE_ATOMIC_ADD(int32, int32_t)
ACLSHMEM_MTE_ATOMIC_ADD(half, half)
ACLSHMEM_MTE_ATOMIC_ADD(int8, int8_t)
ACLSHMEM_MTE_ATOMIC_ADD(int16, int16_t)
ACLSHMEM_MTE_ATOMIC_ADD(bfloat16, bfloat16_t)
#if defined(__DAV_C310__)
ACLSHMEM_MTE_ATOMIC_ADD(int64, int64_t)
ACLSHMEM_MTE_ATOMIC_ADD(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_ADD(uint64, uint64_t)
#endif

#undef ACLSHMEM_MTE_ATOMIC_ADD

#if defined(__DAV_C310__)
#define ACLSHMEM_MTE_ATOMIC_FETCH_INC(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_mte_atomic_fetch_inc_##NAME(                      \
          memref_t<__gm__ TYPE, 1> *dst, int32_t pe) {                         \
    return aclshmemx_mte_atomic_fetch_inc<TYPE>(dst->aligned + dst->offset,    \
                                                pe);                           \
  }

ACLSHMEM_MTE_ATOMIC_FETCH_INC(float, float)
ACLSHMEM_MTE_ATOMIC_FETCH_INC(int32, int32_t)
ACLSHMEM_MTE_ATOMIC_FETCH_INC(int64, int64_t)
ACLSHMEM_MTE_ATOMIC_FETCH_INC(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_FETCH_INC(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_FETCH_INC

#define ACLSHMEM_MTE_ATOMIC_FETCH_ADD(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_mte_atomic_fetch_add_##NAME(                      \
          memref_t<__gm__ TYPE, 1> *dst, TYPE value, int32_t pe) {             \
    return aclshmemx_mte_atomic_fetch_add<TYPE>(dst->aligned + dst->offset,    \
                                                value, pe);                    \
  }

ACLSHMEM_MTE_ATOMIC_FETCH_ADD(float, float)
ACLSHMEM_MTE_ATOMIC_FETCH_ADD(int32, int32_t)
ACLSHMEM_MTE_ATOMIC_FETCH_ADD(int64, int64_t)
ACLSHMEM_MTE_ATOMIC_FETCH_ADD(uint32, uint32_t)
ACLSHMEM_MTE_ATOMIC_FETCH_ADD(uint64, uint64_t)

#undef ACLSHMEM_MTE_ATOMIC_FETCH_ADD
#endif // defined(__DAV_C310__)

// RoCE uses the same runtime-configured UB workspace and event ID as generic
// SHMEM RMA. Transfer counts are bytes; apply typed memref offsets before
// casting payload pointers to char for the element-counted SHMEM APIs.
#define ACLSHMEM_ROCE_GET_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_get_nbi_##NAME(                              \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe) {                                       \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ char *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_get_nbi<char>(                                              \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset), buf,      \
        nbytes, pe, sync_id);                                                  \
  }

#define ACLSHMEM_ROCE_PUT_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_put_nbi_##NAME(                              \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe) {                                       \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ char *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_put_nbi<char>(                                              \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset), buf,      \
        nbytes, pe, sync_id);                                                  \
  }

#define ACLSHMEM_ROCE_QUIET_WRAPPER(NAME, TYPE)                                \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_quiet_##NAME(uint32_t pe) {                  \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ TYPE *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_quiet<TYPE>(pe, buf, sync_id);                              \
  }

// Match the existing PIPE_S UDMA put ABI: no UB workspace or event is needed.
#define ACLSHMEM_UDMA_GET_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_get_nbi_##NAME(                              \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe) {                                       \
    aclshmemx_udma_get_nbi<char, PIPE_S>(                                      \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        (__ubuf__ char *)nullptr, nbytes, pe, 0);                              \
  }

#define ACLSHMEM_UDMA_PUT_NBI_WRAPPER(NAME, TYPE)                              \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_put_nbi_##NAME(                              \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe) {                                       \
    aclshmemx_udma_put_nbi<char, PIPE_S>(                                      \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        (__ubuf__ char *)nullptr, nbytes, pe, 0);                              \
  }

// Relay wrappers are opt-in via BISHENGIR_SHMEM_RELAY_SUPPORT for C310 and
// require a matching relay-enabled runtime. C220 never enables these wrappers.
#ifdef ACLSHMEM_RELAY_SUPPORT
#define ACLSHMEM_UDMA_RELAY_GET_NBI_WRAPPER(NAME, TYPE)                        \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_relay_get_nbi_##NAME(                        \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe, int32_t relay_pe) {                     \
    aclshmemx_udma_relay_get_nbi<char, PIPE_S>(                                \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        (__ubuf__ char *)nullptr, nbytes, pe, relay_pe, 0);                    \
  }

#define ACLSHMEM_UDMA_RELAY_PUT_NBI_WRAPPER(NAME, TYPE)                        \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_relay_put_nbi_##NAME(                        \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe, int32_t relay_pe) {                     \
    aclshmemx_udma_relay_put_nbi<char, PIPE_S>(                                \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        (__ubuf__ char *)nullptr, nbytes, pe, relay_pe, 0);                    \
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

// Byte-counted UDMA put-with-signal.
#define ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_putmem_signal_nbi_##NAME(                    \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, memref_t<__gm__ uint64_t, 1> *sig_addr,             \
          uint64_t signal, int32_t pe) {                                       \
    aclshmemx_udma_put_signal_nbi<char, PIPE_S>(                               \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset), nbytes,   \
        sig_addr->aligned + sig_addr->offset, signal, pe,                      \
        (__ubuf__ uint8_t *)nullptr, 0);                                       \
  }

ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(half, half)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(float, float)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(int8, int8_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(int16, int16_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(int32, int32_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(int64, int64_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(uint8, uint8_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(uint16, uint16_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(uint32, uint32_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(uint64, uint64_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(bfloat16, bfloat16_t)
ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER(char, char)

#undef ACLSHMEM_UDMA_PUTMEM_SIGNAL_NBI_WRAPPER

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmemx_udma_quiet(int32_t pe) {
  aclshmemx_udma_quiet(pe);
}

// Transport RMA wrappers accept byte counts. Apply typed memref offsets before
// casting payload pointers to char for the element-counted SHMEM APIs.
// QP-specific wrappers preserve the upstream transport restrictions.
// In a relay-enabled build, UDMA QP APIs abort.

#define ACLSHMEMX_ROCE_QP_GET_NBI_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_qp_get_nbi_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe, uint32_t qp_idx) {                      \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ char *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_qp_get_nbi<char>(                                           \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset), buf,      \
        nbytes, pe, qp_idx, sync_id);                                          \
  }

#define ACLSHMEMX_ROCE_QP_PUT_NBI_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_qp_put_nbi_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe, uint32_t qp_idx) {                      \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ char *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_qp_put_nbi<char>(                                           \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset), buf,      \
        nbytes, pe, qp_idx, sync_id);                                          \
  }

#define ACLSHMEMX_UDMA_QP_GET_NBI_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_qp_get_nbi_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe, uint32_t qp_idx) {                      \
    aclshmemx_udma_qp_get_nbi<char, PIPE_S>(                                   \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        (__ubuf__ char *)nullptr, nbytes, pe, qp_idx, 0);                      \
  }

#define ACLSHMEMX_UDMA_QP_PUT_NBI_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_qp_put_nbi_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, int32_t pe, uint32_t qp_idx) {                      \
    aclshmemx_udma_qp_put_nbi<char, PIPE_S>(                                   \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset),           \
        (__ubuf__ char *)nullptr, nbytes, pe, qp_idx, 0);                      \
  }

#define ACLSHMEMX_ROCE_QP_QUIET_WRAPPER(NAME, TYPE)                            \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_qp_quiet_##NAME(uint32_t pe,                 \
                                                  uint32_t qp_idx) {           \
    auto *state = aclshmemi_get_state();                                       \
    auto *buf =                                                                \
        reinterpret_cast<__ubuf__ TYPE *>(state->rdma_config.aclshmem_ub);     \
    uint32_t sync_id = state->rdma_config.sync_id;                             \
    aclshmemx_roce_qp_quiet<TYPE>(pe, qp_idx, buf, sync_id);                   \
  }

#define ACLSHMEMX_UDMA_QP_PUT_SIGNAL_NBI_WRAPPER(NAME, TYPE)                   \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_qp_put_signal_nbi_##NAME(                    \
          memref_t<__gm__ TYPE, 1> *dst, memref_t<__gm__ TYPE, 1> *src,        \
          uint32_t nbytes, memref_t<__gm__ uint64_t, 1> *sig_addr,             \
          uint64_t signal, int32_t pe, uint32_t qp_idx) {                      \
    aclshmemx_udma_qp_put_signal_nbi<char, PIPE_S>(                            \
        reinterpret_cast<__gm__ char *>(dst->aligned + dst->offset),           \
        reinterpret_cast<__gm__ char *>(src->aligned + src->offset), nbytes,   \
        sig_addr->aligned + sig_addr->offset, signal, pe, qp_idx,              \
        (__ubuf__ uint8_t *)nullptr, 0);                                       \
  }

// The default IN_DIE RoCE backend rejects QP API instantiation.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) ||                                \
    defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
#define ACLSHMEMX_ROCE_QP_WRAPPERS(NAME, TYPE)                                 \
  ACLSHMEMX_ROCE_QP_GET_NBI_WRAPPER(NAME, TYPE)                                \
  ACLSHMEMX_ROCE_QP_PUT_NBI_WRAPPER(NAME, TYPE)                                \
  ACLSHMEMX_ROCE_QP_QUIET_WRAPPER(NAME, TYPE)
#else
#define ACLSHMEMX_ROCE_QP_WRAPPERS(NAME, TYPE)
#endif

#define ACLSHMEMX_QP_WRAPPERS(NAME, TYPE)                                      \
  ACLSHMEMX_ROCE_QP_WRAPPERS(NAME, TYPE)                                       \
  ACLSHMEMX_UDMA_QP_GET_NBI_WRAPPER(NAME, TYPE)                                \
  ACLSHMEMX_UDMA_QP_PUT_NBI_WRAPPER(NAME, TYPE)                                \
  ACLSHMEMX_UDMA_QP_PUT_SIGNAL_NBI_WRAPPER(NAME, TYPE)

ACLSHMEMX_QP_WRAPPERS(half, half)
ACLSHMEMX_QP_WRAPPERS(float, float)
ACLSHMEMX_QP_WRAPPERS(int8, int8_t)
ACLSHMEMX_QP_WRAPPERS(int16, int16_t)
ACLSHMEMX_QP_WRAPPERS(int32, int32_t)
ACLSHMEMX_QP_WRAPPERS(int64, int64_t)
ACLSHMEMX_QP_WRAPPERS(uint8, uint8_t)
ACLSHMEMX_QP_WRAPPERS(uint16, uint16_t)
ACLSHMEMX_QP_WRAPPERS(uint32, uint32_t)
ACLSHMEMX_QP_WRAPPERS(uint64, uint64_t)
ACLSHMEMX_QP_WRAPPERS(char, char)
ACLSHMEMX_QP_WRAPPERS(bfloat16, bfloat16_t)

#undef ACLSHMEMX_QP_WRAPPERS
#undef ACLSHMEMX_ROCE_QP_WRAPPERS
#undef ACLSHMEMX_ROCE_QP_GET_NBI_WRAPPER
#undef ACLSHMEMX_ROCE_QP_PUT_NBI_WRAPPER
#undef ACLSHMEMX_UDMA_QP_GET_NBI_WRAPPER
#undef ACLSHMEMX_UDMA_QP_PUT_NBI_WRAPPER
#undef ACLSHMEMX_ROCE_QP_QUIET_WRAPPER
#undef ACLSHMEMX_UDMA_QP_PUT_SIGNAL_NBI_WRAPPER

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmemx_udma_qp_quiet(int32_t pe, uint32_t qp_idx) {
  aclshmemx_udma_qp_quiet(pe, qp_idx);
}

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

// The transport-specific atomic APIs require Ascend950, like the high-level
// generic atomics. Do not instantiate them for Ascend910B templates.
#if defined(__DAV_C310__)

// ============================================================================
// RoCE Low-level API Wrappers
// ============================================================================

// Wrapper for aclshmemx_roce_atomic_fetch
#define ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER(NAME, TYPE)                        \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_fetch_##NAME(                         \
          memref_t<__gm__ TYPE, 1> *src, int pe) {                             \
    return aclshmemx_roce_atomic_fetch<TYPE>(src->aligned + src->offset, pe);  \
  }

ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER(uint64, uint64_t)
// HNS_1825 atomics accept only 32-bit and 64-bit integers.
#ifndef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER(float, float)
#endif

#undef ACLSHMEMX_ROCE_ATOMIC_FETCH_WRAPPER

// Wrapper for aclshmemx_roce_atomic_fetch_inc
#define ACLSHMEMX_ROCE_ATOMIC_FETCH_INC_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_fetch_inc_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, int pe) {                             \
    return aclshmemx_roce_atomic_fetch_inc<TYPE>(dst->aligned + dst->offset,   \
                                                 pe);                          \
  }

ACLSHMEMX_ROCE_ATOMIC_FETCH_INC_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_INC_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_INC_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_INC_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_FETCH_INC_WRAPPER

// Wrapper for aclshmemx_roce_atomic_inc
#define ACLSHMEMX_ROCE_ATOMIC_INC_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_atomic_inc_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, int pe) {                             \
    aclshmemx_roce_atomic_inc<TYPE>(dst->aligned + dst->offset, pe);           \
  }

ACLSHMEMX_ROCE_ATOMIC_INC_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_INC_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_INC_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_INC_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_INC_WRAPPER

// Wrapper for aclshmemx_roce_atomic_set
#define ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_atomic_set_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_roce_atomic_set<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER(uint64, uint64_t)
// HNS_1825 atomics accept only 32-bit and 64-bit integers.
#ifndef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER(float, float)
#endif

#undef ACLSHMEMX_ROCE_ATOMIC_SET_WRAPPER

// Wrapper for aclshmemx_roce_atomic_swap
#define ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER(NAME, TYPE)                         \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_swap_##NAME(                          \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_roce_atomic_swap<TYPE>(dst->aligned + dst->offset, value, \
                                            pe);                               \
  }

ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER(uint64, uint64_t)
// HNS_1825 atomics accept only 32-bit and 64-bit integers.
#ifndef ACLSHMEMI_RDMA_K_BACKEND_HNS_1825
ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER(float, float)
#endif

#undef ACLSHMEMX_ROCE_ATOMIC_SWAP_WRAPPER

// Wrapper for aclshmemx_roce_atomic_add
#define ACLSHMEMX_ROCE_ATOMIC_ADD_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_atomic_add_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_roce_atomic_add<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_ROCE_ATOMIC_ADD_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_ADD_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_ADD_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_ADD_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_ADD_WRAPPER

// Wrapper for aclshmemx_roce_atomic_fetch_add
#define ACLSHMEMX_ROCE_ATOMIC_FETCH_ADD_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_fetch_add_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_roce_atomic_fetch_add<TYPE>(dst->aligned + dst->offset,   \
                                                 value, pe);                   \
  }

ACLSHMEMX_ROCE_ATOMIC_FETCH_ADD_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_ADD_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_ADD_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_ADD_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_FETCH_ADD_WRAPPER

// Wrapper for aclshmemx_roce_atomic_and
#define ACLSHMEMX_ROCE_ATOMIC_AND_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_atomic_and_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_roce_atomic_and<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_ROCE_ATOMIC_AND_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_AND_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_AND_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_AND_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_AND_WRAPPER

// Wrapper for aclshmemx_roce_atomic_or
#define ACLSHMEMX_ROCE_ATOMIC_OR_WRAPPER(NAME, TYPE)                           \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_atomic_or_##NAME(                            \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_roce_atomic_or<TYPE>(dst->aligned + dst->offset, value, pe);     \
  }

ACLSHMEMX_ROCE_ATOMIC_OR_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_OR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_OR_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_OR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_OR_WRAPPER

// Wrapper for aclshmemx_roce_atomic_xor
#define ACLSHMEMX_ROCE_ATOMIC_XOR_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_roce_atomic_xor_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_roce_atomic_xor<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_ROCE_ATOMIC_XOR_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_XOR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_XOR_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_XOR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_XOR_WRAPPER

// Wrapper for aclshmemx_roce_atomic_fetch_and
#define ACLSHMEMX_ROCE_ATOMIC_FETCH_AND_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_fetch_and_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_roce_atomic_fetch_and<TYPE>(dst->aligned + dst->offset,   \
                                                 value, pe);                   \
  }

ACLSHMEMX_ROCE_ATOMIC_FETCH_AND_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_AND_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_AND_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_AND_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_FETCH_AND_WRAPPER

// Wrapper for aclshmemx_roce_atomic_fetch_or
#define ACLSHMEMX_ROCE_ATOMIC_FETCH_OR_WRAPPER(NAME, TYPE)                     \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_fetch_or_##NAME(                      \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_roce_atomic_fetch_or<TYPE>(dst->aligned + dst->offset,    \
                                                value, pe);                    \
  }

ACLSHMEMX_ROCE_ATOMIC_FETCH_OR_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_OR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_OR_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_OR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_FETCH_OR_WRAPPER

// Wrapper for aclshmemx_roce_atomic_fetch_xor
#define ACLSHMEMX_ROCE_ATOMIC_FETCH_XOR_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_fetch_xor_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_roce_atomic_fetch_xor<TYPE>(dst->aligned + dst->offset,   \
                                                 value, pe);                   \
  }

ACLSHMEMX_ROCE_ATOMIC_FETCH_XOR_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_XOR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_XOR_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_FETCH_XOR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_FETCH_XOR_WRAPPER

// Wrapper for aclshmemx_roce_atomic_compare_swap
#define ACLSHMEMX_ROCE_ATOMIC_COMPARE_SWAP_WRAPPER(NAME, TYPE)                 \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_roce_atomic_compare_swap_##NAME(                  \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE cond, const TYPE value,    \
          int pe) {                                                            \
    return aclshmemx_roce_atomic_compare_swap<TYPE>(                           \
        dst->aligned + dst->offset, cond, value, pe);                          \
  }

ACLSHMEMX_ROCE_ATOMIC_COMPARE_SWAP_WRAPPER(int32, int32_t)
ACLSHMEMX_ROCE_ATOMIC_COMPARE_SWAP_WRAPPER(uint32, uint32_t)
ACLSHMEMX_ROCE_ATOMIC_COMPARE_SWAP_WRAPPER(int64, int64_t)
ACLSHMEMX_ROCE_ATOMIC_COMPARE_SWAP_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_ROCE_ATOMIC_COMPARE_SWAP_WRAPPER

// ============================================================================
// UDMA Low-level API Wrappers
// ============================================================================

// Wrapper for aclshmemx_udma_atomic_fetch
#define ACLSHMEMX_UDMA_ATOMIC_FETCH_WRAPPER(NAME, TYPE)                        \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_fetch_##NAME(                         \
          memref_t<__gm__ TYPE, 1> *src, int pe) {                             \
    return aclshmemx_udma_atomic_fetch<TYPE>(src->aligned + src->offset, pe);  \
  }

ACLSHMEMX_UDMA_ATOMIC_FETCH_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_FETCH_WRAPPER

// Wrapper for aclshmemx_udma_atomic_fetch_inc
#define ACLSHMEMX_UDMA_ATOMIC_FETCH_INC_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_fetch_inc_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, int pe) {                             \
    return aclshmemx_udma_atomic_fetch_inc<TYPE>(dst->aligned + dst->offset,   \
                                                 pe);                          \
  }

ACLSHMEMX_UDMA_ATOMIC_FETCH_INC_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_INC_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_INC_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_INC_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_FETCH_INC_WRAPPER

// Wrapper for aclshmemx_udma_atomic_inc
#define ACLSHMEMX_UDMA_ATOMIC_INC_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_atomic_inc_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, int pe) {                             \
    aclshmemx_udma_atomic_inc<TYPE>(dst->aligned + dst->offset, pe);           \
  }

ACLSHMEMX_UDMA_ATOMIC_INC_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_INC_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_INC_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_INC_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_INC_WRAPPER

// Wrapper for aclshmemx_udma_atomic_add
#define ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_atomic_add_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_udma_atomic_add<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER(uint64, uint64_t)
ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER(float, float)

#undef ACLSHMEMX_UDMA_ATOMIC_ADD_WRAPPER

// Wrapper for aclshmemx_udma_atomic_fetch_add
#define ACLSHMEMX_UDMA_ATOMIC_FETCH_ADD_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_fetch_add_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_udma_atomic_fetch_add<TYPE>(dst->aligned + dst->offset,   \
                                                 value, pe);                   \
  }

ACLSHMEMX_UDMA_ATOMIC_FETCH_ADD_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_ADD_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_ADD_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_ADD_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_FETCH_ADD_WRAPPER

// Wrapper for aclshmemx_udma_atomic_set
#define ACLSHMEMX_UDMA_ATOMIC_SET_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_atomic_set_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_udma_atomic_set<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_UDMA_ATOMIC_SET_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_SET_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_SET_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_SET_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_SET_WRAPPER

// Wrapper for aclshmemx_udma_atomic_swap
#define ACLSHMEMX_UDMA_ATOMIC_SWAP_WRAPPER(NAME, TYPE)                         \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_swap_##NAME(                          \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_udma_atomic_swap<TYPE>(dst->aligned + dst->offset, value, \
                                            pe);                               \
  }

ACLSHMEMX_UDMA_ATOMIC_SWAP_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_SWAP_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_SWAP_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_SWAP_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_SWAP_WRAPPER

// Wrapper for aclshmemx_udma_atomic_fetch_and
#define ACLSHMEMX_UDMA_ATOMIC_FETCH_AND_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_fetch_and_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_udma_atomic_fetch_and<TYPE>(dst->aligned + dst->offset,   \
                                                 value, pe);                   \
  }

ACLSHMEMX_UDMA_ATOMIC_FETCH_AND_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_AND_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_AND_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_AND_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_FETCH_AND_WRAPPER

// Wrapper for aclshmemx_udma_atomic_and
#define ACLSHMEMX_UDMA_ATOMIC_AND_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_atomic_and_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_udma_atomic_and<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_UDMA_ATOMIC_AND_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_AND_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_AND_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_AND_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_AND_WRAPPER

// Wrapper for aclshmemx_udma_atomic_fetch_or
#define ACLSHMEMX_UDMA_ATOMIC_FETCH_OR_WRAPPER(NAME, TYPE)                     \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_fetch_or_##NAME(                      \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_udma_atomic_fetch_or<TYPE>(dst->aligned + dst->offset,    \
                                                value, pe);                    \
  }

ACLSHMEMX_UDMA_ATOMIC_FETCH_OR_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_OR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_OR_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_OR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_FETCH_OR_WRAPPER

// Wrapper for aclshmemx_udma_atomic_or
#define ACLSHMEMX_UDMA_ATOMIC_OR_WRAPPER(NAME, TYPE)                           \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_atomic_or_##NAME(                            \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_udma_atomic_or<TYPE>(dst->aligned + dst->offset, value, pe);     \
  }

ACLSHMEMX_UDMA_ATOMIC_OR_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_OR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_OR_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_OR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_OR_WRAPPER

// Wrapper for aclshmemx_udma_atomic_fetch_xor
#define ACLSHMEMX_UDMA_ATOMIC_FETCH_XOR_WRAPPER(NAME, TYPE)                    \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_fetch_xor_##NAME(                     \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    return aclshmemx_udma_atomic_fetch_xor<TYPE>(dst->aligned + dst->offset,   \
                                                 value, pe);                   \
  }

ACLSHMEMX_UDMA_ATOMIC_FETCH_XOR_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_XOR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_XOR_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_FETCH_XOR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_FETCH_XOR_WRAPPER

// Wrapper for aclshmemx_udma_atomic_xor
#define ACLSHMEMX_UDMA_ATOMIC_XOR_WRAPPER(NAME, TYPE)                          \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_aclshmemx_udma_atomic_xor_##NAME(                           \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE value, int pe) {           \
    aclshmemx_udma_atomic_xor<TYPE>(dst->aligned + dst->offset, value, pe);    \
  }

ACLSHMEMX_UDMA_ATOMIC_XOR_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_XOR_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_XOR_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_XOR_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_XOR_WRAPPER

// Wrapper for aclshmemx_udma_atomic_compare_swap
#define ACLSHMEMX_UDMA_ATOMIC_COMPARE_SWAP_WRAPPER(NAME, TYPE)                 \
  __aicore__ __attribute__((always_inline)) TYPE                               \
      _mlir_ciface_aclshmemx_udma_atomic_compare_swap_##NAME(                  \
          memref_t<__gm__ TYPE, 1> *dst, const TYPE cond, const TYPE value,    \
          int pe) {                                                            \
    return aclshmemx_udma_atomic_compare_swap<TYPE>(                           \
        dst->aligned + dst->offset, cond, value, pe);                          \
  }

ACLSHMEMX_UDMA_ATOMIC_COMPARE_SWAP_WRAPPER(int32, int32_t)
ACLSHMEMX_UDMA_ATOMIC_COMPARE_SWAP_WRAPPER(uint32, uint32_t)
ACLSHMEMX_UDMA_ATOMIC_COMPARE_SWAP_WRAPPER(int64, int64_t)
ACLSHMEMX_UDMA_ATOMIC_COMPARE_SWAP_WRAPPER(uint64, uint64_t)

#undef ACLSHMEMX_UDMA_ATOMIC_COMPARE_SWAP_WRAPPER

#endif // defined(__DAV_C310__)

} // extern "C"

#endif // BISHENGIR_TEMPLATE_SHMEM_TRANSPORT_H
