/**
 * @file Shmem.cpp
 * @brief MLIR C interface wrapper functions for aclshmem APIs
 */

#include <cstdint>
#include "device/gm2gm/shmem_device_amo.h"
#include "device/gm2gm/shmem_device_cc.h"
#include "device/gm2gm/shmem_device_mo.h"
#include "device/gm2gm/shmem_device_p2p_sync.h"
#include "device/gm2gm/shmem_device_rma.h"
#include "device/gm2gm/shmem_device_so.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "device/gm2gm/engine/shmem_device_sdma.h"
#include "device/ub2gm/shmem_device_rma.h"
#include "device/ub2gm/engine/shmem_device_mte.h"
#include "device/shmem_def.h"
#include "device/team/shmem_device_team.h"
#include "Utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper for aclshmem_my_pe
__aicore__ __attribute__((always_inline)) int
_mlir_ciface_aclshmem_my_pe(void) {
  return aclshmem_my_pe();
}

// Wrapper for aclshmem_n_pes
__aicore__ __attribute__((always_inline)) int
_mlir_ciface_aclshmem_n_pes(void) {
  return aclshmem_n_pes();
}

// Wrapper for aclshmem_barrier_all
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmem_barrier_all(void) {
  aclshmem_barrier_all();
}

// Wrapper for aclshmemx_barrier_all_vec
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_aclshmemx_barrier_all_vec(void) {
  aclshmemx_barrier_all_vec();
}

// TODO: Consider enhance memref_as_ptr attribute and adapt for HIVMToStandard &
// ConvertMemrefToBarePtr pass, so that template function can directly use
// pointer, no need to wrap with memref struct. Macro to generate wrappers for
// aclshmem_ptr functions
#define ACLSHMEM_PTR_WRAPPER(NAME, TYPE)                                       \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmem_ptr_##NAME(memref_t<__gm__ TYPE, 1> *symmPtr,          \
                                   memref_t<__gm__ TYPE, 1> *ptr, int pe) {    \
    __gm__ TYPE *symmAlignedPtr =                                              \
        (__gm__ TYPE *)aclshmem_ptr(ptr->aligned + ptr->offset, pe);           \
    symmPtr->aligned = symmAlignedPtr;                                         \
    symmPtr->allocated = symmAlignedPtr;                                       \
    symmPtr->offset = 0;                                                       \
    symmPtr->sizes[0] = ptr->sizes[0];                                         \
    symmPtr->strides[0] = ptr->strides[0];                                     \
  }

// Generate wrappers for all data types
ACLSHMEM_PTR_WRAPPER(half, half)
ACLSHMEM_PTR_WRAPPER(float, float)
ACLSHMEM_PTR_WRAPPER(int8, int8_t)
ACLSHMEM_PTR_WRAPPER(int16, int16_t)
ACLSHMEM_PTR_WRAPPER(int32, int32_t)
ACLSHMEM_PTR_WRAPPER(int64, int64_t)
ACLSHMEM_PTR_WRAPPER(uint8, uint8_t)
ACLSHMEM_PTR_WRAPPER(uint16, uint16_t)
ACLSHMEM_PTR_WRAPPER(uint32, uint32_t)
ACLSHMEM_PTR_WRAPPER(uint64, uint64_t)
ACLSHMEM_PTR_WRAPPER(char, char)
ACLSHMEM_PTR_WRAPPER(bfloat16, bfloat16_t)

#undef ACLSHMEM_PTR_WRAPPER

// Macro to generate wrappers for aclshmem_##NAME##_p functions
#define ACLSHMEM_P_WRAPPER(NAME, TYPE)                                         \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmem_##NAME##_p(memref_t<__gm__ TYPE, 1> *dst,              \
                                   const TYPE value, int pe) {                 \
    aclshmem_##NAME##_p(dst->aligned + dst->offset, value, pe);                \
  }

// Generate wrappers for all data types
ACLSHMEM_P_WRAPPER(half, half)
ACLSHMEM_P_WRAPPER(float, float)
ACLSHMEM_P_WRAPPER(int8, int8_t)
ACLSHMEM_P_WRAPPER(int16, int16_t)
ACLSHMEM_P_WRAPPER(int32, int32_t)
ACLSHMEM_P_WRAPPER(int64, int64_t)
ACLSHMEM_P_WRAPPER(uint8, uint8_t)
ACLSHMEM_P_WRAPPER(uint16, uint16_t)
ACLSHMEM_P_WRAPPER(uint32, uint32_t)
ACLSHMEM_P_WRAPPER(uint64, uint64_t)
ACLSHMEM_P_WRAPPER(char, char)
ACLSHMEM_P_WRAPPER(bfloat16, bfloat16_t)

#undef ACLSHMEM_P_WRAPPER

// Macro to generate wrappers for aclshmem_##NAME##_wait_until functions
#define ACLSHMEM_WAIT_UNTIL_WRAPPER(NAME, TYPE)                                \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmem_wait_until_##NAME(memref_t<__gm__ TYPE, 1> ivar,       \
                                          int cmp, TYPE cmp_value) {           \
    aclshmem_##NAME##_wait_until(ivar.aligned + ivar.offset, cmp, cmp_value);  \
  }

// Generate wrappers for all data types
ACLSHMEM_WAIT_UNTIL_WRAPPER(float, float)
ACLSHMEM_WAIT_UNTIL_WRAPPER(int8, int8_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(int16, int16_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(int32, int32_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(int64, int64_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(uint8, uint8_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(uint16, uint16_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(uint32, uint32_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(uint64, uint64_t)
ACLSHMEM_WAIT_UNTIL_WRAPPER(char, char)

#undef ACLSHMEM_WAIT_UNTIL_WRAPPER

// Macro to generate wrappers for aclshmem_wait functions (with numBarriers
// loop)
#define ACLSHMEM_WAIT_WRAPPER(NAME, TYPE)                                      \
  __aicore__ __attribute__((always_inline)) int                                \
  _mlir_ciface_aclshmem_wait_##NAME(memref_t<__gm__ TYPE, 1> barrierPtr,       \
                                    int64_t numBarriers, TYPE waitVal) {       \
    for (int64_t i = 0; i < numBarriers; ++i) {                                \
      __gm__ TYPE *ptr =                                                       \
          barrierPtr.aligned + barrierPtr.offset + i * (64 / sizeof(TYPE));    \
      aclshmem_##NAME##_wait_until(ptr, 0, waitVal);                           \
    }                                                                          \
    return 0;                                                                  \
  }

ACLSHMEM_WAIT_WRAPPER(int8, int8_t)
ACLSHMEM_WAIT_WRAPPER(int16, int16_t)
ACLSHMEM_WAIT_WRAPPER(int32, int32_t)
ACLSHMEM_WAIT_WRAPPER(int64, int64_t)

#undef ACLSHMEM_WAIT_WRAPPER

// Macro to generate wrappers for scalar consume_token functions
#define CONSUME_TOKEN_SCALAR_WRAPPER(NAME, TYPE)                               \
  __aicore__ __attribute__((always_inline)) TYPE                               \
  _mlir_ciface_aclshmem_consume_token_##NAME(TYPE value, int token) {          \
    return value;                                                              \
  }

CONSUME_TOKEN_SCALAR_WRAPPER(int8, int8_t)
CONSUME_TOKEN_SCALAR_WRAPPER(int16, int16_t)
CONSUME_TOKEN_SCALAR_WRAPPER(int32, int32_t)
CONSUME_TOKEN_SCALAR_WRAPPER(int64, int64_t)
CONSUME_TOKEN_SCALAR_WRAPPER(half, half)
CONSUME_TOKEN_SCALAR_WRAPPER(float, float)
CONSUME_TOKEN_SCALAR_WRAPPER(char, char)
CONSUME_TOKEN_SCALAR_WRAPPER(bfloat16, bfloat16_t)

#undef CONSUME_TOKEN_SCALAR_WRAPPER

// Macro to generate wrappers for memref consume_token functions
#define CONSUME_TOKEN_PTR_WRAPPER(NAME, DIM, ...)                              \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_aclshmem_consume_token_##NAME##_##DIM##d(                       \
      __VA_ARGS__ *res, __VA_ARGS__ *value, int token) {                       \
    res->aligned = value->aligned;                                             \
    res->allocated = value->allocated;                                         \
    res->offset = value->offset;                                               \
    for (int i = 0; i < DIM; i++) {                                            \
      res->sizes[i] = value->sizes[i];                                         \
      res->strides[i] = value->strides[i];                                     \
    }                                                                          \
  }

CONSUME_TOKEN_PTR_WRAPPER(int8_ptr, 1, memref_t<__gm__ int8_t, 1>)
CONSUME_TOKEN_PTR_WRAPPER(int8_ptr, 2, memref_t<__gm__ int8_t, 2>)
CONSUME_TOKEN_PTR_WRAPPER(int8_ptr, 3, memref_t<__gm__ int8_t, 3>)
CONSUME_TOKEN_PTR_WRAPPER(int8_ptr, 4, memref_t<__gm__ int8_t, 4>)
CONSUME_TOKEN_PTR_WRAPPER(int8_ptr, 5, memref_t<__gm__ int8_t, 5>)
CONSUME_TOKEN_PTR_WRAPPER(int16_ptr, 1, memref_t<__gm__ int16_t, 1>)
CONSUME_TOKEN_PTR_WRAPPER(int16_ptr, 2, memref_t<__gm__ int16_t, 2>)
CONSUME_TOKEN_PTR_WRAPPER(int16_ptr, 3, memref_t<__gm__ int16_t, 3>)
CONSUME_TOKEN_PTR_WRAPPER(int16_ptr, 4, memref_t<__gm__ int16_t, 4>)
CONSUME_TOKEN_PTR_WRAPPER(int16_ptr, 5, memref_t<__gm__ int16_t, 5>)
CONSUME_TOKEN_PTR_WRAPPER(int32_ptr, 1, memref_t<__gm__ int32_t, 1>)
CONSUME_TOKEN_PTR_WRAPPER(int32_ptr, 2, memref_t<__gm__ int32_t, 2>)
CONSUME_TOKEN_PTR_WRAPPER(int32_ptr, 3, memref_t<__gm__ int32_t, 3>)
CONSUME_TOKEN_PTR_WRAPPER(int32_ptr, 4, memref_t<__gm__ int32_t, 4>)
CONSUME_TOKEN_PTR_WRAPPER(int32_ptr, 5, memref_t<__gm__ int32_t, 5>)
CONSUME_TOKEN_PTR_WRAPPER(int64_ptr, 1, memref_t<__gm__ int64_t, 1>)
CONSUME_TOKEN_PTR_WRAPPER(int64_ptr, 2, memref_t<__gm__ int64_t, 2>)
CONSUME_TOKEN_PTR_WRAPPER(int64_ptr, 3, memref_t<__gm__ int64_t, 3>)
CONSUME_TOKEN_PTR_WRAPPER(int64_ptr, 4, memref_t<__gm__ int64_t, 4>)
CONSUME_TOKEN_PTR_WRAPPER(int64_ptr, 5, memref_t<__gm__ int64_t, 5>)
CONSUME_TOKEN_PTR_WRAPPER(half_ptr, 1, memref_t<__gm__ half, 1>)
CONSUME_TOKEN_PTR_WRAPPER(half_ptr, 2, memref_t<__gm__ half, 2>)
CONSUME_TOKEN_PTR_WRAPPER(half_ptr, 3, memref_t<__gm__ half, 3>)
CONSUME_TOKEN_PTR_WRAPPER(half_ptr, 4, memref_t<__gm__ half, 4>)
CONSUME_TOKEN_PTR_WRAPPER(half_ptr, 5, memref_t<__gm__ half, 5>)
CONSUME_TOKEN_PTR_WRAPPER(float_ptr, 1, memref_t<__gm__ float, 1>)
CONSUME_TOKEN_PTR_WRAPPER(float_ptr, 2, memref_t<__gm__ float, 2>)
CONSUME_TOKEN_PTR_WRAPPER(float_ptr, 3, memref_t<__gm__ float, 3>)
CONSUME_TOKEN_PTR_WRAPPER(float_ptr, 4, memref_t<__gm__ float, 4>)
CONSUME_TOKEN_PTR_WRAPPER(float_ptr, 5, memref_t<__gm__ float, 5>)
CONSUME_TOKEN_PTR_WRAPPER(char_ptr, 1, memref_t<__gm__ char, 1>)
CONSUME_TOKEN_PTR_WRAPPER(char_ptr, 2, memref_t<__gm__ char, 2>)
CONSUME_TOKEN_PTR_WRAPPER(char_ptr, 3, memref_t<__gm__ char, 3>)
CONSUME_TOKEN_PTR_WRAPPER(char_ptr, 4, memref_t<__gm__ char, 4>)
CONSUME_TOKEN_PTR_WRAPPER(char_ptr, 5, memref_t<__gm__ char, 5>)
CONSUME_TOKEN_PTR_WRAPPER(bfloat16_ptr, 1, memref_t<__gm__ bfloat16_t, 1>)
CONSUME_TOKEN_PTR_WRAPPER(bfloat16_ptr, 2, memref_t<__gm__ bfloat16_t, 2>)
CONSUME_TOKEN_PTR_WRAPPER(bfloat16_ptr, 3, memref_t<__gm__ bfloat16_t, 3>)
CONSUME_TOKEN_PTR_WRAPPER(bfloat16_ptr, 4, memref_t<__gm__ bfloat16_t, 4>)
CONSUME_TOKEN_PTR_WRAPPER(bfloat16_ptr, 5, memref_t<__gm__ bfloat16_t, 5>)

#undef CONSUME_TOKEN_PTR_WRAPPER

#ifdef __cplusplus
}
#endif