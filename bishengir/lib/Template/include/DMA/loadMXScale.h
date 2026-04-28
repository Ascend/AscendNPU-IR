
#ifndef HIVM_MLIR_TEMPLATE_LOAD_MX_SCALE_UTILS_H
#define HIVM_MLIR_TEMPLATE_LOAD_MX_SCALE_UTILS_H
#include "Utils.h"

#define DECLARE_LOAD_MX_SCALE(src_scope, dst_scope, src_dim, dst_dim, type)    \
  __aicore__ __attribute__((always_inline)) void                               \
  _mlir_ciface_load_scale_##src_scope##_to_##dst_scope##_##src_dim##d_##type(  \
      memref_t<__##src_scope##__ type, src_dim> *src,                          \
      memref_t<__##dst_scope##__ type, dst_dim> *dst)

#define REGISTE_LOAD_MX_SCALE(src_scope, dst_scope, src_dim, dst_dim, type)    \
  DECLARE_LOAD_MX_SCALE(src_scope, dst_scope, src_dim, dst_dim, type) {        \
    copy_##src_scope##_to_##dst_scope##_load_mx_scale_core<type>(src, dst);    \
  }

#if defined(__DAV_C310__)
extern "C" {
DECLARE_LOAD_MX_SCALE(gm, cbuf, 2, 4, int8_t);
}
#endif

#endif // HIVM_MLIR_TEMPLATE_LOAD_MX_SCALE_UTILS_H
