/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RegBase/ReductionUtils.h"
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include "Vector/VecUtils.h"
#if defined(__DAV_C310__)

template <typename TIn, typename TOp>
__aiv__ __attribute__((always_inline)) VectorReg<TIn>
vmodui(VectorReg<TIn> src0, VectorReg<TIn> src1, ave_preg preg) {
  VectorReg<TIn> dst;
  vector_bool mask = convertAVEPregToVecBool(preg);
  // Ascendnpu ir never uses uXX operands, so we must cast them here.
  vmod((VectorReg<TOp> &)dst, (VectorReg<TOp> &)src0, (VectorReg<TOp> &)src1,
       mask);
  return dst;
}

template <typename TIn>
__aiv__ __attribute__((always_inline)) VectorReg<TIn>
vmodsi(VectorReg<TIn> src0, VectorReg<TIn> src1, ave_preg preg) {
  VectorReg<TIn> dst;
  vector_bool mask = convertAVEPregToVecBool(preg);
  vmod<DivisionMode::DIV_TRUNCATE>(dst, src0, src1, mask);
  return dst;
}

extern "C" {
RETURN_REGISTE_VMODUI(vmodui, int16_t, uint16_t);
RETURN_REGISTE_VMODUI(vmodui, int32_t, uint32_t);
RETURN_REGISTE_VMOD(vmod, int16_t);
RETURN_REGISTE_VMOD(vmod, int32_t);
}
#endif