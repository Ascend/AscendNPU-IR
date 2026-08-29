//===- AutoScheduleAttrDefs.h - Auto Schedule attributes --------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Defines constant attributes used in Auto Schedule.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEATTRDEFS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEATTRDEFS_H

#include "llvm/ADT/StringRef.h"

// TODO: obtain ub size from platform config
inline int64_t kUBReservedSizeInBits = 64 * 8;
inline int64_t kUBMaxSizeInBits = 192 * 1024 * 8 - kUBReservedSizeInBits;
inline int64_t kUBAlignSizeInBytes = 32;
inline int64_t kNumBitsInByte = 8;

inline char kFuncArgIdxFormat[] = "__arg{0}__";

inline llvm::StringLiteral kIntermediateProducerTagName =
    "__intermediate_producer__";

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEATTRDEFS_H
