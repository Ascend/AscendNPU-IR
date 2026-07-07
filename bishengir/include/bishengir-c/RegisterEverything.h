//===-------- RegisterEverything.h - Registration functions -------*- C -*-===//
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

#ifndef BISHENGIR_C_REGISTEREVERYTHING_H
#define BISHENGIR_C_REGISTEREVERYTHING_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers all dialects with a context.
/// This is needed before creating IR for these Dialects.
MLIR_CAPI_EXPORTED void bishengirRegisterAllDialects(MlirContext context);

/// Registers all passes for symbolic access with the global registry.
MLIR_CAPI_EXPORTED void bishengirRegisterAllPasses(void);

MLIR_CAPI_EXPORTED void bishengirRegisterAllTranslations(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // BISHENGIR_C_REGISTEREVERYTHING_H
