//===- ConvertMemRefToBarePtr.h - Replace memref args to base ptr ---------===//
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
// This file implements a pass to replace and eliminate args in LLVM IR dialect.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_CONVERSION_HIVMTOLLVM_CONVERTMEMREFTOBAREPTR_H
#define BISHENGIR_CONVERSION_HIVMTOLLVM_CONVERTMEMREFTOBAREPTR_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTMEMREFTOBAREPTR
#include "bishengir/Conversion/Passes.h.inc"
#include "bishengir/Conversion/HIVMToLLVM/MemRefAndTritonGlobalConstants.h"

std::unique_ptr<Pass> createConvertMemRefToBarePtrPass();
} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOLLVM_CONVERTMEMREFTOBAREPTR_H
