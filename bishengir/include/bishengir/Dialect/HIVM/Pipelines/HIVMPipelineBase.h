//===- HIVMPipelineBase.h - Shared HIVM pipeline base -----------*- C++ -*-===//
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
// This header file declares the pipeline options and the building blocks that
// are shared by the regbase and non-regbase HIVM pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_HIVMPIPELINEBASE_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_HIVMPIPELINEBASE_H

#include "bishengir/Dialect/HACC/IR/HACCInterfaces.h"
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.h.inc"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hivm {

struct HIVMPipelineOptions
    : public mlir::PassPipelineOptions<HIVMPipelineOptions> {
#define GEN_HIVM_OPTION_REGISTRATION
#include "bishengir/Tools/bishengir-compile/PassPipelineOptions.cpp.inc"
};

/// A canonicalization pipeline for HIVM pipeline.
void canonicalizationHIVMPipeline(OpPassManager &pm);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_HIVMPIPELINEBASE_H
