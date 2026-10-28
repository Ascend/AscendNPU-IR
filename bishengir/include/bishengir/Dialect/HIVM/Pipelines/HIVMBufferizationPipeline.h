//===- HIVMBufferizationPipeline.h - HIVM bufferization pipeline -*- C++
//-*-===//
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
// This header file declares the tensor -> memref bufferization pipelines
// shared by the regbase, non-regbase and v2 HIVM pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_HIVMBUFFERIZATIONPIPELINE_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_HIVMBUFFERIZATIONPIPELINE_H

#include "bishengir/Dialect/HIVM/Pipelines/HIVMPipelineBase.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace hivm {

/// The core bufferization passes (slice folding, NormalizeToTensorOp, tensor
/// copy insertion and one-shot-bufferize) shared by every HIVM pipeline.
/// `isRegBased` selects the reg-based flavor.
void bufferizationPipelineBase(OpPassManager &pm, bool isRegBased);

/// The full bufferization pipeline, including the pre-bufferization
/// optimizations and the post-bufferization cleanups. The reg-based or
/// non-reg-based flavor is selected by `hivmPipelineOptions.target`.
void bufferizationPipeline(OpPassManager &pm,
                           const HIVMPipelineOptions &hivmPipelineOptions);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_HIVMBUFFERIZATIONPIPELINE_H
