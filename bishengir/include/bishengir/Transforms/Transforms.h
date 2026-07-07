//====- Transforms.h - Transform Extend Fuse Into ContainingOp ---*- C++ -*-==//
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
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

#ifndef BISHENGIR_TRANSFORMS_TRANSFORMS_H
#define BISHENGIR_TRANSFORMS_TRANSFORMS_H

namespace bishengir {
void unionProducerUsers(mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
                        mlir::Operation *producerOp,
                        mlir::Operation *containingOp);
std::tuple<llvm::SmallVector<mlir::Operation *>, mlir::Operation *>

tileAndFuseFirstExtractUse(mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
                           mlir::Operation *producerOp,
                           mlir::Operation *containingOp,
                           bool duplicateProducer);
llvm::SmallVector<mlir::Operation *>

tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    mlir::RewriterBase &rewriter, mlir::Diagnostic &diag,
    mlir::Operation *producerOp, mlir::Operation *containingOp);

mlir::Operation *cloneAndFuseFirstUse(mlir::RewriterBase &rewriter,
                                      mlir::Diagnostic &diag,
                                      mlir::Operation *producerOp,
                                      mlir::Operation *containingOp);

void normalizeLoop(mlir::RewriterBase &rewriter, mlir::scf::ForOp op,
                   mlir::Value oldStep);

void fuseNestedSiblingLoops(mlir::Operation *fusedOp,
                             mlir::RewriterBase &rewriter, bool recursive);

} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_TRANSFORMS_H