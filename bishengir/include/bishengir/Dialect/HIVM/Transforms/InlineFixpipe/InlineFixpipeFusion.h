//===- InlineFixpipeFusion.h - Inline fixpipe fusions -----------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPEFUSION_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPEFUSION_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::hivm {

bool isUserSliceSwappable(FixpipeOp fixpipe,
                          tensor::ExtractSliceOp extractSlice);
bool hasQuantScaleCompileHint(VMulOp op);
bool isUserQuantScaleInlinable(FixpipeOp op, VMulOp vMulOp);

void inlineFixpipeWithQuant(PatternRewriter &rewriter, Location loc,
                            FixpipeOp op, ArrayRef<VCastOp> castChain,
                            Value newFixpipeSrcTensor);
void inlineFixpipeWithRelu(PatternRewriter &rewriter, Location loc,
                           FixpipeOp op, VReluOp reluOp);
void inlineFixpipeWithStore(PatternRewriter &rewriter, Location loc,
                            FixpipeOp op, StoreOp storeOp,
                            Value fixpipeSrcTensor);
void inlineFixpipeWithQuantScale(PatternRewriter &rewriter, FixpipeOp op,
                                 VMulOp vMulOp);
void inlineFixpipeWithTranspose(PatternRewriter &rewriter, FixpipeOp op,
                                VTransposeOp transpose);
void swapFixpipeWithExtractSlice(PatternRewriter &rewriter, Location loc,
                                 FixpipeOp op,
                                 tensor::ExtractSliceOp extractSliceOp);
void moveFixpipeOutOfScfFor(PatternRewriter &rewriter, Location loc,
                            FixpipeOp fixPipeOp, scf::ForOp scfForOp,
                            Value fixpipeResTensor);

} // namespace mlir::hivm

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPEFUSION_H
