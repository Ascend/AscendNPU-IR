//===- InsertPropagation.h --- Insert Propagation for trivial operations --===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_INSERT_PROPAGATION_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_INSERT_PROPAGATION_H

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"

namespace mlir {
namespace hivm {

enum InsertionPriority : uint8_t {
  DefaultInsertion = 1,
  RegbaseInsertion = 2,
  TightCoupledBufferInsertion = 3
};

struct InsertPropagationPattern : public RewritePattern {
public:
  explicit InsertPropagationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/DefaultInsertion,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  bool isPropagatorInserted(Operation *op) const;
  LogicalResult insertPropagatorForCubeOp(Operation *op,
                                          PatternRewriter &rewriter) const;
  LogicalResult insertPropagatorForVectorOp(Operation *op,
                                            PatternRewriter &rewriter) const;
  LogicalResult insertPropagatorForDMAOp(Operation *op,
                                         PatternRewriter &rewriter) const;
  std::optional<LogicalResult>
  handleSpecialCase(Operation *op, PatternRewriter &rewriter) const;
};

struct A5InsertionPattern : public RewritePattern {
public:
  explicit A5InsertionPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/RegbaseInsertion,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  bool isPropagatorInserted(Operation *op) const;
};

struct TightCoupledBufferInsertionPattern : public RewritePattern {
public:
  explicit TightCoupledBufferInsertionPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(),
                       /*benefit=*/TightCoupledBufferInsertion, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  bool isPropagatorInserted(Operation *op) const;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_PROPAGATE_OP_H