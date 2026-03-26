//===- NormalizePatterns.h ---------------------------------------*- C++ -*-===//
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

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mutex.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#define DEBUG_TYPE "hfusion-normalize-ops"

using namespace mlir;
namespace mlir::hfusion {

extern thread_local bool archIsRegbased;
extern thread_local bool archisAscend950;
extern thread_local bool archisAscend310B;
extern thread_local bool archisMembased;

void populateNormalizeTrigPatterns(RewritePatternSet &patterns, bool enableHighPrecision);
void populateNormalizeI8I32CmpPatterns(RewritePatternSet &patterns);
void populateNormalizeCmpToCastPatterns(RewritePatternSet &patterns);
void populateNormalizeComparisonCleanupPatterns(RewritePatternSet &patterns);
void populateNormalizeBitwiseComparisonPatterns(RewritePatternSet &patterns);
void populateNormalizeCmpVnePatterns(RewritePatternSet &patterns);
void populateNormalizeMulRecPatterns(RewritePatternSet &patterns);
void populateNormalizeArithmeticPatterns(RewritePatternSet &patterns);
void populateNormalizePreFinalArithmeticPatterns(RewritePatternSet &patterns);
void populateNormalizeFinalArithmeticPatterns(RewritePatternSet &patterns);
void populateNormalizeModPatterns(RewritePatternSet &patterns);
void populateNormalizePrimaryMathPatterns(RewritePatternSet &patterns);
void populateNormalizeLateMathPatterns(RewritePatternSet &patterns);
void populateNormalizeCastingPatterns(RewritePatternSet &patterns);
void populateNormalizePreScalarCastingPatterns(RewritePatternSet &patterns);
void populateNormalizeFinalCastingPatterns(RewritePatternSet &patterns);
void populateNormalizeScalarLikeHFusionPatterns(RewritePatternSet &patterns);
void populateNormalizeFinalScalarPatterns(RewritePatternSet &patterns);
void populateNormalizeI1ToTargetPatterns(RewritePatternSet &patterns);
void populateNormalizeI8ToTargetPatterns(RewritePatternSet &patterns);
void populateNormalizeF16ToF32Patterns(RewritePatternSet &patterns);
void populateNormalizePreReductionPatterns(RewritePatternSet &patterns);
void populateNormalizeReductionPatterns(RewritePatternSet &patterns);
void populateNormalizeShiftI8ToI16(RewritePatternSet &patterns);
void populateNormalizeFinalReductionPatterns(RewritePatternSet &patterns);
void populateNormalizeAtomicAndSortPatterns(RewritePatternSet &patterns);

} // namespace mlir::hfusion
