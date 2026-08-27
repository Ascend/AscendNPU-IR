//===- FusibleProducerAnalyzer.h - Analyze producer info for schedule -----===//
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
#ifndef BISHENGIR_DIALECT_HFUSION_ANALYSIS_FUSIBLE_PRODUCER_ANALYZER_H
#define BISHENGIR_DIALECT_HFUSION_ANALYSIS_FUSIBLE_PRODUCER_ANALYZER_H

#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hfusion {
namespace detail {

/// Forward declarations
struct OpInfo;
struct StoreOpInfo;
struct ReduceInfo;

enum class ConsumerType : uint8_t {
  kUnknown,
  kOutput,   // Output nodes (a.k.a operands of `func.return` op)
  kReduction // Reduction ops
};

/// Represent a consumer operation that has one or more reduction axes.
/// Note that one consumer can have multiple `FusibleProducers`,
/// one for each reduction axis.
class ConsumerWithReduction {
public:
  ConsumerWithReduction() = default;
  explicit ConsumerWithReduction(Operation *consumer, ConsumerType type,
                                 NamedAttribute identifier);

  /// Get the consumer type.
  ConsumerType getType() const { return type; }

  /// Get the unique identifier for the consumer.
  NamedAttribute getIdentifier() const { return *identifier; }

private:
  ConsumerType type{ConsumerType::kUnknown};
  std::optional<NamedAttribute> identifier{std::nullopt};
};

/// A fusible producers represents a group of producers operations that can be
/// fused into the containing loop of a consumer. Since the consumer may have
/// multiple reduction axis, each instance of fusible producers will only be
/// fusible to one loop.
///
/// For example:
///   A         = [d0, d1]
///   B         = [d0    ]
///   C         = [    d1]
///   Consumer  = [d0, d1]
/// For reduction axis 0 of the consumer, the fusible producers are A and B.
/// For axis 1, the fusible producers are A and C.
class FusibleProducers {
public:
  FusibleProducers() = default;

  /// Get the unique identifier for the group of producers.
  NamedAttribute getIdentifier() const { return *identifier; }

  /// Set the producer info.
  void setProducerInfo(SetVector<Operation *> producers,
                       NamedAttribute identifier);

private:
  /// All the producers related to the axis.
  SetVector<Operation *> producers;
  std::optional<NamedAttribute> identifier{std::nullopt};
};

/// A pair of consumer op and the reduction dimension w.r.t anchor.
using ConsumerReduceAxis = std::pair<Operation *, size_t>;

/// Mapping from consumer's axis to the fusible producer info.
using Consumer2ProducerMap = std::map<ConsumerReduceAxis, FusibleProducers>;

/// Mapping from consumer op to its information.
using Consumer2InfoMap = std::map<Operation *, ConsumerWithReduction>;

/// Structure for holding the analysis result for a single consumer.
struct FusibleProducerAnalysisResult {
  ConsumerWithReduction consumerInfo;
  Consumer2ProducerMap consumer2ProducerMap{};
};

/// Main entry to analyze linalg.reduce op's fusible producers for different
/// reduction axes.
FusibleProducerAnalysisResult
analyzeProducersForReductionOp(linalg::ReduceOp reduceOp,
                               const ReduceInfo &reduceOpInfo,
                               DimensionAnalyzer *analyzer);

/// Main entry to analyze hfusion.store's fusible producers for different
/// reduction axes.
FailureOr<FusibleProducerAnalysisResult>
analyzeProducersForStoreOp(hfusion::StoreOp storeOp, StoreOpInfo &storeOpInfo,
                           const SetVector<int64_t> &reduceDimsInAnchor,
                           DimensionAnalyzer *analyzer);

} // namespace detail
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_ANALYSIS_FUSIBLE_PRODUCER_ANALYZER_H
