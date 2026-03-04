//===- AnyPBRKernelInfo.h -- AnyPBR Kernel Info ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRKERNELINFO_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRKERNELINFO_H

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "mlir/IR/Attributes.h"

namespace mlir {
namespace hfusion {

class AnyPBRKernelInfo : public KernelInfo {
public:
  explicit AnyPBRKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::AnyPBR, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::AnyPBR;
  }

  /// Get the consumer and producer info.
  const detail::Consumer2InfoMap &getConsumer2Info() const;

  /// Record the consumer and its fusible producers.
  void recordFusibleProducerAnalysisResult(
      detail::FusibleProducerAnalysisResult &&result);

  /// Get the set of fusible producer tags given a consumer and tiling key.
  SmallVector<NamedAttribute> getReductionProducers(Operation *consumer,
                                                    int64_t key) const;

  /// Reduction dimension shared by all reduce op.
  SetVector<int64_t> reduceDimsInAnchor;

private:
  /// Mapping from a pair of consumer op and the reduction dimension to the
  /// fusible producers.
  /// \note the reduction dimension is w.r.t. the global anchor.
  detail::Consumer2ProducerMap consumer2Producer_{};

  /// Consumers that have fusible producers.
  detail::Consumer2InfoMap consumer2Info_{};
};
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBRKERNELINFO_H
