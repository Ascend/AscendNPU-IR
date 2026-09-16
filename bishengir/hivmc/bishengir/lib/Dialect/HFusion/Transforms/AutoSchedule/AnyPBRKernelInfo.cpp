//===- AnyPBRKernelInfo.cpp -- Definition for AnyPBR Kernel Info ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements anypbr kernel info definition.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRKernelInfo.h"

using namespace mlir;
using namespace mlir::hfusion;

//===----------------------------------------------------------------------===//
// AnyPBRKernelInfo
//===----------------------------------------------------------------------===//

void AnyPBRKernelInfo::recordFusibleProducerAnalysisResult(
    detail::FusibleProducerAnalysisResult &&result) {
  Operation *consumer = nullptr;
  for (auto [key, value] : result.consumer2ProducerMap) {
    if (!consumer)
      consumer = key.first;

    auto [_, isInserted] = consumer2Producer_.try_emplace(key, value);
    if (!isInserted)
      llvm_unreachable("duplicate consumer + axis pair");
  }
  assert(consumer != nullptr);
  auto [_, isInserted] =
      consumer2Info_.try_emplace(consumer, result.consumerInfo);
  if (!isInserted)
    llvm_unreachable("duplicate consumer");
}

SmallVector<NamedAttribute>
AnyPBRKernelInfo::getReductionProducers(Operation *consumer,
                                        int64_t key) const {
  // If the pair {consumer, anchorDim} is recorded in the
  // `consumer2ProducerMap`, it means that:
  //    1) the anchorDim is a reduce axis
  //    2) the consumer has some producers
  //
  // So we can construct pairs from {consumer, 0}, ..., {consumer, anchorDim}
  // And for all the dim-idx less than or equal to the key, we can get the
  // producers tags.P
  SmallVector<NamedAttribute> producerTags;
  for (auto dimIdx = 0; dimIdx <= key; ++dimIdx) {
    auto iter = consumer2Producer_.find({consumer, dimIdx});
    if (iter == consumer2Producer_.cend())
      continue;

    producerTags.push_back(iter->second.getIdentifier());
  }
  return producerTags;
}

const hfusion::detail::Consumer2InfoMap &
AnyPBRKernelInfo::getConsumer2Info() const {
  return consumer2Info_;
}
