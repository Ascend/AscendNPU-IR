//===- VFFusionHIVM.h - HIVM VF fusion partition helpers -------*- C++ --*-===//
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

#ifndef BISHENGIR_DIALECT_ANALYSIS_PARTITION_HIVM_VFFUSIONHIVM_H
#define BISHENGIR_DIALECT_ANALYSIS_PARTITION_HIVM_VFFUSIONHIVM_H

#include "bishengir/Dialect/Analysis/Partition/Partition.h"
namespace mlir::analysis::partition::hivm {

enum class VFFusionMode {
  LinearScan,
};

struct LinearScanFusion : public Partition<LinearScanFusion> {
  static LogicalResult rewriteImpl(Group &&group);
};

Groups sameBlockRelation(GroupRef group);
Groups filterFuncOpRelation(GroupRef group);
Groups linearScanVectorizableOpsRelation(GroupRef group);

void handoverGroupForOutlining(Group &&g);

} // namespace mlir::analysis::partition::hivm

#endif // BISHENGIR_DIALECT_ANALYSIS_PARTITION_HIVM_VFFUSIONHIVM_H
