//===- AVECostModelInfo.h - AVE operation cost table --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVMAVE_COSTMODELINFO_AVECOSTMODELINFO_H
#define BISHENGIR_DIALECT_HIVMAVE_COSTMODELINFO_AVECOSTMODELINFO_H

#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfoBase.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hivmave {

enum class AVECostTypeKind {
  F8E4M3FN,
  F8E5M2,
  F16,
  BF16,
  F32,
  I8,
  I16,
  I32,
  I64,
  Unknown
};

using AVEDestinationConfigMap =
    llvm::DenseMap<AVECostTypeKind, analysis::CostInfo>;
using AVETypeConfigMap =
    llvm::DenseMap<AVECostTypeKind, AVEDestinationConfigMap>;
using AVEOpConfigMap = llvm::DenseMap<TypeID, AVETypeConfigMap>;

/// A5 SIMD costs keyed by AVE operation and source/result element types.
/// execInterval and execUnit come from the target throughput/issue-slot table.
/// Unless noted otherwise, execLatency retains the previous VFFusion-derived
/// or AVE fallback value until target latency data is available.
class AVECostModelInfo {
public:
  const AVEOpConfigMap &getConfigMap() const { return opCostInfos; }

  static const AVECostModelInfo &getInstance() {
    static const AVECostModelInfo instance;
    return instance;
  }

private:
  static AVEOpConfigMap makeOpConfig() {
    AVEOpConfigMap config;

// Keep the target data declarative and separate from the lookup machinery.
// New AVE operations intentionally use the fallback cost until a measured or
// otherwise confirmed entry is added to the table.
#define AVE_COST_ENTRY(OP, SOURCE, RESULT, INTERVAL, LATENCY, UNIT)            \
  config[TypeID::get<OP>()][AVECostTypeKind::SOURCE].insert(                   \
      {AVECostTypeKind::RESULT, analysis::CostInfo{INTERVAL, LATENCY, UNIT}});
#include "bishengir/Dialect/HIVMAVE/CostModelInfo/AVECostModelInfo.def"
#undef AVE_COST_ENTRY

    return config;
  }

  const AVEOpConfigMap opCostInfos = makeOpConfig();
};

} // namespace hivmave
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVMAVE_COSTMODELINFO_AVECOSTMODELINFO_H
