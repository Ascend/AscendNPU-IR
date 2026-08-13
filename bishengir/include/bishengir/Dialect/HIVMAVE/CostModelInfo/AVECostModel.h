//===- AVECostModel.h - AVE profitability cost model ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVMAVE_COSTMODELINFO_AVECOSTMODEL_H
#define BISHENGIR_DIALECT_HIVMAVE_COSTMODELINFO_AVECOSTMODEL_H

#include "bishengir/Dialect/HIVMAVE/CostModelInfo/AVECostModelInfo.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace hivmave {

struct AVEExecutionCostEntry {
  analysis::CostInfo cost;
  int64_t count;
};

/// Throughput and instruction-mix estimate for one AVE scheduling window.
/// Load, execute and store are independent pipelines. Execute entries retain
/// latency and execution-unit information so unrolling can also be checked
/// against the issue-queue parallelism model used by VFFusion.
struct AVEPipelineCost {
  int64_t loadInstructionCount = 0;
  int64_t storeInstructionCount = 0;
  SmallVector<AVEExecutionCostEntry, 8> executionCosts;

  void addLoad(int64_t count = 1);
  void addStore(int64_t count = 1);
  void addExecution(const analysis::CostInfo &cost, int64_t count = 1);

  AVEPipelineCost &operator+=(const AVEPipelineCost &other);
  AVEPipelineCost scaled(int64_t factor) const;

  float ioScore() const;
  float executeScore() const;
  float bottleneck() const;
  float totalThroughput() const;
  int64_t executeInstructionCount() const;
  int64_t totalInstructionCount() const;
  float requiredParallelism() const;
  float executionUnitUtilization() const;
  float issueQueueParallelism() const;
};

/// AVE profitability model backed by a direct AVE op/type cost table. AVE-only
/// or currently unmodelled instructions use an explicit fallback rather than
/// silently inheriting an unrelated arith/math cost.
class AVECostModel {
public:
  explicit AVECostModel(Operation &anchor);

  AVEPipelineCost estimateOperation(const Operation &op) const;
  AVEPipelineCost estimateLoop(scf::ForOp loop) const;

  analysis::CostInfo getInterleaveCost(Type dataType) const;
  analysis::CostInfo getDeInterleaveCost(Type dataType) const;

  /// Compare factor original iterations with the planned unrolled body. The
  /// original one-iteration body is supplied separately because its smaller
  /// body exposes more loop-level issue-queue parallelism than the unrolled
  /// body.
  bool isProfitable(const AVEPipelineCost &originalIteration,
                    const AVEPipelineCost &beforeWindow,
                    const AVEPipelineCost &afterWindow) const;

private:
  std::optional<analysis::CostInfo>
  lookupExecutionCost(const Operation &op) const;
  std::optional<analysis::CostInfo>
  lookupExecutionCost(TypeID opTypeID, Type sourceType, Type resultType) const;

  const AVEOpConfigMap *targetConfig = nullptr;
};

} // namespace hivmave
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVMAVE_COSTMODELINFO_AVECOSTMODEL_H
