//===- SmsScheduler.cpp - Swing modulo scheduler -------------------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SmsScheduler.h"

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DepthHeight.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/MRT.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/ResMII.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SwingOrder.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <vector>

// Unified debug channel for the MS preload scheduler pipeline; the message
// prefix identifies the producing component.
#define DEBUG_TYPE "ms-preload-scheduler"

using llvm::dbgs;

namespace bishengir::hivm::sms {
namespace {

std::optional<int> placeNode(const DDG &ddg, int idx, int ii,
                             const std::vector<int> &asap,
                             const std::vector<int> &alap, const MRT &mrt,
                             const std::map<int, int> &scheduled) {
  const OpNode &node = ddg.getNodes()[idx];
  std::optional<int> lower;
  for (const DDGEdge &edge : ddg.getInEdges()[idx]) {
    auto iter = scheduled.find(edge.src);
    if (iter == scheduled.end())
      continue;
    int candidate = iter->second + edge.latency;
    lower = lower ? std::max(*lower, candidate) : candidate;
  }

  std::optional<int> upper;
  for (const DDGEdge &edge : ddg.getOutEdges()[idx]) {
    auto iter = scheduled.find(edge.dst);
    if (iter == scheduled.end())
      continue;
    int candidate = iter->second - edge.latency;
    upper = upper ? std::min(*upper, candidate) : candidate;
  }
  if (lower && upper && *lower > *upper) {
    LLVM_DEBUG(dbgs() << "[sms-scheduler] II=" << ii << ": node #" << idx
                      << " '" << node.name << "' dependency window empty"
                      << " (lower=" << *lower << ", upper=" << *upper << ")\n");
    return std::nullopt;
  }

  int cycle = asap[idx];
  if (scheduled.empty()) {
    cycle = asap[idx];
  } else if (lower && upper) {
    if ((*upper - asap[idx]) <= (asap[idx] - *lower))
      cycle = *upper;
    else
      cycle = *lower;
  } else if (upper) {
    cycle = *upper;
  } else if (lower) {
    cycle = *lower;
  }

  if (mrt.isParallel(node.pipeline) ||
      mrt.isIntervalFree(cycle, node.pipeline, node.issueTime))
    return cycle;

  int direction = -1;
  if (lower && !upper) {
    direction = 1;
  } else if (lower && upper) {
    const int roomTowardSmallerCycles = cycle - *lower;
    const int roomTowardLargerCycles = *upper - cycle;
    direction = roomTowardLargerCycles <= roomTowardSmallerCycles ? -1 : 1;
  }

  const int lowerBound = lower.value_or(std::numeric_limits<int>::min());
  const int upperBound =
      upper.value_or(std::max(alap[idx], asap[idx] + ii - 1));
  for (int delta = 0; delta < ii; ++delta) {
    int candidate = cycle + direction * delta;
    if (candidate < lowerBound || candidate > upperBound)
      break;
    if (mrt.isIntervalFree(candidate, node.pipeline, node.issueTime))
      return candidate;
  }
  LLVM_DEBUG({
    dbgs() << "[sms-scheduler] II=" << ii << ": node #" << idx << " '"
           << node.name << "' no free " << node.pipeline
           << " slot (issueTime=" << node.issueTime << ") near cycle " << cycle;
    if (lower)
      dbgs() << ", lower=" << *lower;
    if (upper)
      dbgs() << ", upper=" << *upper;
    dbgs() << "\n";
  });
  return std::nullopt;
}

std::optional<ScheduleResult> trySchedule(const DDG &ddg, int ii) {
  const std::vector<int> asap = computeASAP(ddg);
  const int maxAsap =
      asap.empty() ? 0 : *std::max_element(asap.begin(), asap.end());
  const std::vector<int> alap = computeALAP(ddg, maxAsap + ii);
  const std::vector<int> mov = computeMOV(asap, alap);
  const std::vector<OrderEntry> order =
      buildScheduleOrder(ddg, asap, mov, computeDepth(ddg), computeHeight(ddg));

  MRT mrt(ii);
  std::map<int, int> scheduled;
  for (const OrderEntry &entry : order) {
    std::optional<int> cycle =
        placeNode(ddg, entry.nodeIdx, ii, asap, alap, mrt, scheduled);
    if (!cycle)
      return std::nullopt;
    scheduled[entry.nodeIdx] = *cycle;
    const OpNode &node = ddg.getNodes()[entry.nodeIdx];
    mrt.reserve(*cycle, node.pipeline, entry.nodeIdx, node.issueTime);
  }

  if (!scheduled.empty()) {
    int minCycle = scheduled.begin()->second;
    for (const auto &[nodeIdx, cycle] : scheduled)
      minCycle = std::min(minCycle, cycle);
    if (minCycle < 0) {
      for (auto &[nodeIdx, cycle] : scheduled)
        cycle -= minCycle;
      mrt.rebuild(ii, scheduled, ddg);
    }
  }

  ScheduleResult result{ii, scheduled, {}, 0};
  int maxStage = -1;
  for (const auto &[nodeIdx, cycle] : scheduled) {
    int stage = cycle / ii;
    result.nodeToStage[nodeIdx] = stage;
    maxStage = std::max(maxStage, stage);
  }
  result.stageCount = maxStage + 1;
  return result;
}

} // namespace

std::optional<ScheduleResult> smsSchedule(const DDG &ddg, int maxIISearch) {
  if (maxIISearch < 0) {
    LLVM_DEBUG(dbgs() << "[sms-scheduler] invalid maxIISearch=" << maxIISearch
                      << "\n");
    return std::nullopt;
  }
  VerifyResult vr = verifyDag(ddg);
  if (!vr.ok) {
    LLVM_DEBUG(dbgs() << "[sms-scheduler] DAG verification failed: " << vr.error
                      << "\n");
    return std::nullopt;
  }
  const int startII = computeResMII(ddg);
  LLVM_DEBUG(dbgs() << "[sms-scheduler] scheduling " << ddg.size()
                    << " nodes, ResMII=" << startII << ", searching II in ["
                    << startII << ", " << startII + maxIISearch << "]\n");
  for (int ii = startII; ii <= startII + maxIISearch; ++ii)
    if (std::optional<ScheduleResult> result = trySchedule(ddg, ii)) {
      LLVM_DEBUG(dbgs() << "[sms-scheduler] schedule found: II=" << result->ii
                        << ", stages=" << result->stageCount << "\n");
      return result;
    }
  LLVM_DEBUG(dbgs() << "[sms-scheduler] no feasible II in [" << startII << ", "
                    << startII + maxIISearch << "]\n");
  return std::nullopt;
}

} // namespace bishengir::hivm::sms
