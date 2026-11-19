//===- MRT.cpp - Modulo reservation table --------------------------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/MRT.h"

#include <cassert>
#include <set>

namespace bishengir::hivm::sms {
namespace {
const std::set<std::string> singleUnits{"cube", "vector", "vector1", "vector2"};
const std::set<std::string> parallelUnits{"mte1", "mte2", "mte3", "fixpipe"};

int modulo(int value, int divisor) {
  int result = value % divisor;
  return result < 0 ? result + divisor : result;
}
} // namespace

MRT::MRT(int ii) : ii(ii) {
  assert(ii > 0 && "initiation interval must be positive");
  for (const std::string &pipeline : singleUnits)
    slots.emplace(pipeline, std::vector<int>(ii, -1));
  for (const std::string &pipeline : parallelUnits)
    parallelLog.emplace(pipeline, std::vector<std::tuple<int, int, int>>{});
}

bool MRT::isParallel(const std::string &pipeline) const {
  return parallelUnits.count(pipeline) != 0;
}

bool MRT::isIntervalFree(int cycle, const std::string &pipeline,
                         int duration) const {
  if (isParallel(pipeline))
    return true;
  auto iter = slots.find(pipeline);
  if (iter == slots.end())
    return true;
  for (int offset = 0; offset < duration; ++offset)
    if (iter->second[modulo(cycle + offset, ii)] != -1)
      return false;
  return true;
}

void MRT::reserve(int cycle, const std::string &pipeline, int nodeIdx,
                  int duration) {
  assert(duration > 0 && "reservation duration must be positive");
  if (isParallel(pipeline)) {
    auto &log = parallelLog[pipeline];
    for (int offset = 0; offset < duration; ++offset)
      log.emplace_back(modulo(cycle + offset, ii), nodeIdx, duration);
    return;
  }
  auto iter = slots.find(pipeline);
  if (iter == slots.end())
    return;
  assert(isIntervalFree(cycle, pipeline, duration) &&
         "reservation conflicts with an existing node");
  for (int offset = 0; offset < duration; ++offset)
    iter->second[modulo(cycle + offset, ii)] = nodeIdx;
}

void MRT::rebuild(int newII, const std::map<int, int> &scheduled,
                  const DDG &ddg) {
  MRT rebuilt(newII);
  for (const auto &[nodeIdx, cycle] : scheduled) {
    assert(nodeIdx >= 0 && nodeIdx < ddg.size());
    const OpNode &node = ddg.getNodes()[nodeIdx];
    rebuilt.reserve(cycle, node.pipeline, nodeIdx, node.issueTime);
  }
  *this = std::move(rebuilt);
}

} // namespace bishengir::hivm::sms
