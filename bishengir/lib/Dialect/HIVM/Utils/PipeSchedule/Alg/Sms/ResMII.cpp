//===- ResMII.cpp - Resource minimum initiation interval -----------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/ResMII.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>

namespace bishengir::hivm::sms {

int computeResMII(const DDG &ddg) {
  static const std::set<std::string> singleUnits{"cube", "vector", "vector1",
                                                 "vector2"};
  std::map<std::string, int> totals;
  for (const OpNode &node : ddg.getNodes())
    if (singleUnits.count(node.pipeline))
      totals[node.pipeline] += node.issueTime;

  int result = 1;
  for (const auto &entry : totals)
    result = std::max(result, entry.second);
  return result;
}

} // namespace bishengir::hivm::sms
