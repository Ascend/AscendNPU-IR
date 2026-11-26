//===- DagVerifier.cpp - SMS DAG validation ------------------------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.h"

#include <set>
#include <string>

namespace bishengir::hivm::sms {

VerifyResult verifyDag(const DDG &ddg) {
  static const std::set<std::string> validPipelines{
      "cube", "vector", "vector1", "vector2",
      "mte1", "mte2",   "mte3",    "fixpipe"};
  if (ddg.size() == 0)
    return {false, "DDG must not be empty"};

  for (int i = 0; i < ddg.size(); ++i) {
    const OpNode &node = ddg.getNodes()[i];
    if (!validPipelines.count(node.pipeline))
      return {false, "node " + std::to_string(i) + " has invalid pipeline '" +
                         node.pipeline + "'"};
    if (node.latency <= 0)
      return {false, "node " + std::to_string(i) + " has non-positive latency"};
    if (node.issueTime <= 0)
      return {false,
              "node " + std::to_string(i) + " has non-positive issueTime"};
  }

  if (topoOrder(ddg).size() != static_cast<size_t>(ddg.size()))
    return {false, "DDG contains a cycle"};
  return {true, {}};
}

} // namespace bishengir::hivm::sms
