//===- DepthHeight.cpp - DDG depth and height analyses -------------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DepthHeight.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.h"

#include <algorithm>

namespace bishengir::hivm::sms {

std::vector<int> computeDepth(const DDG &ddg) {
  std::vector<int> depth(ddg.size(), 0);
  for (int idx : topoOrder(ddg)) {
    const int latency = ddg.getNodes()[idx].latency;
    for (const DDGEdge &edge : ddg.getInEdges()[idx])
      depth[idx] = std::max(depth[idx], depth[edge.src] + latency);
  }
  return depth;
}

std::vector<int> computeHeight(const DDG &ddg) {
  std::vector<int> height(ddg.size(), 0);
  std::vector<int> order = topoOrder(ddg);
  for (auto iter = order.rbegin(); iter != order.rend(); ++iter) {
    int idx = *iter;
    const int latency = ddg.getNodes()[idx].latency;
    for (const DDGEdge &edge : ddg.getOutEdges()[idx])
      height[idx] = std::max(height[idx], height[edge.dst] + latency);
  }
  return height;
}

} // namespace bishengir::hivm::sms
