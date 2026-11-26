//===- AsapAlap.cpp - ASAP, ALAP, and mobility analyses -----------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.h"

#include <algorithm>
#include <cassert>
#include <queue>

namespace bishengir::hivm::sms {

std::vector<int> topoOrder(const DDG &ddg) {
  std::vector<int> inDegree(ddg.size());
  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (int i = 0; i < ddg.size(); ++i) {
    inDegree[i] = static_cast<int>(ddg.getPreds()[i].size());
    if (inDegree[i] == 0)
      ready.push(i);
  }

  std::vector<int> order;
  order.reserve(ddg.size());
  while (!ready.empty()) {
    int node = ready.top();
    ready.pop();
    order.push_back(node);
    for (int succ : ddg.getSuccs()[node])
      if (--inDegree[succ] == 0)
        ready.push(succ);
  }
  return order;
}

std::vector<int> computeASAP(const DDG &ddg) {
  std::vector<int> asap(ddg.size(), 0);
  for (int idx : topoOrder(ddg))
    for (const DDGEdge &edge : ddg.getInEdges()[idx])
      asap[idx] = std::max(asap[idx], asap[edge.src] + edge.latency);
  return asap;
}

std::vector<int> computeALAP(const DDG &ddg, int horizon) {
  std::vector<int> alap(ddg.size(), horizon);
  std::vector<int> order = topoOrder(ddg);
  for (auto iter = order.rbegin(); iter != order.rend(); ++iter) {
    int idx = *iter;
    if (ddg.getOutEdges()[idx].empty()) {
      alap[idx] = horizon - ddg.getNodes()[idx].latency;
      continue;
    }
    int best = horizon;
    for (const DDGEdge &edge : ddg.getOutEdges()[idx])
      best = std::min(best, alap[edge.dst] - edge.latency);
    alap[idx] = best;
  }
  return alap;
}

std::vector<int> computeMOV(const std::vector<int> &asap,
                            const std::vector<int> &alap) {
  assert(asap.size() == alap.size() && "ASAP and ALAP sizes differ");
  std::vector<int> mov(asap.size());
  for (size_t i = 0; i < asap.size(); ++i)
    mov[i] = alap[i] - asap[i];
  return mov;
}

} // namespace bishengir::hivm::sms
