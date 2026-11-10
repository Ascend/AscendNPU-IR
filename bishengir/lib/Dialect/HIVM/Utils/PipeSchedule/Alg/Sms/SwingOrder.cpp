//===- SwingOrder.cpp - Swing modulo scheduling order -------------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/SwingOrder.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/AsapAlap.h"

#include <algorithm>
#include <queue>
#include <tuple>

namespace bishengir::hivm::sms {
namespace {
using HeapKey = std::tuple<int, int, int>;

int popUnvisited(std::priority_queue<HeapKey, std::vector<HeapKey>,
                                     std::greater<HeapKey>> &heap,
                 const std::vector<bool> &visited) {
  while (!heap.empty()) {
    int idx = std::get<2>(heap.top());
    heap.pop();
    if (!visited[idx])
      return idx;
  }
  return -1;
}
} // namespace

std::vector<OrderEntry> buildScheduleOrder(const DDG &ddg,
                                           const std::vector<int> &asap,
                                           const std::vector<int> &mov,
                                           const std::vector<int> &depth,
                                           const std::vector<int> &height) {
  const int n = ddg.size();
  std::vector<bool> visited(n, false);
  std::vector<OrderEntry> order;
  order.reserve(n);
  if (n == 0)
    return order;

  // 6a: seed = argmax(asap desc, mov asc, idx asc)
  // Encode as min(-asap, mov, idx) to pick the right node.
  int seed = 0;
  for (int i = 1; i < n; ++i) {
    const HeapKey candidate{-asap[i], mov[i], i};
    const HeapKey current{-asap[seed], mov[seed], seed};
    if (candidate < current)
      seed = i;
  }

  std::priority_queue<HeapKey, std::vector<HeapKey>, std::greater<HeapKey>>
      heap;
  heap.emplace(-depth[seed], mov[seed], seed);
  while (true) {
    int idx = popUnvisited(heap, visited);
    if (idx < 0)
      break;
    visited[idx] = true;
    order.push_back({idx, false});
    for (const DDGEdge &edge : ddg.getInEdges()[idx])
      if (!visited[edge.src])
        heap.emplace(-depth[edge.src], mov[edge.src], edge.src);
  }

  std::priority_queue<HeapKey, std::vector<HeapKey>, std::greater<HeapKey>>
      branch;
  for (int i = 0; i < n; ++i) {
    bool ready = !visited[i];
    for (int pred : ddg.getPreds()[i])
      ready &= visited[pred];
    if (ready)
      branch.emplace(-height[i], mov[i], i);
  }
  while (true) {
    int idx = popUnvisited(branch, visited);
    if (idx < 0)
      break;
    visited[idx] = true;
    order.push_back({idx, true});
    for (const DDGEdge &edge : ddg.getOutEdges()[idx]) {
      if (visited[edge.dst])
        continue;
      bool ready = true;
      for (int pred : ddg.getPreds()[edge.dst])
        ready &= visited[pred];
      if (ready)
        branch.emplace(-height[edge.dst], mov[edge.dst], edge.dst);
    }
  }

  for (int idx : topoOrder(ddg))
    if (!visited[idx]) {
      visited[idx] = true;
      order.push_back({idx, true});
    }
  return order;
}

} // namespace bishengir::hivm::sms
