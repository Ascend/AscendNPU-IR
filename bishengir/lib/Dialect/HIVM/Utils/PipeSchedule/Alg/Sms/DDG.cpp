//===- DDG.cpp - SMS data dependence graph -------------------------------===//
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"

#include <cassert>
#include <utility>

namespace bishengir::hivm::sms {

void DDG::addNode(OpNode node) {
  nodes.push_back(std::move(node));
  preds.emplace_back();
  succs.emplace_back();
  inEdges.emplace_back();
  outEdges.emplace_back();
}

void DDG::addEdge(int src, int dst) {
  assert(src >= 0 && src < size() && "source index outside DDG");
  assert(dst >= 0 && dst < size() && "destination index outside DDG");
  DDGEdge edge{src, dst, nodes[src].latency};
  edges.push_back(edge);
  preds[dst].push_back(src);
  succs[src].push_back(dst);
  inEdges[dst].push_back(edge);
  outEdges[src].push_back(edge);
}

void DDG::refreshEdgeLatencies() {
  for (DDGEdge &edge : edges)
    edge.latency = nodes[edge.src].latency;
  for (auto &perNode : inEdges)
    for (DDGEdge &edge : perNode)
      edge.latency = nodes[edge.src].latency;
  for (auto &perNode : outEdges)
    for (DDGEdge &edge : perNode)
      edge.latency = nodes[edge.src].latency;
}

void DDG::clearEdges() {
  edges.clear();
  for (auto &pred : preds)
    pred.clear();
  for (auto &succ : succs)
    succ.clear();
  for (auto &in : inEdges)
    in.clear();
  for (auto &out : outEdges)
    out.clear();
}

} // namespace bishengir::hivm::sms
