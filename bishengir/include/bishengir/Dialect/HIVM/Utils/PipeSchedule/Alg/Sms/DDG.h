#pragma once
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/OpNode.h"
#include <vector>
namespace bishengir::hivm::sms {
struct DDGEdge {
  int src;
  int dst;
  int latency;
};
class DDG {
  std::vector<OpNode> nodes;
  std::vector<DDGEdge> edges;
  std::vector<std::vector<int>> preds, succs;
  std::vector<std::vector<DDGEdge>> inEdges, outEdges;

public:
  int size() const { return static_cast<int>(nodes.size()); }
  void addNode(OpNode node);
  void addEdge(int src, int dst);
  const std::vector<OpNode> &getNodes() const { return nodes; }
  std::vector<OpNode> &getNodes() { return nodes; }
  const std::vector<DDGEdge> &getEdges() const { return edges; }
  const std::vector<std::vector<int>> &getPreds() const { return preds; }
  const std::vector<std::vector<int>> &getSuccs() const { return succs; }
  const std::vector<std::vector<DDGEdge>> &getInEdges() const {
    return inEdges;
  }
  const std::vector<std::vector<DDGEdge>> &getOutEdges() const {
    return outEdges;
  }
  // Re-snapshot node latencies into all edge copies.  addEdge() copies the
  // source node's latency at edge-creation time; when node latencies are
  // rewritten afterwards (e.g. by a cost-model pass), call this to bring
  // edges/inEdges/outEdges back in sync.
  void refreshEdgeLatencies();
  void clearEdges();
};
} // namespace bishengir::hivm::sms
