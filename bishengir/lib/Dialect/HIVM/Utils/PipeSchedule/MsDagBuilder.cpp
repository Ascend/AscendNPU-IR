//===- MsDagBuilder.cpp - MS DAG from MLIR IR ---------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Read-only MLIR → sms::DDG bridge.  Every static helper is a controlled copy
// of the corresponding WorklistBuilder.cpp function (named in comments).
// The only difference from the originals is that we NEVER set
// pipeline.cubeonly / pipeline.veconly attributes — classification results
// are returned in MsDagBuildResult instead.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/MsDagBuilder.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DagVerifier.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <utility>

// Unified debug channel for the MS preload scheduler pipeline; the message
// prefix identifies the producing component.
#define DEBUG_TYPE "ms-preload-scheduler"

using llvm::dbgs;

namespace mlir {
using namespace hivm;
using hivm::detail::queryCoreTypeHelper;

// Attribute names — read-only references, never written.
static constexpr llvm::StringLiteral CubeOnlyAttrName = "pipeline.cubeonly";
static constexpr llvm::StringLiteral VecOnlyAttrName = "pipeline.veconly";

//===----------------------------------------------------------------------===//
// Controlled-copy static classifiers
//===----------------------------------------------------------------------===//

// cf. WorklistBuilder.cpp: traceValueDef
Value MsDagBuilder::traceValueDef(Value v) {
  if (!v)
    return nullptr;
  if (auto result = dyn_cast<OpResult>(v)) {
    Operation *defining = result.getOwner();
    Value srcVal =
        TypeSwitch<Operation *, Value>(defining)
            .Case<tensor::ReshapeOp, tensor::ExtractSliceOp,
                  tensor::CollapseShapeOp, tensor::ExpandShapeOp,
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
                  bufferization::ToTensorOp, bufferization::ToMemrefOp,
#else
                  bufferization::ToTensorOp, bufferization::ToBufferOp,
#endif
                  memref::CastOp, memref::CollapseShapeOp,
                  memref::ExpandShapeOp, memref::MemorySpaceCastOp,
                  memref::ReinterpretCastOp, memref::ReshapeOp, memref::ViewOp,
                  memref::SubViewOp>([](auto op) { return op->getOperand(0); })
            .Case([](tensor::InsertSliceOp insert) { return insert.getDest(); })
            .Case([result](LoopLikeOpInterface loop) {
              return loop.getTiedLoopInit(result)->get();
            })
            .Default([](Operation *) { return nullptr; });
    if (srcVal)
      return traceValueDef(srcVal);
    return result;
  }

  auto blkArg = dyn_cast<BlockArgument>(v);
  if (!blkArg)
    return nullptr;
  Operation *parent = blkArg.getOwner()->getParentOp();
  auto loop = dyn_cast<LoopLikeOpInterface>(parent);
  if (!loop)
    return blkArg;
  return traceValueDef(loop.getTiedLoopInit(blkArg)->get());
}

// cf. WorklistBuilder.cpp: traceAlloc
memref::AllocOp MsDagBuilder::traceAlloc(Value v) {
  Value maybeAlloc = traceValueDef(v);
  if (!maybeAlloc)
    return nullptr;
  return dyn_cast_if_present<memref::AllocOp>(maybeAlloc.getDefiningOp());
}

// cf. WorklistBuilder.cpp: isCrossCoreCopy
bool MsDagBuilder::isCrossCoreCopy(Operation *copy) {
  auto copyOp = dyn_cast<CopyOp>(copy);
  if (!copyOp)
    return false;
  Value dst = copyOp.getDst();
  memref::AllocOp alloc = traceAlloc(dst);
  if (!alloc)
    return false;
  auto memSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(alloc.getType().getMemorySpace());
  if (!memSpaceAttr)
    return false;
  return memSpaceAttr.getAddressSpace() == AddressSpace::L1;
}

// cf. WorklistBuilder.cpp: isSeparator
bool MsDagBuilder::isSeparator(Operation *op) {
  return isa<FixpipeOp, StoreOp>(op) || isCrossCoreCopy(op);
}

// cf. WorklistBuilder.cpp: isCoreOp
// READ-ONLY: checks existing pipeline.cubeonly/veconly attrs (which may have
// been set by an earlier pass or by the default WorklistBuilder path in a
// prior invocation).  For SMS's own classification we also check the HIVM
// dialect + DPS trait, but we NEVER set the attrs ourselves.
bool MsDagBuilder::isCoreOp(Operation &op) {
  return op.hasAttr(CubeOnlyAttrName) || op.hasAttr(VecOnlyAttrName) ||
         (isa_and_nonnull<HIVMDialect>(op.getDialect()) &&
          isa<DestinationStyleOpInterface>(op));
}

// cf. WorklistBuilder.cpp: isLoadLikeOp
bool MsDagBuilder::isLoadLikeOp(Operation *op) {
  return isa<LoadOp, ND2NZOp>(op);
}

// cf. WorklistBuilder.cpp: isMemrefSubnetWriter
bool MsDagBuilder::isMemrefSubnetWriter(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps || dps.getNumDpsInits() != 1 ||
      !isa<MemRefType>(dps.getDpsInitOperand(0)->get().getType()))
    return isCrossCoreCopy(op);
  return isa<LoadOp, FixpipeOp, ND2NZOp>(op) || isCrossCoreCopy(op);
}

// cf. WorklistBuilder.cpp: isExtractedScalarGather
bool MsDagBuilder::isExtractedScalarGather(scf::ForOp forOp) {
  if (!forOp->hasAttr("ExtractedLoadOrStore"))
    return false;
  bool hasMemRefLoad = false;
  bool hasTensorWrite = false;
  forOp.getBody()->walk([&](Operation *inner) {
    if (isa<memref::LoadOp>(inner))
      hasMemRefLoad = true;
    else if (isa<tensor::InsertOp, tensor::InsertSliceOp, memref::StoreOp,
                 memref::CopyOp>(inner))
      hasTensorWrite = true;
  });
  return hasMemRefLoad && hasTensorWrite;
}

// cf. WorklistBuilder.cpp: getContainedParent
Operation *MsDagBuilder::getContainedParent(Operation *containing,
                                            Operation &inner) {
  Operation *cur = &inner;
  Operation *parent = cur->getParentOp();
  while (parent && parent != containing && containing->isAncestor(cur)) {
    cur = parent;
    parent = cur->getParentOp();
  }
  return cur;
}

Operation *MsDagBuilder::getContainedParent(Operation *containing,
                                            Value inner) {
  Operation *defining = inner.getDefiningOp();
  if (defining)
    return getContainedParent(containing, *defining);
  return nullptr;
}

// cf. WorklistBuilder.cpp: memrefDFS
void MsDagBuilder::memrefDFS(Value memrefVal, SmallVector<Operation *> &users) {
  SmallVector<Operation *> traceStack;
  DenseSet<Operation *> visited;
  Value rootVal = traceValueDef(memrefVal);
  if (!rootVal)
    return;
  traceStack.append(rootVal.user_begin(), rootVal.user_end());
  while (!traceStack.empty()) {
    Operation *op = traceStack.pop_back_val();
    if (visited.contains(op))
      continue;
    visited.insert(op);

    if (isLoadLikeOp(op)) {
      auto dpsOp = cast<DestinationStyleOpInterface>(op);
      Value dst = dpsOp.getDpsInitOperand(0)->get();
      if (traceValueDef(dst) != rootVal)
        continue;
    }

    users.push_back(op);
    if (op->getNumResults() == 1 &&
        !isa<MemRefType>(op->getResult(0).getType()))
      continue;
    traceStack.append(op->user_begin(), op->user_end());
  }
}

// Read-only variant of illegalRegionedOp.  Returns the core classification
// of a region op's body WITHOUT setting any attributes.
// cf. WorklistBuilder.cpp: illegalRegionedOp (lines 231-285)
MsDagBuilder::RegionCoreKind MsDagBuilder::classifyRegion(Operation &op) {
  if (op.getRegions().empty())
    return RegionCoreKind::None;

  bool hasCube = false;
  bool hasVector = false;
  WalkResult walkRes = op.walk([&hasCube, &hasVector](Operation *curOp) {
    if (!isa_and_nonnull<HIVMDialect>(curOp->getDialect()))
      return WalkResult::advance();
    auto core = queryCoreTypeHelper(curOp).value_or(TCoreType::CUBE_OR_VECTOR);
    if (core == TCoreType::CUBE_OR_VECTOR && isCrossCoreCopy(curOp))
      core = TCoreType::VECTOR;
    if (core == TCoreType::VECTOR) {
      if (hasCube)
        return WalkResult::interrupt();
      hasVector = true;
    } else if (core == TCoreType::CUBE) {
      if (hasVector)
        return WalkResult::interrupt();
      hasCube = true;
    }
    return WalkResult::advance();
  });

  if (walkRes.wasInterrupted())
    return RegionCoreKind::Mixed;

  // Non-HIVM scalar gather fallback.
  if (!hasCube && !hasVector) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op);
        forOp && isExtractedScalarGather(forOp))
      hasVector = true;
  }

  if (hasCube)
    return RegionCoreKind::Cube;
  if (hasVector)
    return RegionCoreKind::Vector;
  return RegionCoreKind::None;
}

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

MsDagBuilder::MsDagBuilder(scf::ForOp loop,
                           const LatencyProvider &latencyProvider,
                           const DenseMap<Value, Operation *> &counterClones,
                           const WorklistBuildResult &defaultBuild,
                           const costmodel::CostModel *costModel)
    : pipelineLoop(loop), scopeOp(loop.getOperation()),
      latencyProvider(latencyProvider), counterClones(counterClones),
      defaultBuild(&defaultBuild), costModel(costModel) {}

MsDagBuilder::MsDagBuilder(scf::ForOp loop,
                           const LatencyProvider &latencyProvider,
                           const DenseMap<Value, Operation *> &counterClones)
    : pipelineLoop(loop), scopeOp(loop.getOperation()),
      latencyProvider(latencyProvider), counterClones(counterClones) {}

//===----------------------------------------------------------------------===//
// DOT dump helpers — Graphviz output for DAG1/DAG2/DAG3
//===----------------------------------------------------------------------===//

static bool isTransferOp(Operation *op);

static const char *msNodeKindStr(MsNodeKind kind) {
  switch (kind) {
  case MsNodeKind::CubeCompute:
    return "Cube";
  case MsNodeKind::VectorGroup:
    return "Vector";
  case MsNodeKind::Transfer:
    return "Transfer";
  }
  return "?";
}

static const char *msStorageStr(MsStorage s) {
  switch (s) {
  case MsStorage::GM:
    return "GM";
  case MsStorage::L1:
    return "L1";
  case MsStorage::L0A:
    return "L0A";
  case MsStorage::L0B:
    return "L0B";
  case MsStorage::L0C:
    return "L0C";
  case MsStorage::UB:
    return "UB";
  case MsStorage::Virtual:
    return "Virtual";
  }
  return "?";
}

/// Pick the op shown in DOT node labels.  WorkItem-based compute nodes list
/// address/index arithmetic first in block order, so primaryOp is usually a
/// scalar helper (e.g. arith.maxsi) rather than the representative compute op.
/// Prefer the first non-transfer op whose core type matches the node kind
/// (the mmad for CubeCompute, the first vector op for VectorGroup).
static Operation *pickDisplayOp(const MsNode &sn) {
  if (sn.kind == MsNodeKind::Transfer)
    return sn.primaryOp;
  Operation *hivmFallback = nullptr;
  for (Operation *op : sn.ops) {
    if (isTransferOp(op))
      continue;
    auto core = queryCoreTypeHelper(op);
    if (core &&
        ((sn.kind == MsNodeKind::CubeCompute && *core == TCoreType::CUBE) ||
         (sn.kind == MsNodeKind::VectorGroup && *core == TCoreType::VECTOR)))
      return op;
    if (!hivmFallback && op->getDialect() &&
        op->getDialect()->getNamespace() == HIVMDialect::getDialectNamespace())
      hivmFallback = op;
  }
  return hivmFallback ? hivmFallback : sn.primaryOp;
}

/// Render a shaped type as "128x128xf16"; empty string for non-shaped types.
static std::string formatShapedType(Type ty) {
  auto shapedTy = dyn_cast_if_present<ShapedType>(ty);
  if (!shapedTy)
    return {};
  std::string s;
  llvm::raw_string_ostream os(s);
  for (int64_t d : shapedTy.getShape())
    os << d << "x";
  os << shapedTy.getElementType();
  return s;
}

/// Shape shown for a compute node.  For Cube nodes with a mmad-like display
/// op, show "A:MxKxf16 B:KxNxf16->MxNxf32"; for others show the first shaped
/// result or operand.
static std::string displayShape(Operation *op, MsNodeKind kind) {
  if (!op)
    return {};
  if (kind == MsNodeKind::CubeCompute) {
    std::string s;
    llvm::raw_string_ostream os(s);
    int numOperands = static_cast<int>(op->getNumOperands());
    bool hasA = numOperands > 0;
    bool hasB = numOperands > 1;
    std::string aStr =
        hasA ? formatShapedType(op->getOperand(0).getType()) : "";
    std::string bStr =
        hasB ? formatShapedType(op->getOperand(1).getType()) : "";
    std::string rStr;
    if (op->getNumResults() > 0)
      rStr = formatShapedType(op->getResult(0).getType());
    if (aStr.empty() && bStr.empty() && rStr.empty())
      return {};
    if (!aStr.empty())
      os << "A:" << aStr;
    if (!bStr.empty()) {
      if (!aStr.empty())
        os << " ";
      os << "B:" << bStr;
    }
    if (!rStr.empty()) {
      if (!aStr.empty() || !bStr.empty())
        os << "->";
      os << rStr;
    }
    return s;
  }
  for (Value res : op->getResults()) {
    std::string s = formatShapedType(res.getType());
    if (!s.empty())
      return s;
  }
  for (Value operand : op->getOperands()) {
    std::string s = formatShapedType(operand.getType());
    if (!s.empty())
      return s;
  }
  return {};
}

/// Shape shown for a synthetic DAG2 transfer node, derived from the linked
/// compute nodes: cube-input chains report the consumer cube op's operand
/// slot; output-side transfers report the producer's result shape.
static std::string transferDisplayShape(const MsDagBuildResult &result,
                                        const MsNode &sn) {
  if (!sn.isSynthetic)
    return displayShape(sn.primaryOp, sn.kind);
  if (sn.targetComputeIdx >= 0 && sn.cubeOperandSlot >= 0) {
    const MsNode &tgt = result.nodes[sn.targetComputeIdx];
    for (Operation *op : tgt.ops) {
      if (isTransferOp(op))
        continue;
      auto core = queryCoreTypeHelper(op);
      if (!core || *core != TCoreType::CUBE)
        continue;
      if (sn.cubeOperandSlot < static_cast<int>(op->getNumOperands()))
        return formatShapedType(op->getOperand(sn.cubeOperandSlot).getType());
      break;
    }
    return {};
  }
  if (sn.sourceComputeIdx >= 0)
    return displayShape(pickDisplayOp(result.nodes[sn.sourceComputeIdx]),
                        result.nodes[sn.sourceComputeIdx].kind);
  if (sn.targetComputeIdx >= 0)
    return displayShape(pickDisplayOp(result.nodes[sn.targetComputeIdx]),
                        result.nodes[sn.targetComputeIdx].kind);
  return {};
}

/// Escape a string for use inside a DOT record label.  Record labels treat
/// '{', '}', '|', '<' and '>' as structure characters (e.g. the '>' in a
/// "f16->f32" shape summary), and the label itself is a double-quoted DOT
/// string, so '"' and '\' must be escaped as well.
static std::string escapeRecordLabel(llvm::StringRef text) {
  std::string s;
  s.reserve(text.size());
  for (char c : text) {
    if (c == '{' || c == '}' || c == '|' || c == '<' || c == '>' || c == '"' ||
        c == '\\')
      s += '\\';
    s += c;
  }
  return s;
}

/// Emit a DOT-format representation of the DAG after DAG1 (compute nodes +
/// inter-compute edges only, no synthetic transfers or data nodes).
static void dumpDag1Dot(llvm::raw_ostream &os, const MsDagBuildResult &result) {
  os << "digraph DAG1 {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=record, fontname=\"Courier\"];\n\n";

  for (const MsNode &sn : result.nodes) {
    os << "  n" << sn.ddgIndex << " [label=\"{N" << sn.ddgIndex << " | "
       << msNodeKindStr(sn.kind) << " | " << escapeRecordLabel(sn.pipeline);
    if (Operation *dispOp = pickDisplayOp(sn)) {
      os << " | " << escapeRecordLabel(dispOp->getName().getStringRef());
      if (std::string shape = displayShape(dispOp, sn.kind); !shape.empty())
        os << " | " << escapeRecordLabel(shape);
    }
    if (sn.workItemIndex >= 0)
      os << " | WI" << sn.workItemIndex;
    const auto &ddgNode = result.ddg.getNodes()[sn.ddgIndex];
    os << " | lat=" << ddgNode.latency;
    os << "}\"";
    if (sn.kind == MsNodeKind::CubeCompute)
      os << ", style=filled, fillcolor=\"#BDE0FE\"";
    else if (sn.kind == MsNodeKind::VectorGroup)
      os << ", style=filled, fillcolor=\"#C8E6C9\"";
    os << "];\n";
  }

  os << "\n";
  for (const sms::DDGEdge &e : result.ddg.getEdges()) {
    os << "  n" << e.src << " -> n" << e.dst << " [label=\"" << e.latency
       << "\"];\n";
  }

  os << "}\n";
}

/// Emit a DOT-format representation of the DAG after DAG2 (compute + synthetic
/// transfer nodes and all edges).
static void dumpDag2Dot(llvm::raw_ostream &os, const MsDagBuildResult &result) {
  os << "digraph DAG2 {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=record, fontname=\"Courier\"];\n\n";

  for (const MsNode &sn : result.nodes) {
    os << "  n" << sn.ddgIndex << " [label=\"{N" << sn.ddgIndex << " | "
       << msNodeKindStr(sn.kind) << " | " << escapeRecordLabel(sn.pipeline);
    if (!sn.isSynthetic) {
      if (Operation *dispOp = pickDisplayOp(sn))
        os << " | " << escapeRecordLabel(dispOp->getName().getStringRef());
    } else {
      os << " | " << escapeRecordLabel(sn.storageTag);
    }
    if (std::string shape = sn.kind == MsNodeKind::Transfer
                                ? transferDisplayShape(result, sn)
                                : displayShape(pickDisplayOp(sn), sn.kind);
        !shape.empty())
      os << " | " << escapeRecordLabel(shape);
    if (sn.workItemIndex >= 0)
      os << " | WI" << sn.workItemIndex;
    const auto &ddgNode = result.ddg.getNodes()[sn.ddgIndex];
    os << " | lat=" << ddgNode.latency;
    os << "}\"";
    if (sn.kind == MsNodeKind::CubeCompute)
      os << ", style=filled, fillcolor=\"#BDE0FE\"";
    else if (sn.kind == MsNodeKind::VectorGroup)
      os << ", style=filled, fillcolor=\"#C8E6C9\"";
    else if (sn.isSynthetic)
      os << ", style=filled, fillcolor=\"#FFF9C4\"";
    else
      os << ", style=filled, fillcolor=\"#FFE0B2\"";
    os << "];\n";
  }

  os << "\n";
  for (const sms::DDGEdge &e : result.ddg.getEdges()) {
    os << "  n" << e.src << " -> n" << e.dst << " [label=\"" << e.latency
       << "\"];\n";
  }

  os << "}\n";
}

/// Emit a DOT-format representation of the full DAG3 (DAG2 schedulable graph +
/// data nodes shown as ellipses with storage/shape/byte metadata).
static void dumpDag3Dot(llvm::raw_ostream &os, const MsDagBuildResult &result) {
  os << "digraph DAG3 {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=record, fontname=\"Courier\"];\n\n";

  // Schedulable nodes (same as DAG2).
  for (const MsNode &sn : result.nodes) {
    os << "  n" << sn.ddgIndex << " [label=\"{N" << sn.ddgIndex << " | "
       << msNodeKindStr(sn.kind) << " | " << escapeRecordLabel(sn.pipeline);
    if (!sn.isSynthetic) {
      if (Operation *dispOp = pickDisplayOp(sn))
        os << " | " << escapeRecordLabel(dispOp->getName().getStringRef());
    } else {
      os << " | " << escapeRecordLabel(sn.storageTag);
    }
    if (std::string shape = sn.kind == MsNodeKind::Transfer
                                ? transferDisplayShape(result, sn)
                                : displayShape(pickDisplayOp(sn), sn.kind);
        !shape.empty())
      os << " | " << escapeRecordLabel(shape);
    const auto &ddgNode = result.ddg.getNodes()[sn.ddgIndex];
    os << " | lat=" << ddgNode.latency;
    os << "}\"";
    if (sn.kind == MsNodeKind::CubeCompute)
      os << ", style=filled, fillcolor=\"#BDE0FE\"";
    else if (sn.kind == MsNodeKind::VectorGroup)
      os << ", style=filled, fillcolor=\"#C8E6C9\"";
    else if (sn.isSynthetic)
      os << ", style=filled, fillcolor=\"#FFF9C4\"";
    else
      os << ", style=filled, fillcolor=\"#FFE0B2\"";
    os << "];\n";
  }

  // Data nodes (non-schedulable).
  for (const MsDataNode &dn : result.dataNodes) {
    os << "  d" << dn.id << " [shape=ellipse, style=filled"
       << ", fillcolor=\"#F3E5F5\""
       << ", label=\"D" << dn.id << "\\n"
       << msStorageStr(dn.storage) << "\\n"
       << dn.totalBytes << "B";
    if (!dn.shape.empty()) {
      os << "\\n[";
      for (size_t i = 0; i < dn.shape.size(); ++i) {
        if (i)
          os << "x";
        os << dn.shape[i];
      }
      os << "]";
    }
    os << "\"];\n";
  }

  // Schedulable edges.
  os << "\n";
  for (const sms::DDGEdge &e : result.ddg.getEdges()) {
    os << "  n" << e.src << " -> n" << e.dst << " [label=\"" << e.latency
       << "\"];\n";
  }

  // Data-node edges (producer → data → consumers).
  for (const MsDataNode &dn : result.dataNodes) {
    if (dn.producerIdx >= 0)
      os << "  n" << dn.producerIdx << " -> d" << dn.id
         << " [style=dashed, color=\"#9C27B0\"];\n";
    for (int ci : dn.consumerIdxs)
      os << "  d" << dn.id << " -> n" << ci
         << " [style=dashed, color=\"#9C27B0\"];\n";
  }

  os << "}\n";
}

//===----------------------------------------------------------------------===//
// DAG1 analysis — WorkItem-based node creation
//===----------------------------------------------------------------------===//

/// Determine whether `op` is a transfer op that should get its own DDG node
/// (separate from any WorkItem compute node).
static bool isTransferOp(Operation *op) {
  if (isa<LoadOp, ND2NZOp, StoreOp, FixpipeOp>(op))
    return true;
  // Cross-core copy: CopyOp whose destination traces to an L1 alloc.
  auto copyOp = dyn_cast<CopyOp>(op);
  if (!copyOp)
    return false;
  Value dst = copyOp.getDst();
  // Trace through views/casts to the root alloc.
  while (auto result = dyn_cast<OpResult>(dst)) {
    Operation *def = result.getOwner();
    Value next =
        TypeSwitch<Operation *, Value>(def)
            .Case<memref::CastOp, memref::ReinterpretCastOp,
                  memref::MemorySpaceCastOp, memref::CollapseShapeOp,
                  memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp>(
                [](auto castOp) { return castOp->getOperand(0); })
            .Default([](Operation *) { return nullptr; });
    if (!next)
      break;
    dst = next;
  }
  auto alloc = dyn_cast_if_present<memref::AllocOp>(dst.getDefiningOp());
  if (!alloc)
    return false;
  auto memSpaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(alloc.getType().getMemorySpace());
  return memSpaceAttr && memSpaceAttr.getAddressSpace() == AddressSpace::L1;
}

/// Extract shape and element type from a shaped Value.  Returns false if the
/// value is not a shaped type (scalar, index, etc.).
static bool extractShapeAndType(Value v, SmallVector<int64_t> &shape,
                                Type &elemType) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy)
    return false;
  shape.assign(shapedTy.getShape().begin(), shapedTy.getShape().end());
  elemType = shapedTy.getElementType();
  return true;
}

LogicalResult MsDagBuilder::buildNodesFromWorkItems() {
  assert(defaultBuild && "buildNodesFromWorkItems requires Default WorkItems");

  int ddgIdx = 0;

  // Snapshot the Default worklist and ownership map into the result.
  result.defaultWorklist = defaultBuild->worklist;
  result.defaultOpToWorkItemMap = defaultBuild->opToWorkItemMap;
  result.outputMemrefMap = defaultBuild->outputMemrefMap;

  // --- Step 1: Create one compute DDG node per Default WorkItem. ---
  for (int wiIdx = 0, wiEnd = static_cast<int>(defaultBuild->worklist.size());
       wiIdx < wiEnd; ++wiIdx) {
    WorkItem *wi = defaultBuild->worklist[wiIdx].get();

    // Separate the WorkItem's ops into compute ops and transfer ops.
    SmallVector<Operation *> coreOps;
    SmallVector<Operation *> wiTransferOps;
    for (Operation *op : wi->ops) {
      if (isTransferOp(op))
        wiTransferOps.push_back(op);
      else
        coreOps.push_back(op);
    }

    // Skip WorkItems with no core ops (shouldn't happen, but be safe).
    if (coreOps.empty() && wiTransferOps.empty())
      continue;

    // The compute node represents the WorkItem's compute aspect.
    // Even if there are only transfer ops (e.g., a WorkItem that is all
    // loads), we still create a compute node to maintain 1:1 mapping.
    MsNodeKind kind;
    std::string pipeline;
    if (wi->core == TCoreType::CUBE) {
      kind = MsNodeKind::CubeCompute;
      pipeline = "cube";
    } else {
      kind = MsNodeKind::VectorGroup;
      pipeline = "vector";
    }

    // Compute latency as max over core ops in the WorkItem.
    int wiLatency = 1;
    int wiIssue = 1;
    for (Operation *op : coreOps) {
      int lat = latencyProvider.getLatency(op);
      int iss = latencyProvider.getIssueTime(op);
      if (lat <= 0 || iss <= 0) {
        return op->emitWarning(
            "[ms-dag-builder] non-positive latency/issue time for core op");
      }
      wiLatency = std::max(wiLatency, lat);
      wiIssue = std::max(wiIssue, iss);
    }

    MsNode sn;
    sn.ddgIndex = ddgIdx;
    sn.kind = kind;
    sn.primaryOp = coreOps.empty() ? wiTransferOps.front() : coreOps.front();
    // Include ALL ops in the WorkItem — both core and transfer —  so the
    // downstream partitioner can find them via the compute node's ops list.
    for (Operation *op : wi->ops)
      sn.ops.push_back(op);
    sn.pipeline = pipeline;
    sn.isParallelUnit = false;
    sn.workItem = wi;
    sn.workItemIndex = wiIdx;
    result.nodes.push_back(sn);

    // Map all ops in the WorkItem to this compute DDG index.
    // Multi-owner ops (Default lazy-load) keep the first (earliest-worklist)
    // mapping to avoid reverse edges from later WorkItems.
    for (Operation *op : wi->ops)
      result.opToDdgIndex.try_emplace(op, ddgIdx);
    result.workItemToDdgIndex[wi] = ddgIdx;

    sms::OpNode ddgNode;
    ddgNode.ordinal = ddgIdx;
    ddgNode.name = (llvm::Twine("wi") + llvm::Twine(wiIdx) + "_" +
                    (wi->core == TCoreType::CUBE ? "cube" : "vector"))
                       .str();
    ddgNode.pipeline = pipeline;
    ddgNode.latency = wiLatency;
    ddgNode.issueTime = wiIssue;
    ddgNode.isParallelUnit = false;
    result.ddg.addNode(ddgNode);

    ++ddgIdx;
  }

  // IR transfer ops (loads/fixpipes/stores that are WorkItem members) stay
  // inside their WorkItem's compute node — DAG2 models the transfer timing
  // with synthetic nodes instead (buildTransferNodes).

  LLVM_DEBUG({
    dbgs() << "[ms-dag-builder] DAG1 WorkItem nodes:\n"
           << "  compute nodes: " << ddgIdx << " (one per WorkItem)\n";
  });

  return success();
}

LogicalResult MsDagBuilder::buildWorkItemDependencies() {
  assert(defaultBuild &&
         "buildWorkItemDependencies requires Default WorkItems");

  // For each compute node, trace its ops' operands to find producer nodes.
  // Two compute nodes have a dependency if any op in the consumer node uses
  // a Value produced by an op owned by the producer node.
  // IMPORTANT: Only trace DPS *input* operands — DPS output/init operands
  // (e.g. tensor.empty used as outs) are storage destinations, not data
  // dependencies.  Including them creates false reverse edges when the same
  // tensor.empty is shared across WorkItems (multi-owner).
  for (const MsNode &consNode : result.nodes) {
    DenseSet<int> preds;
    for (Operation *op : consNode.ops) {
      // Determine operand range: for DPS ops, only iterate DPS inputs.
      auto dps = dyn_cast<DestinationStyleOpInterface>(op);
      int numInputOperands =
          dps ? dps.getNumDpsInputs() : static_cast<int>(op->getNumOperands());
      for (int oi = 0; oi < numInputOperands; ++oi) {
        Value operand = op->getOperand(oi);
        // Walk through view/cast/to_tensor to find the root producer.
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        Operation *scopedDef = getContainedParent(scopeOp, *defOp);
        if (!scopedDef)
          continue;
        auto it = result.opToDdgIndex.find(scopedDef);
        if (it == result.opToDdgIndex.end())
          continue;
        int predIdx = it->second;
        if (predIdx == consNode.ddgIndex)
          continue; // Internal dependency — same node.
        if (preds.insert(predIdx).second)
          result.ddg.addEdge(predIdx, consNode.ddgIndex);
      }
    }
  }

  // Separator dependency edges (as in the legacy path).
  // Separators are ops like FixpipeOp, StoreOp, cross-core CopyOp.
  // Walk each separator's shaped results forward to find downstream ops.
  SmallVector<Operation *> localSeparators;
  for (Operation &op : *pipelineLoop.getBody()) {
    if (isSeparator(&op))
      localSeparators.push_back(&op);
  }

  for (Operation *separator : localSeparators) {
    auto sepIt = result.opToDdgIndex.find(separator);
    if (sepIt == result.opToDdgIndex.end())
      continue;
    int sepIdx = sepIt->second;

    SmallVector<Operation *> dfsStack = {separator};
    DenseSet<Operation *> visited;
    while (!dfsStack.empty()) {
      Operation *op = dfsStack.pop_back_val();
      if (visited.contains(op) || !scopeOp->isAncestor(op))
        continue;
      visited.insert(op);
      for (Value resultVal : op->getResults()) {
        if (!isa<ShapedType>(resultVal.getType()))
          continue;
        for (OpOperand &use : resultVal.getUses()) {
          Operation *usr = use.getOwner();
          Operation *scopedUsr = getContainedParent(scopeOp, *usr);
          if (!scopedUsr)
            continue;
          if (isa<scf::YieldOp, scf::ConditionOp>(scopedUsr) ||
              scopedUsr == separator)
            continue;
          dfsStack.push_back(scopedUsr);

          auto depIt = result.opToDdgIndex.find(scopedUsr);
          if (depIt == result.opToDdgIndex.end())
            continue;
          int depIdx = depIt->second;
          if (depIdx == sepIdx)
            continue;
          // Add edge if not already present.
          bool exists = false;
          for (const sms::DDGEdge &e : result.ddg.getOutEdges()[sepIdx]) {
            if (e.dst == depIdx) {
              exists = true;
              break;
            }
          }
          if (!exists)
            result.ddg.addEdge(sepIdx, depIdx);
        }
      }
    }
  }

  // --- Cross-WorkItem loop-carried recurrence detection ---
  // Same-WorkItem recurrences are legal: they never enter the DDG (iter_args
  // have no defining op inside the loop) and are preserved by the existing
  // CVPipelining iter_arg/yield codegen.  A recurrence whose producer and
  // consumer live in DIFFERENT WorkItems cannot be honored by the SMS
  // schedule (no RecMII support) — reject so the caller falls back to the
  // Default scheduler.
  auto ownersOf = [&](Operation *op) -> ArrayRef<WorkItem *> {
    auto it = defaultBuild->opToWorkItemMap.find(op);
    if (it == defaultBuild->opToWorkItemMap.end())
      return {};
    return it->second;
  };

  auto iterArgs = pipelineLoop.getRegionIterArgs();
  auto yieldedVals = pipelineLoop.getYieldedValues();
  for (unsigned i = 0, e = iterArgs.size(); i < e; ++i) {
    Operation *prodOp = yieldedVals[i].getDefiningOp();
    if (!prodOp)
      continue;
    Operation *prodTop = getContainedParent(scopeOp, *prodOp);
    if (!prodTop)
      continue;
    ArrayRef<WorkItem *> prodOwners = ownersOf(prodTop);
    if (prodOwners.empty())
      continue;

    for (Operation *usr : iterArgs[i].getUsers()) {
      if (isa<scf::YieldOp>(usr))
        continue;
      Operation *usrTop = getContainedParent(scopeOp, *usr);
      if (!usrTop)
        continue;
      ArrayRef<WorkItem *> consOwners = ownersOf(usrTop);
      if (consOwners.empty())
        continue;

      bool sharesOwner = llvm::any_of(prodOwners, [&](WorkItem *pw) {
        return llvm::is_contained(consOwners, pw);
      });
      if (!sharesOwner) {
        result.hasLoopCarriedDeps = true;
        result.loopCarriedDiag =
            ("cross-WorkItem loop-carried recurrence on iter_arg #" +
             llvm::Twine(i))
                .str();
        return pipelineLoop->emitWarning(
                   "[ms-dag-builder] cross-WorkItem loop-carried recurrence "
                   "on iter_arg #")
               << i << "; SMS cannot schedule it (no RecMII) — falling back";
      }
    }
  }

  LLVM_DEBUG({
    dbgs() << "[ms-dag-builder] DAG1 edges: " << result.ddg.getEdges().size()
           << "\n";
    for (const sms::DDGEdge &e : result.ddg.getEdges())
      dbgs() << "  " << e.src << " -> " << e.dst << "\n";
    dbgs() << "[ms-dag-builder] DAG1 DOT:\n";
    dumpDag1Dot(dbgs(), result);
  });

  return success();
}

void MsDagBuilder::collectOperandMetadata() {
  assert(defaultBuild && "collectOperandMetadata requires Default WorkItems");

  for (MsNode &sn : result.nodes) {
    if (sn.kind == MsNodeKind::Transfer)
      continue;

    // For CUBE nodes, record the two input operands (A and B) of the primary
    // cube op and the output.
    if (sn.kind == MsNodeKind::CubeCompute) {
      for (Operation *op : sn.ops) {
        // Transfer ops (nd2nz/load/fixpipe) may also report a CUBE core type
        // and typically precede the mmad in block order — skip them so the
        // recorded slots belong to the compute op, not the transfer.
        if (isTransferOp(op))
          continue;
        auto maybeCore = queryCoreTypeHelper(op);
        if (!maybeCore || *maybeCore != TCoreType::CUBE)
          continue;
        // HIVM core ops are DPS: getOperands() spans both the ins() and the
        // outs() region.  An outs buffer is the op's output destination, not
        // an input — recording it as one would pollute the slot metadata
        // (buildCubeCostInput reads slots 0/1 and is unaffected today, but
        // the recorded semantics must stay correct for future consumers).
        auto dps = dyn_cast<DestinationStyleOpInterface>(op);
        // Record shaped input operands.
        for (int i = 0, e = static_cast<int>(op->getNumOperands()); i < e;
             ++i) {
          if (dps && dps.isDpsInit(&op->getOpOperand(i)))
            continue;
          MsNode::OperandInfo info;
          if (extractShapeAndType(op->getOperand(i), info.shape,
                                  info.elementType)) {
            info.isInput = true;
            info.operandIndex = i;
            sn.operandInfos.push_back(std::move(info));
          }
        }
        // Record shaped results.
        for (Value res : op->getResults()) {
          MsNode::OperandInfo info;
          if (extractShapeAndType(res, info.shape, info.elementType)) {
            info.isInput = false;
            info.operandIndex = -1;
            sn.operandInfos.push_back(std::move(info));
          }
        }
        break; // Only the first cube op (primary).
      }
      continue;
    }

    // For VECTOR nodes, record every core op's operand and result.  Transfer
    // ops (load/nd2nz/...) are HIVM DPS ops too — skip them for the same
    // reason as in the cube branch: their cost is billed by the DAG2
    // transfer nodes, and letting them claim input slots here would corrupt
    // the per-op grouping (ownerOp) the cost model relies on.
    assert(sn.kind == MsNodeKind::VectorGroup);
    for (Operation *op : sn.ops) {
      if (isTransferOp(op))
        continue;
      if (!isCoreOp(*op))
        continue;
      // Same DPS rule as the cube branch: an outs() buffer is an output
      // destination, not an input.  The MS cost model bills a vector op by
      // its input element count (§4.4.2: outputs do not participate), so an
      // outs recorded as input inflates x by the buffer size — e.g. a f32
      // [128x128] DPS op counted ~1.7x its real input volume.
      auto dps = dyn_cast<DestinationStyleOpInterface>(op);
      // vcast and other ExtraBufferOpInterface ops carry a trailing
      // temp_buffer operand — scratch space, not a real input.  The v2
      // multiplier chain counts real inputs (n_input) and takes the dtype
      // from the FIRST real input, so a recorded temp buffer would skew
      // both.  Membership test per operand (getExtraBuffers() is cheap:
      // it returns an OperandRange over the temp_buffer operands).
      auto extraBuffer = dyn_cast<hivm::ExtraBufferOpInterface>(op);
      auto isExtraBuffer = [&extraBuffer](Value v) {
        return extraBuffer &&
               llvm::is_contained(extraBuffer.getExtraBuffers(), v);
      };
      for (int i = 0, e = static_cast<int>(op->getNumOperands()); i < e; ++i) {
        if (dps && dps.isDpsInit(&op->getOpOperand(i)))
          continue;
        if (isExtraBuffer(op->getOperand(i)))
          continue;
        MsNode::OperandInfo info;
        if (extractShapeAndType(op->getOperand(i), info.shape,
                                info.elementType)) {
          info.isInput = true;
          info.operandIndex = i;
          info.ownerOp = op;
          sn.operandInfos.push_back(std::move(info));
        }
      }
      for (Value res : op->getResults()) {
        MsNode::OperandInfo info;
        if (extractShapeAndType(res, info.shape, info.elementType)) {
          info.isInput = false;
          info.operandIndex = -1;
          info.ownerOp = op;
          sn.operandInfos.push_back(std::move(info));
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase DAG2 — synthetic transfer nodes (CVpie rules)
//===----------------------------------------------------------------------===//

LogicalResult MsDagBuilder::buildTransferNodes() {
  assert(defaultBuild && "buildTransferNodes requires Default WorkItems");

  // Snapshot the compute-only DDG size — transfer nodes start after this.
  const int numComputeNodes = result.ddg.size();
  result.computeNodeCount = numComputeNodes;

  // Determine which compute nodes have compute successors so we can identify
  // sinks (write to GM).
  DenseSet<int> hasComputeSucc;
  for (const sms::DDGEdge &edge : result.ddg.getEdges())
    hasComputeSucc.insert(edge.src);

  int ddgIdx = numComputeNodes;

  // Record compute-level edges that acquire an explicit transfer chain.  Such
  // edges are removed after all chains have been built; otherwise the graph
  // models the same data dependence both directly and through the transfer.
  SmallVector<std::pair<int, int>> replacedComputeEdges;
  auto markReplaced = [&](int src, int dst) {
    if (!llvm::is_contained(replacedComputeEdges, std::make_pair(src, dst)))
      replacedComputeEdges.emplace_back(src, dst);
  };

  // Helper: create a synthetic transfer DDG node and return its DDG index.
  auto addTransfer = [&](StringRef pipeline, StringRef tag, int srcCompute,
                         int dstCompute, int cubeSlot) -> int {
    int thisIdx = ddgIdx++;

    MsNode sn;
    sn.ddgIndex = thisIdx;
    sn.kind = MsNodeKind::Transfer;
    sn.primaryOp = nullptr;
    sn.pipeline = pipeline.str();
    sn.isParallelUnit = true;
    sn.isSynthetic = true;
    sn.storageTag = tag.str();
    sn.sourceComputeIdx = srcCompute;
    sn.targetComputeIdx = dstCompute;
    sn.cubeOperandSlot = cubeSlot;
    result.nodes.push_back(sn);

    sms::OpNode ddgNode;
    ddgNode.ordinal = thisIdx;
    ddgNode.name = (llvm::Twine("xfer_") + pipeline + "_" + tag).str();
    ddgNode.pipeline = pipeline.str();
    ddgNode.latency = 1;
    ddgNode.issueTime = 1;
    ddgNode.isParallelUnit = true;
    result.ddg.addNode(ddgNode);

    return thisIdx;
  };

  // Helper: create a two-stage cube input chain (e.g. mte2→mte1 or
  // mte3→mte1 or fixpipe→mte1).
  auto addCubeInputChain = [&](StringRef firstPipeline, StringRef firstTag,
                               StringRef l0Tag, int predCompute,
                               int consCompute, int cubeSlot) {
    int first = addTransfer(firstPipeline, firstTag, predCompute, consCompute,
                            cubeSlot);
    int mte1 = addTransfer("mte1", l0Tag, predCompute, consCompute, cubeSlot);
    if (predCompute >= 0)
      result.ddg.addEdge(predCompute, first);
    result.ddg.addEdge(first, mte1);
    result.ddg.addEdge(mte1, consCompute);
    if (predCompute >= 0)
      markReplaced(predCompute, consCompute);
  };

  // --- Process each consumer compute node ---
  // Iterate by index; copy node fields before any addTransfer call, since
  // pushing to result.nodes invalidates references.
  for (int ci = 0; ci < numComputeNodes; ++ci) {
    MsNodeKind consumerKind = result.nodes[ci].kind;
    bool isCube = (consumerKind == MsNodeKind::CubeCompute);
    bool isVector = (consumerKind == MsNodeKind::VectorGroup);
    if (!isCube && !isVector)
      continue;

    // Collect compute predecessors of this node from existing DDG edges.
    SmallVector<int> cubePreds, vectorPreds;
    for (const sms::DDGEdge &e : result.ddg.getInEdges()[ci]) {
      if (e.src >= numComputeNodes)
        continue; // Synthetic edges — not compute predecessors.
      MsNodeKind predKind = result.nodes[e.src].kind;
      if (predKind == MsNodeKind::CubeCompute)
        cubePreds.push_back(e.src);
      else if (predKind == MsNodeKind::VectorGroup)
        vectorPreds.push_back(e.src);
    }
    int totalPreds = static_cast<int>(cubePreds.size() + vectorPreds.size());

    if (isCube) {
      // Locate the primary cube op for per-operand-slot source tracing
      // (cf. CVpie per_input_transfers / vector_to_cube_mixed rules).
      // Transfer ops (nd2nz/load/fixpipe) may also report a CUBE core type
      // and typically precede the mmad in block order — skip them, otherwise
      // the slots traced below belong to the transfer, not the compute op.
      Operation *cubeOp = nullptr;
      for (Operation *op : result.nodes[ci].ops) {
        if (isTransferOp(op))
          continue;
        auto maybeCore = queryCoreTypeHelper(op);
        if (maybeCore && *maybeCore == TCoreType::CUBE) {
          cubeOp = op;
          break;
        }
      }

      // Resolve the producing compute node of a cube input slot; -1 means
      // the input is external (GM).
      auto slotSource = [&](int slot) -> int {
        if (!cubeOp || slot >= static_cast<int>(cubeOp->getNumOperands()))
          return -1;
        Value root = traceValueDef(cubeOp->getOperand(slot));
        if (!root)
          return -1;

        // A tensor backed by an explicitly allocated UB/CBUF buffer is often
        // represented as `to_tensor(alloc)` rather than as an OpResult of the
        // producer.  In that form traceValueDef intentionally ends at the
        // alloc, so also walk users of the root buffer to find the mapped
        // transfer/copy operation that populated it.
        auto mappedOwner = [&](Operation *op) -> int {
          if (!op)
            return -1;

          // WorkItem ownership is recorded for the concrete operation, not
          // necessarily for its containing scope.  Check the operation before
          // climbing to the loop-level contained parent.
          auto directIt = result.opToDdgIndex.find(op);
          if (directIt != result.opToDdgIndex.end() && directIt->second != ci &&
              directIt->second < numComputeNodes)
            return directIt->second;

          Operation *scoped = getContainedParent(scopeOp, *op);
          if (!scoped)
            return -1;
          auto it = result.opToDdgIndex.find(scoped);
          if (it == result.opToDdgIndex.end() || it->second == ci ||
              it->second >= numComputeNodes)
            return -1;
          return it->second;
        };

        // If this WorkItem itself contains a load-like op (nd2nz/load) that
        // writes into the same root buffer, the slot's data comes from GM,
        // not from another compute node.  Check this BEFORE the cross-node
        // DPS init scan, which matches purely on buffer identity and would
        // otherwise confuse a foreign fixpipe writing into the same L1 alloc
        // with the actual GM source.
        for (Operation *selfOp : result.nodes[ci].ops) {
          if (!isLoadLikeOp(selfOp))
            continue;
          auto selfDps = dyn_cast<DestinationStyleOpInterface>(selfOp);
          if (!selfDps)
            continue;
          for (unsigned ii = 0, ie = selfDps.getNumDpsInits(); ii < ie; ++ii) {
            if (traceValueDef(selfDps.getDpsInitOperand(ii)->get()) == root)
              return -1;
          }
        }

        // The producer of a tensor backed by an allocated buffer is the DPS
        // writer whose init operand aliases that buffer.  Resolve this from
        // the authoritative WorkItem op lists instead of relying on the
        // alloc's use walk: the writer may be hidden behind a scope boundary
        // or a memref view, while the WorkItem still records it exactly.
        for (int producerIdx = 0; producerIdx < numComputeNodes;
             ++producerIdx) {
          if (producerIdx == ci)
            continue;
          for (Operation *producerOp : result.nodes[producerIdx].ops) {
            auto dps = dyn_cast<DestinationStyleOpInterface>(producerOp);
            if (!dps)
              continue;
            for (unsigned initIdx = 0, initEnd = dps.getNumDpsInits();
                 initIdx < initEnd; ++initIdx) {
              OpOperand *init = dps.getDpsInitOperand(initIdx);
              if (traceValueDef(init->get()) == root)
                return producerIdx;
            }
          }
        }

        if (int direct = mappedOwner(root.getDefiningOp()); direct >= 0)
          return direct;

        SmallVector<Value, 4> values{root};
        DenseSet<Value> visitedValues;
        DenseSet<Operation *> visitedOps;
        while (!values.empty()) {
          Value value = values.pop_back_val();
          if (!visitedValues.insert(value).second)
            continue;
          for (Operation *user : value.getUsers()) {
            if (!visitedOps.insert(user).second)
              continue;
            if (int owner = mappedOwner(user); owner >= 0)
              return owner;
            // Only an alias/view chain (or a transfer) may lead to the DPS
            // writer of the root buffer.  Do not walk through a compute op:
            // its consumers (e.g. fixpipe -> vexp after mmad) live in other
            // WorkItems and would be mistaken for the input's producer,
            // creating a cycle (verifyDag: DDG contains a cycle).
            if (!isa<ViewLikeOpInterface>(user) && !isTransferOp(user))
              continue;
            for (Value result : user->getResults())
              values.push_back(result);
          }
        }
        return -1;
      };

      for (int slot = 0; slot < 2; ++slot) {
        int pred = slotSource(slot);
        StringRef l0Tag =
            (slot == 0) ? StringRef("L1_to_L0A") : StringRef("L1_to_L0B");
        if (pred < 0) {
          // External input: mte2(GM_to_L1) → mte1(L1_to_L0A/B).
          addCubeInputChain("mte2", "GM_to_L1", l0Tag, /*predCompute=*/-1, ci,
                            slot);
        } else if (result.nodes[pred].kind == MsNodeKind::CubeCompute) {
          // cube → cube: fixpipe(L0C_to_L1) → mte1(L1_to_L0A/B).
          addCubeInputChain("fixpipe", "L0C_to_L1", l0Tag, pred, ci, slot);
        } else {
          // vector → cube: mte3(UB_to_L1) → mte1(L1_to_L0A/B).
          addCubeInputChain("mte3", "UB_to_L1", l0Tag, pred, ci, slot);
        }
      }
    } else {
      // Vector consumer.
      if (totalPreds == 0) {
        // GM → vector: mte2(GM_to_UB)
        int mte2 = addTransfer("mte2", "GM_to_UB", /*srcCompute=*/-1, ci, -1);
        result.ddg.addEdge(mte2, ci);
      } else {
        for (int pred : cubePreds) {
          // cube → vector: fixpipe(L0C_to_UB)
          int fix =
              addTransfer("fixpipe", "L0C_to_UB", pred, ci, /*cubeSlot=*/-1);
          result.ddg.addEdge(pred, fix);
          result.ddg.addEdge(fix, ci);
          markReplaced(pred, ci);
        }
        // vector → vector: no transfer needed.
      }
    }
  }

  // --- Sink transfers: compute nodes with no compute successors ---
  for (int ci = 0; ci < numComputeNodes; ++ci) {
    if (hasComputeSucc.contains(ci))
      continue;
    MsNodeKind producerKind = result.nodes[ci].kind;
    if (producerKind == MsNodeKind::CubeCompute) {
      // cube sink: fixpipe(L0C_to_GM)
      int fix = addTransfer("fixpipe", "L0C_to_GM", ci, /*dstCompute=*/-1, -1);
      result.ddg.addEdge(ci, fix);
    } else if (producerKind == MsNodeKind::VectorGroup) {
      // vector sink: mte3(UB_to_GM)
      int mte3 = addTransfer("mte3", "UB_to_GM", ci, /*dstCompute=*/-1, -1);
      result.ddg.addEdge(ci, mte3);
    }
  }

  // Rebuild adjacency after transfer insertion. Keep vector→vector edges
  // (there is no CVpie transfer for that path), but replace every compute edge
  // for which a legal transfer chain was emitted. DDG has no edge-removal API,
  // so snapshot all edges before clearing and restoring the survivors.
  SmallVector<sms::DDGEdge> dag2Edges(result.ddg.getEdges().begin(),
                                      result.ddg.getEdges().end());
  result.ddg.clearEdges();
  for (const sms::DDGEdge &edge : dag2Edges) {
    if (edge.src < numComputeNodes && edge.dst < numComputeNodes &&
        llvm::is_contained(replacedComputeEdges,
                           std::make_pair(edge.src, edge.dst)))
      continue;
    result.ddg.addEdge(edge.src, edge.dst);
  }

  LLVM_DEBUG({
    dbgs() << "[ms-dag-builder] DAG2 transfer nodes:\n"
           << "  compute nodes: " << numComputeNodes << "\n"
           << "  transfer nodes: " << (ddgIdx - numComputeNodes) << "\n"
           << "  total DDG nodes: " << result.ddg.size() << "\n"
           << "  total DDG edges: " << result.ddg.getEdges().size() << "\n";
    dbgs() << "[ms-dag-builder] DAG2 DOT:\n";
    dumpDag2Dot(dbgs(), result);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Phase DAG3 — data nodes (non-schedulable metadata)
//===----------------------------------------------------------------------===//

/// Map a storage tag string (from DAG2) to the MsStorage enum.
static MsStorage parseStorageTag(StringRef tag) {
  if (tag.contains("L0A"))
    return MsStorage::L0A;
  if (tag.contains("L0B"))
    return MsStorage::L0B;
  if (tag.contains("L0C"))
    return MsStorage::L0C;
  if (tag.contains("UB"))
    return MsStorage::UB;
  if (tag.contains("L1"))
    return MsStorage::L1;
  if (tag.contains("GM"))
    return MsStorage::GM;
  return MsStorage::GM;
}

/// Determine the output storage of a DAG2 transfer node from its storageTag.
/// The tag format is "SRC_to_DST"; the output storage is the DST part.
static MsStorage transferOutputStorage(const MsNode &xfer) {
  StringRef tag = xfer.storageTag;
  size_t pos = tag.find("_to_");
  if (pos == StringRef::npos)
    return MsStorage::GM;
  return parseStorageTag(tag.substr(pos + 4));
}

/// Compute element byte width from an MLIR element type.
static int64_t getElementBytes(Type elemTy) {
  if (!elemTy)
    return 0;
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return (intTy.getWidth() + 7) / 8;
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return floatTy.getWidth() / 8;
  return 0;
}

LogicalResult MsDagBuilder::buildDataNodes() {
  assert(defaultBuild && "buildDataNodes requires Default WorkItems");

  int dataId = 0;

  // Create a data node for each synthetic transfer node's output.
  // The data node describes what the transfer produces at its target storage.
  for (int ti = result.computeNodeCount;
       ti < static_cast<int>(result.nodes.size()); ++ti) {
    const MsNode &xfer = result.nodes[ti];
    if (!xfer.isSynthetic)
      continue;

    MsDataNode dn;
    dn.id = dataId++;
    dn.storage = transferOutputStorage(xfer);
    dn.producerIdx = xfer.ddgIndex;

    // Inherit shape/dtype from the source compute node's operand metadata.
    int srcCompute = xfer.sourceComputeIdx;
    if (srcCompute >= 0 && srcCompute < result.computeNodeCount) {
      const MsNode &src = result.nodes[srcCompute];
      // For compute output → transfer input, use the output operand info.
      for (const auto &info : src.operandInfos) {
        if (!info.isInput) {
          dn.shape = info.shape;
          dn.elementType = info.elementType;
          break;
        }
      }
    }

    // If no shape from producer, try the consumer's input operand info.
    // Prefer static shapes — a VectorGroup may contain ops with dynamic GM
    // operands (e.g. stride_load from memref<?xbf16>) alongside ops with
    // static UB shapes; the transfer cost needs the actual data block size.
    int dstCompute = xfer.targetComputeIdx;
    if (dn.shape.empty() && dstCompute >= 0 &&
        dstCompute < result.computeNodeCount) {
      const MsNode &dst = result.nodes[dstCompute];
      const MsNode::OperandInfo *dynFallback = nullptr;
      for (const auto &info : dst.operandInfos) {
        if (info.isInput && (xfer.cubeOperandSlot < 0 ||
                             info.operandIndex == xfer.cubeOperandSlot)) {
          if (llvm::any_of(info.shape, [](int64_t d) {
                return ShapedType::isDynamic(d);
              })) {
            LLVM_DEBUG({
              dbgs() << "[ms-dag-builder] D" << dn.id
                     << " skipping dynamic input shape [";
              for (int64_t d : info.shape)
                dbgs() << d << "x";
              dbgs() << "]\n";
            });
            if (!dynFallback)
              dynFallback = &info;
            continue;
          }
          dn.shape = info.shape;
          dn.elementType = info.elementType;
          break;
        }
      }
      if (dn.shape.empty() && dynFallback) {
        dn.shape = dynFallback->shape;
        dn.elementType = dynFallback->elementType;
      }
    }

    // Compute byte sizes.
    dn.elementBytes = getElementBytes(dn.elementType);
    if (dn.elementBytes > 0 && !dn.shape.empty()) {
      int64_t numElements = 1;
      for (int64_t dim : dn.shape) {
        if (ShapedType::isDynamic(dim)) {
          numElements = ShapedType::kDynamic;
          break;
        }
        numElements *= dim;
      }
      dn.totalBytes = ShapedType::isDynamic(numElements)
                          ? ShapedType::kDynamic
                          : numElements * dn.elementBytes;
    }

    // Find consumers: DDG successors of this transfer node.
    for (int succ : result.ddg.getSuccs()[xfer.ddgIndex])
      dn.consumerIdxs.push_back(succ);

    result.dataNodes.push_back(std::move(dn));
  }

  // --- Validation ---
  // After DAG2 every compute node must have at least one input path (a
  // synthetic transfer or a compute predecessor).
  for (int ci = 0; ci < result.computeNodeCount; ++ci) {
    const MsNode &cn = result.nodes[ci];
    if (cn.kind != MsNodeKind::CubeCompute &&
        cn.kind != MsNodeKind::VectorGroup)
      continue;
    if (result.ddg.getPreds()[ci].empty()) {
      LLVM_DEBUG(dbgs() << "[ms-dag-builder] DAG3 warning: compute node " << ci
                        << " has no input transfers\n");
    }
  }

  LLVM_DEBUG({
    dbgs() << "[ms-dag-builder] DAG3 data nodes: " << result.dataNodes.size()
           << "\n";
    dbgs() << "[ms-dag-builder] DAG3 DOT:\n";
    dumpDag3Dot(dbgs(), result);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Phase cost — MS cost model queries and latency backfill
//===----------------------------------------------------------------------===//

FailureOr<costmodel::CubeCostInput>
MsDagBuilder::buildCubeCostInput(const MsNode &sn) const {
  costmodel::CubeCostInput input;
  bool haveLhs = false;
  bool haveRhs = false;
  for (const MsNode::OperandInfo &info : sn.operandInfos) {
    if (!info.isInput || info.operandIndex < 0)
      continue;
    if (info.operandIndex == 0 && !haveLhs) {
      input.lhs = costmodel::ShapedOperandDesc(
          SmallVector<int64_t, costmodel::ShapeInlineCapacity>(
              info.shape.begin(), info.shape.end()),
          info.elementType);
      haveLhs = true;
    } else if (info.operandIndex == 1 && !haveRhs) {
      input.rhs = costmodel::ShapedOperandDesc(
          SmallVector<int64_t, costmodel::ShapeInlineCapacity>(
              info.shape.begin(), info.shape.end()),
          info.elementType);
      haveRhs = true;
    }
  }
  if (!haveLhs || !haveRhs) {
    LLVM_DEBUG({
      dbgs() << "[ms-dag-builder] cube cost input unusable: node #"
             << sn.ddgIndex << " (pipeline=" << sn.pipeline << ", op=";
      if (sn.primaryOp)
        dbgs() << sn.primaryOp->getName().getStringRef();
      else
        dbgs() << "<null>";
      dbgs() << ") missing " << (!haveLhs ? "lhs" : "rhs")
             << " operand shape metadata\n";
    });
    return failure();
  }
  return input;
}

costmodel::VectorGroupCostInput
MsDagBuilder::buildVectorGroupCostInput(const MsNode &sn) const {
  costmodel::VectorGroupCostInput input;

  // Reserve the upper bound once: filtering transfer ops only removes, so
  // sn.ops.size() bounds the op count.  Large groups over the inline
  // capacity get one allocation up front instead of repeated 2x growth.
  input.ops.reserve(sn.ops.size());

  // Walk in block order, skipping transfer ops (their cost is billed by
  // the DAG2 transfer nodes).  The op count is fully decided by the IR.
  for (Operation *op : sn.ops) {
    if (isTransferOp(op))
      continue;
    costmodel::VectorOpDesc opDesc;
    opDesc.op = op;
    // Filter operandInfos by ownerOp: both lists are bounded by
    // operandInfos.size(), reserve first then append.
    opDesc.inputs.reserve(sn.operandInfos.size());
    opDesc.outputs.reserve(sn.operandInfos.size());
    for (const MsNode::OperandInfo &info : sn.operandInfos) {
      if (info.ownerOp != op)
        continue;
      // Copy the full shape (exact rank, one-shot sizing).
      costmodel::ShapedOperandDesc desc(
          SmallVector<int64_t, costmodel::ShapeInlineCapacity>(
              info.shape.begin(), info.shape.end()),
          info.elementType);
      if (info.isInput)
        opDesc.inputs.push_back(std::move(desc));
      else
        opDesc.outputs.push_back(std::move(desc));
    }
    input.ops.push_back(std::move(opDesc));
  }
  return input;
}

FailureOr<costmodel::TransferCostInput>
MsDagBuilder::buildTransferCostInput(const MsNode &sn) const {
  // The transfer's data block is described by the MsDataNode it produces.
  const MsDataNode *dn = nullptr;
  for (const MsDataNode &data : result.dataNodes)
    if (data.producerIdx == sn.ddgIndex) {
      dn = &data;
      break;
    }
  if (!dn || dn->shape.empty() || !dn->elementType) {
    LLVM_DEBUG({
      dbgs() << "[ms-dag-builder] transfer cost input unusable: node #"
             << sn.ddgIndex << " (pipeline=" << sn.pipeline
             << ", tag=" << sn.storageTag << ") ";
      if (!dn)
        dbgs() << "has no data node\n";
      else if (dn->shape.empty())
        dbgs() << "has an empty shape\n";
      else
        dbgs() << "has no element type\n";
    });
    return failure();
  }
  costmodel::TransferCostInput input;
  input.data = costmodel::ShapedOperandDesc(
      SmallVector<int64_t, costmodel::ShapeInlineCapacity>(dn->shape.begin(),
                                                           dn->shape.end()),
      dn->elementType);
  input.pipeline = sn.pipeline;
  input.storageTag = sn.storageTag;
  return input;
}

/// Print one query's result line (shared by all three node kinds).
static void debugPrintCostResult(const costmodel::CostResult &cost) {
  LLVM_DEBUG(dbgs() << "[ms-cost-model]   -> costTime=" << cost.costTime << "ns"
                    << (cost.note.empty() ? "" : " note=" + cost.note) << "\n");
}

LogicalResult MsDagBuilder::assignCosts() {
  if (!costModel)
    return success(); // Keep LatencyProvider placeholders (compat mode).

  // --- Compute nodes (cube / vector groups) ---
  for (int i = 0; i < result.computeNodeCount; ++i) {
    MsNode &sn = result.nodes[i];
    auto &ddgNode = result.ddg.getNodes()[i];
    costmodel::CostResult cost{0, ""};

    if (sn.kind == MsNodeKind::CubeCompute) {
      FailureOr<costmodel::CubeCostInput> input = buildCubeCostInput(sn);
      if (failed(input))
        return scopeOp->emitWarning("[ms-cost-model] cube cost query failed: ")
               << "N" << sn.ddgIndex << " missing shaped operand metadata";
      // Print the query inputs BEFORE querying, so a failed query still
      // shows what it was fed.
      LLVM_DEBUG({
        Operation *dispOp = pickDisplayOp(sn);
        dbgs() << "[ms-cost-model] cube query N" << sn.ddgIndex << " op="
               << (dispOp ? dispOp->getName().getStringRef() : "<unknown>")
               << " lhs=[";
        for (int64_t d : input->lhs.getShape())
          dbgs() << d << "x";
        dbgs() << "] " << input->lhs.getElementType() << " rhs=[";
        for (int64_t d : input->rhs.getShape())
          dbgs() << d << "x";
        dbgs() << "] " << input->rhs.getElementType() << "\n";
      });
      cost = costModel->getCubeCost(*input);
    } else if (sn.kind == MsNodeKind::VectorGroup) {
      costmodel::VectorGroupCostInput input = buildVectorGroupCostInput(sn);
      // Per-op input details (vector groups must print per op, not per
      // group).
      LLVM_DEBUG({
        dbgs() << "[ms-cost-model] vector query N" << sn.ddgIndex << " ("
               << input.ops.size() << " ops)\n";
        for (const costmodel::VectorOpDesc &opDesc : input.ops) {
          dbgs() << "[ms-cost-model]   op="
                 << (opDesc.op ? opDesc.op->getName().getStringRef()
                               : "<unknown>")
                 << " inputs=";
          for (const costmodel::ShapedOperandDesc &in : opDesc.inputs) {
            dbgs() << "[";
            for (int64_t d : in.getShape())
              dbgs() << d << "x";
            dbgs() << "]" << in.getElementType() << " ";
          }
          dbgs() << "\n";
        }
      });
      cost = costModel->getVectorGroupCost(input);
    } else {
      continue;
    }

    debugPrintCostResult(cost);

    if (cost.costTime == -1) {
      // Error: the note carries the specifics; print it and fail the SMS
      // build (the caller falls back to the Default scheduler).
      Operation *anchor = sn.ops.empty() ? scopeOp : sn.ops.front();
      return anchor->emitWarning("[ms-cost-model] cost query failed: ")
             << cost.note;
    }
    if (!cost.note.empty() && sn.kind == MsNodeKind::VectorGroup) {
      // Vector ops whose dtype is outside the dtype table — the only
      // anomaly that does not fail the build.  The latency already counted
      // via the fallback multiplier; the note lists every fallback op name
      // so one warning covers the whole group (no per-op name matching,
      // which could misfire on substring-equal names).
      Operation *anchor = sn.ops.empty() ? scopeOp : sn.ops.front();
      anchor->emitWarning("[ms-cost-model] vector op(s) with uncalibrated "
                          "dtype, using fallback param: ")
          << cost.note;
    }

    ddgNode.latency = static_cast<int>(cost.costTime);
    ddgNode.issueTime = static_cast<int>(cost.costTime);
  }

  // --- Transfer nodes (DAG2 synthetic) ---
  for (int i = result.computeNodeCount; i < result.ddg.size(); ++i) {
    MsNode &sn = result.nodes[i];
    auto &ddgNode = result.ddg.getNodes()[i];
    FailureOr<costmodel::TransferCostInput> input = buildTransferCostInput(sn);
    if (failed(input)) {
      return pipelineLoop->emitWarning(
                 "[ms-cost-model] transfer cost query failed: ")
             << "N" << sn.ddgIndex << " (" << sn.pipeline << ", "
             << sn.storageTag << ") missing data node metadata";
    }
    LLVM_DEBUG({
      dbgs() << "[ms-cost-model] transfer query N" << sn.ddgIndex << " "
             << sn.pipeline << " " << sn.storageTag << " data=[";
      for (int64_t d : input->data.getShape())
        dbgs() << d << "x";
      dbgs() << "] " << input->data.getElementType() << "\n";
    });
    costmodel::CostResult cost = costModel->getTransferCost(*input);
    debugPrintCostResult(cost);

    if (cost.costTime == -1) {
      // Transfer nodes are synthetic (no real op); anchor the diagnostic
      // on the pipeline loop, like other DAG2 diagnostics.
      return pipelineLoop->emitWarning(
                 "[ms-cost-model] transfer cost query failed: ")
             << cost.note;
    }
    ddgNode.latency = static_cast<int>(cost.costTime);
    ddgNode.issueTime = static_cast<int>(cost.costTime);
  }

  result.ddg.refreshEdgeLatencies();

  // --- Whole-DAG summary after the backfill (§9.2: reuse dumpDag3Dot —
  // its labels already carry lat=, so the real costs show up unchanged).
  LLVM_DEBUG({
    dbgs() << "[ms-cost-model] costs assigned:\n"
           << "  compute nodes: " << result.computeNodeCount << "\n"
           << "  transfer nodes: "
           << (result.ddg.size() - result.computeNodeCount) << "\n"
           << "  total DDG edges: " << result.ddg.getEdges().size() << "\n";
    dbgs() << "[ms-cost-model] cost-assigned DAG DOT:\n";
    dumpDag3Dot(dbgs(), result);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Phase 1 — classify ops (legacy)
//===----------------------------------------------------------------------===//

/// Determine the canonical SMS pipeline name for an operation.
std::string MsDagBuilder::canonicalizePipeline(TCoreType core, Operation *op) {
  // Transfer ops first — their pipeline is dictated by op type, not core.
  if (isa<LoadOp, ND2NZOp>(op))
    return "mte2";
  if (isa<StoreOp>(op))
    return "mte3";
  if (isa<FixpipeOp>(op))
    return "fixpipe";
  if (isCrossCoreCopy(op)) {
    // Cross-core copies are conceptually L1→UB transfers = MTE1.
    return "mte1";
  }

  // Compute ops.
  if (core == TCoreType::CUBE)
    return "cube";
  return "vector";
}

LogicalResult MsDagBuilder::classifyOps() {
  // Walk the entire for loop body recursively to find all HIVM ops within
  // scope bodies.  The for loop body contains scope.scope ops; the actual
  // compute/transfer ops (hivm.hir.load, mmad, fixpipe, store, etc.) live
  // inside those scopes.  SMS builds a fine-grained per-op DDG, not the
  // coarse scope-level partitioning that WorklistBuilder does.
  Block *targetBlock = pipelineLoop.getBody();
  if (!targetBlock || targetBlock->empty())
    return scopeOp->emitWarning("[ms-dag-builder] empty loop body");

  // Walk all ops inside the for loop body, recursing into scope regions.
  WalkResult classifyWalk =
      pipelineLoop->walk([&](Operation *op) -> WalkResult {
        // Skip the for loop itself and yields/returns.
        if (op == pipelineLoop.getOperation())
          return WalkResult::advance();
        if (isa<scf::YieldOp, scope::ReturnOp>(op))
          return WalkResult::advance();

        // Skip scope.scope ops themselves — we recurse into their bodies.
        if (isa<scope::ScopeOp>(op))
          return WalkResult::advance();

        // Separators — these become their own transfer nodes.
        if (isSeparator(op)) {
          separators.push_back(op);
          transferOps.push_back(op);
          return WalkResult::advance();
        }

        // Region ops (other than scope): classify body core type, reject mixed.
        if (!op->getRegions().empty() && !isa<scope::ScopeOp>(op)) {
          RegionCoreKind rk = classifyRegion(*op);
          if (rk == RegionCoreKind::Mixed) {
            op->emitWarning("[ms-dag-builder] mixed-core region op "
                            "unsupported by SMS");
            return WalkResult::interrupt();
          }
          if (rk == RegionCoreKind::Cube || rk == RegionCoreKind::Vector) {
            computeOps.push_back(op);
            return WalkResult::skip(); // Don't recurse into the region body.
          }
          return WalkResult::advance();
        }

        // Load-like ops — transfer nodes (MTE2).
        if (isLoadLikeOp(op)) {
          transferOps.push_back(op);
          return WalkResult::advance();
        }

        // Core compute ops — cube or vector.
        if (isCoreOp(*op)) {
          computeOps.push_back(op);
          return WalkResult::advance();
        }

        // Non-core ops (arith, tensor.empty, annotation, memref.alloc, etc.)
        // are not added to the DAG.
        LLVM_DEBUG(dbgs() << "[ms-dag-builder] skipping non-core op: " << *op
                          << "\n");
        return WalkResult::advance();
      });

  if (classifyWalk.wasInterrupted())
    return failure();

  LLVM_DEBUG({
    dbgs() << "[ms-dag-builder] Phase 1 classification:\n"
           << "  separators: " << separators.size() << "\n"
           << "  computeOps: " << computeOps.size() << "\n"
           << "  transferOps: " << transferOps.size() << "\n";
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Phase 2 — SSA dependencies
//===----------------------------------------------------------------------===//

// cf. WorklistBuilder.cpp: populateDependencies (lines 637-681)
// Walks shaped SSA results forward from each separator to find downstream ops
// that depend on it.  Unlike the original, we do NOT build a dependenceMap
// keyed by (consumer → set<separator>) because we need DDG edges, not
// WorkItem grouping.  Instead, we record the reachable ops for each separator.
LogicalResult MsDagBuilder::buildSsaDependencies() {
  for (Operation *separator : separators) {
    SmallVector<Operation *> dfsStack = {separator};
    DenseSet<Operation *> visited;

    while (!dfsStack.empty()) {
      Operation *op = dfsStack.pop_back_val();
      if (visited.contains(op) || !scopeOp->isAncestor(op))
        continue;
      visited.insert(op);

      // Validate DPS memref root (cf. original lines 647-660).
      if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
        Value init = dpsOp.getDpsInitOperand(0)->get();
        if (isa<MemRefType>(init.getType())) {
          Value src = traceValueDef(init);
          auto blkArg = dyn_cast<BlockArgument>(src);
          if (!blkArg) {
            auto *defOp = src.getDefiningOp();
            if (!defOp || !isa<memref::AllocOp>(defOp))
              return separator->emitWarning(
                  "[ms-dag-builder] unexpected memref root");
          } else if (!isa<func::FuncOp>(blkArg.getOwner()->getParentOp())) {
            return separator->emitWarning(
                "[ms-dag-builder] unexpected block arg root");
          }
        }
      }

      for (Value resultVal : op->getResults()) {
        if (!isa<ShapedType>(resultVal.getType()))
          continue;
        for (OpOperand &use : resultVal.getUses()) {
          Operation *usr = use.getOwner();
          Operation *scopedUsr = getContainedParent(scopeOp, *usr);
          if (!scopedUsr)
            continue;
          if (isa<scf::YieldOp, scf::ConditionOp>(scopedUsr) ||
              scopedUsr == separator)
            continue;
          dfsStack.push_back(scopedUsr);
          separatorDeps[separator].insert(scopedUsr);
        }
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Phase 3 — loop-carried dependency detection
//===----------------------------------------------------------------------===//

// cf. WorklistBuilder.cpp: populateLoopCarriedDependencies (lines 683-728)
// P0: any loop-carried dependency causes SMS rejection (no RecMII support).
void MsDagBuilder::detectLoopCarriedDeps() {
  auto maybeYield = pipelineLoop.getYieldedValuesMutable();
  if (!maybeYield.has_value())
    return;

  for (OpOperand &yieldOperand : *maybeYield) {
    Value yieldVal = yieldOperand.get();
    if (!isa<TensorType>(yieldVal.getType()))
      continue;
    Operation *defining = yieldVal.getDefiningOp();
    if (!defining || !scopeOp->isAncestor(defining))
      continue;

    BlockArgument iterArg =
        pipelineLoop.getRegionIterArgs()[yieldOperand.getOperandNumber()];
    SmallVector<Operation *> dfsStack(iterArg.getUsers());
    DenseSet<Operation *> visited;

    while (!dfsStack.empty()) {
      Operation *op = getContainedParent(scopeOp, *dfsStack.pop_back_val());
      if (visited.contains(op) || op == defining)
        continue;
      visited.insert(op);

      if (isa<DestinationStyleOpInterface>(op)) {
        // Found a loop-carried dependency: a DPS consumer of the iter-arg
        // that is different from the yielding op.
        result.hasLoopCarriedDeps = true;
        result.loopCarriedDiag =
            "[ms-dag-builder] loop-carried dependency detected "
            "(no RecMII support in P0)";
        LLVM_DEBUG(dbgs() << result.loopCarriedDiag << "\n  producer: "
                          << *defining << "\n  consumer: " << *op << "\n");
        return;
      }

      for (Operation *usr : op->getUsers()) {
        if (isa<scf::YieldOp>(usr))
          continue;
        dfsStack.push_back(usr);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase 4 — memref subnet tracing
//===----------------------------------------------------------------------===//

// cf. WorklistBuilder.cpp: traceMemrefSubnet (lines 730-813)
// Traces the alloc→cast/view→writer/toTensor chain for each memref-subnet
// writer.  Populates result.outputMemrefMap (toTensor → DPS writer).
// READ-ONLY: never pushes onto a workingStack for WorkItem construction.
LogicalResult MsDagBuilder::traceMemrefSubnets() {
  // Collect all memref-subnet writers within the pipeline loop.
  SmallVector<Operation *> writers;
  pipelineLoop->walk([&](Operation *op) {
    if (isMemrefSubnetWriter(op))
      writers.push_back(op);
  });

  for (Operation *op : writers) {
    // Find the writer DPS op and the toTensor user.
    DestinationStyleOpInterface writer = nullptr;
    if (op->getNumOperands() < 2)
      continue;
    Value targetOperand = op->getOperand(1);
    if (isa<TensorType>(targetOperand.getType()))
      writer = cast<DestinationStyleOpInterface>(op);

    Operation *traceStart = op;
    Operation *defining = targetOperand.getDefiningOp();
    while (defining) {
      if (!scopeOp->isAncestor(defining))
        break;
      traceStart = defining;
      if (isa<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp>(
              defining))
        break;
      if (isa<memref::CastOp, memref::ReinterpretCastOp,
              memref::MemorySpaceCastOp, memref::CollapseShapeOp,
              memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp,
              bufferization::ToTensorOp, tensor::ExtractSliceOp>(defining))
        defining = defining->getOperand(0).getDefiningOp();
      else
        return defining->emitWarning(
            "[ms-dag-builder] unexpected memref op in chain");
    }

    // Walk users of the traced root down to find toTensor and DPS writer.
    SmallVector<Operation *> userTraceStack = {traceStart};
    bufferization::ToTensorOp toTensor = nullptr;
    Operation *separatorOp = op;

    while (!userTraceStack.empty()) {
      Operation *def = userTraceStack.pop_back_val();
      for (OpOperand &use : def->getUses()) {
        Operation *usr = use.getOwner();
        if (auto dps = dyn_cast<DestinationStyleOpInterface>(usr)) {
          OpOperand *init = dps.getDpsInitOperand(0);
          if (!init || init != &use)
            continue;
          if (writer && usr != separatorOp)
            return usr->emitWarning(
                "[ms-dag-builder] expecting only one op writing to a defined "
                "memref");
          writer = dps;
          continue;
        }
        if (auto tt = dyn_cast<bufferization::ToTensorOp>(usr)) {
          if (toTensor)
            return usr->emitWarning(
                "[ms-dag-builder] expecting only one toTensor for a "
                "defined memref");
          toTensor = tt;
          continue;
        }
        if (isa<memref::CastOp, memref::ReinterpretCastOp,
                memref::MemorySpaceCastOp, memref::CollapseShapeOp,
                memref::ExpandShapeOp, memref::SubViewOp, memref::ViewOp>(
                usr)) {
          userTraceStack.push_back(usr);
        }
      }
    }

    if (toTensor && !writer) {
      return toTensor->emitWarning(
          "[ms-dag-builder] expecting toTensor to have dps op to write to it");
    }
    if (toTensor && writer)
      result.outputMemrefMap[toTensor] = writer;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Phase 5 — construct the pure DDG
//===----------------------------------------------------------------------===//

LogicalResult MsDagBuilder::constructDdg() {
  //--- Step 1: Create compute nodes ----------------------------------------

  // Identify cube vs vector for each compute op.
  // Group contiguous vector ops into indivisible VectorGroups.
  SmallVector<std::pair<Operation *, TCoreType>> classifiedCompute;
  for (Operation *op : computeOps) {
    // Determine core type from attrs or queryCoreTypeHelper.
    TCoreType core;
    if (op->hasAttr(CubeOnlyAttrName)) {
      core = TCoreType::CUBE;
    } else if (op->hasAttr(VecOnlyAttrName)) {
      core = TCoreType::VECTOR;
    } else {
      auto maybeCore = queryCoreTypeHelper(op);
      if (!maybeCore) {
        // Check if it's a region op we already classified.
        RegionCoreKind rk = classifyRegion(*op);
        if (rk == RegionCoreKind::Cube)
          core = TCoreType::CUBE;
        else if (rk == RegionCoreKind::Vector)
          core = TCoreType::VECTOR;
        else
          continue; // Non-core, skip.
      } else {
        core = *maybeCore;
        if (core == TCoreType::CUBE_OR_VECTOR) {
          // Ambiguous — check cross-core copy fallback.
          if (isCrossCoreCopy(op))
            core = TCoreType::VECTOR;
          else
            continue; // Cannot classify → skip.
        }
      }
    }
    classifiedCompute.push_back({op, core});
  }

  // Build vector groups: contiguous runs of VECTOR ops in block order.
  // Each group becomes one DDG node (indivisible).
  SmallVector<SmallVector<Operation *, 4>> vectorGroupList;
  SmallVector<Operation *> cubeOps;
  SmallVector<Operation *, 4> currentVecGroup;

  for (auto &[op, core] : classifiedCompute) {
    if (core == TCoreType::VECTOR) {
      currentVecGroup.push_back(op);
    } else {
      // Flush any accumulated vector group.
      if (!currentVecGroup.empty()) {
        vectorGroupList.push_back(std::move(currentVecGroup));
        currentVecGroup.clear();
      }
      if (core == TCoreType::CUBE)
        cubeOps.push_back(op);
    }
  }
  if (!currentVecGroup.empty())
    vectorGroupList.push_back(std::move(currentVecGroup));

  //--- Step 2: Assign DDG indices and create MsNodes ----------------------

  int ddgIdx = 0;

  // Cube compute nodes — each is one DDG node.
  for (Operation *op : cubeOps) {
    int latency = latencyProvider.getLatency(op);
    int issueTime = latencyProvider.getIssueTime(op);
    if (latency <= 0 || issueTime <= 0)
      return op->emitWarning("[ms-dag-builder] non-positive latency/issue "
                             "time for cube op");

    MsNode sn;
    sn.ddgIndex = ddgIdx;
    sn.kind = MsNodeKind::CubeCompute;
    sn.primaryOp = op;
    sn.ops.push_back(op);
    sn.pipeline = "cube";
    sn.isParallelUnit = false;
    result.nodes.push_back(sn);
    result.opToDdgIndex[op] = ddgIdx;
    assignedOps.insert(op);

    sms::OpNode node;
    node.ordinal = ddgIdx;
    node.name = op->getName().getStringRef().str();
    node.pipeline = "cube";
    node.latency = latency;
    node.issueTime = issueTime;
    node.isParallelUnit = false;
    result.ddg.addNode(node);

    ++ddgIdx;
  }

  // Vector group nodes — each group is one DDG node.
  for (auto &group : vectorGroupList) {
    Operation *primaryOp = group.front();
    // Use the max latency in the group as the group's latency.
    int groupLatency = 0;
    int groupIssue = 0;
    for (Operation *op : group) {
      int lat = latencyProvider.getLatency(op);
      int iss = latencyProvider.getIssueTime(op);
      if (lat <= 0 || iss <= 0)
        return op->emitWarning("[ms-dag-builder] non-positive latency/issue "
                               "time for vector op");
      groupLatency = std::max(groupLatency, lat);
      groupIssue = std::max(groupIssue, iss);
    }

    MsNode sn;
    sn.ddgIndex = ddgIdx;
    sn.kind = MsNodeKind::VectorGroup;
    sn.primaryOp = primaryOp;
    sn.ops.assign(group.begin(), group.end());
    sn.pipeline = "vector";
    sn.isParallelUnit = false;
    result.nodes.push_back(sn);
    for (Operation *op : group) {
      result.opToDdgIndex[op] = ddgIdx;
      assignedOps.insert(op);
    }

    // Store group mapping for edge construction.
    vectorGroups[primaryOp] =
        SmallVector<Operation *, 4>(group.begin(), group.end());

    sms::OpNode node;
    node.ordinal = ddgIdx;
    node.name = primaryOp->getName().getStringRef().str();
    node.pipeline = "vector";
    node.latency = groupLatency;
    node.issueTime = groupIssue;
    node.isParallelUnit = false;
    result.ddg.addNode(node);

    ++ddgIdx;
  }

  //--- Step 3: Create transfer nodes ---------------------------------------

  for (Operation *op : transferOps) {
    if (assignedOps.contains(op))
      continue; // Already in a group.

    std::string pipeline = canonicalizePipeline(TCoreType::CUBE_OR_VECTOR, op);
    bool isParallel = (pipeline == "mte1" || pipeline == "mte2" ||
                       pipeline == "mte3" || pipeline == "fixpipe");

    int latency = latencyProvider.getLatency(op);
    int issueTime = latencyProvider.getIssueTime(op);
    if (latency <= 0 || issueTime <= 0)
      return op->emitWarning("[ms-dag-builder] non-positive latency/issue "
                             "time for transfer op");

    MsNode sn;
    sn.ddgIndex = ddgIdx;
    sn.kind = MsNodeKind::Transfer;
    sn.primaryOp = op;
    sn.ops.push_back(op);
    sn.pipeline = pipeline;
    sn.isParallelUnit = isParallel;
    result.nodes.push_back(sn);
    result.opToDdgIndex[op] = ddgIdx;
    assignedOps.insert(op);

    sms::OpNode node;
    node.ordinal = ddgIdx;
    node.name = op->getName().getStringRef().str();
    node.pipeline = pipeline;
    node.latency = latency;
    node.issueTime = issueTime;
    node.isParallelUnit = isParallel;
    result.ddg.addNode(node);

    ++ddgIdx;
  }

  //--- Step 4: Add edges ---------------------------------------------------

  // 4a: SSA data-flow edges.
  // For each DDG node, trace its operands back to producing ops that are also
  // in the DDG, and add edges.
  for (const MsNode &sn : result.nodes) {
    DenseSet<int> preds; // Avoid duplicate edges.
    for (Operation *op : sn.ops) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        // Trace through views/casts to the root defining op.
        Operation *scopedDef = getContainedParent(scopeOp, *defOp);
        if (!scopedDef)
          continue;
        auto it = result.opToDdgIndex.find(scopedDef);
        if (it == result.opToDdgIndex.end())
          continue;
        int predIdx = it->second;
        if (predIdx == sn.ddgIndex)
          continue; // Self-edge.
        if (preds.insert(predIdx).second)
          result.ddg.addEdge(predIdx, sn.ddgIndex);
      }
    }
  }

  // 4b: Separator dependency edges.
  // If a separator op S reaches a consumer op C (via separatorDeps),
  // and both S and C are DDG nodes, add an edge S → C.
  for (auto &[separator, deps] : separatorDeps) {
    auto sepIt = result.opToDdgIndex.find(separator);
    if (sepIt == result.opToDdgIndex.end())
      continue;
    int sepIdx = sepIt->second;
    for (Operation *dep : deps) {
      Operation *scopedDep = getContainedParent(scopeOp, *dep);
      if (!scopedDep)
        continue;
      auto depIt = result.opToDdgIndex.find(scopedDep);
      if (depIt == result.opToDdgIndex.end())
        continue;
      int depIdx = depIt->second;
      if (depIdx == sepIdx)
        continue;
      // Check if this edge already exists (from SSA flow).
      bool exists = false;
      for (const sms::DDGEdge &edge : result.ddg.getOutEdges()[sepIdx]) {
        if (edge.dst == depIdx) {
          exists = true;
          break;
        }
      }
      if (!exists)
        result.ddg.addEdge(sepIdx, depIdx);
    }
  }

  // 4c: Memory alias edges via memrefDFS.
  // For memref-subnet writers, trace their memref alias set and add edges
  // to/from other DDG nodes that use the same memory.
  for (const MsNode &sn : result.nodes) {
    if (sn.kind != MsNodeKind::Transfer)
      continue;
    Operation *op = sn.primaryOp;
    if (!isMemrefSubnetWriter(op))
      continue;

    // For each memref operand (typically operand 1 = dst), trace aliases.
    if (op->getNumOperands() < 2)
      continue;
    Value memref = op->getOperand(1);
    if (!isa<MemRefType>(memref.getType()))
      continue;

    SmallVector<Operation *> aliasUsers;
    memrefDFS(memref, aliasUsers);
    for (Operation *usr : aliasUsers) {
      if (usr == op)
        continue;
      Operation *scopedUsr = getContainedParent(scopeOp, *usr);
      if (!scopedUsr)
        continue;
      auto it = result.opToDdgIndex.find(scopedUsr);
      if (it == result.opToDdgIndex.end())
        continue;
      int usrIdx = it->second;
      if (usrIdx == sn.ddgIndex)
        continue;

      // Writer → reader: the writer (transfer) must complete before reader.
      // Only add if not already present.
      bool exists = false;
      for (const sms::DDGEdge &edge : result.ddg.getOutEdges()[sn.ddgIndex]) {
        if (edge.dst == usrIdx) {
          exists = true;
          break;
        }
      }
      if (!exists)
        result.ddg.addEdge(sn.ddgIndex, usrIdx);
    }
  }

  LLVM_DEBUG({
    dbgs() << "[ms-dag-builder] Phase 5 DAG constructed:\n"
           << "  nodes: " << result.ddg.size() << "\n"
           << "  edges: " << result.ddg.getEdges().size() << "\n";
    dbgs() << "[ms-dag-builder] Legacy DAG DOT:\n";
    dumpDag2Dot(dbgs(), result);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Top-level build()
//===----------------------------------------------------------------------===//

FailureOr<MsDagBuildResult> MsDagBuilder::build() {
  // Choose the DAG construction path based on whether Default WorkItems are
  // available.
  if (defaultBuild) {
    // --- DAG1 path: WorkItem-based node creation ---
    // Phase 1-WI: Create compute nodes from Default WorkItems.
    if (buildNodesFromWorkItems().failed())
      return failure();

    // Phase 2-WI: Build inter-node dependency edges.
    if (buildWorkItemDependencies().failed())
      return failure();

    // Phase 3-WI: Collect operand shape/dtype metadata.
    collectOperandMetadata();

    // Phase DAG2: Insert synthetic transfer nodes (CVpie rules).
    if (buildTransferNodes().failed())
      return failure();

    // Phase DAG3: Derive data nodes and validate data flow.
    if (buildDataNodes().failed())
      return failure();

    // Phase cost: Query the MS cost model and backfill latencies (no-op
    // when no cost model was injected).
    if (assignCosts().failed())
      return failure();

    // Validate the DAG.
    if (result.ddg.size() == 0)
      return scopeOp->emitWarning("[ms-dag-builder] DAG is empty");

    sms::VerifyResult vr = sms::verifyDag(result.ddg);
    if (!vr.ok) {
      LLVM_DEBUG(dbgs() << "[ms-dag-builder] DAG verification failed: "
                        << vr.error << "\n");
      return scopeOp->emitWarning("[ms-dag-builder] DAG verification failed: ")
             << vr.error;
    }

    // Build the per-node dependency map (mirrors DDG edges but by index).
    for (const sms::DDGEdge &edge : result.ddg.getEdges())
      result.dependenceMap[edge.dst].insert(edge.src);

    return std::move(result);
  }

  // --- Legacy path: per-op node creation (deprecated) ---
  // Phase 1: classify ops in the scope body.
  if (classifyOps().failed())
    return failure();

  // Phase 2: build SSA dependency map from separators.
  if (buildSsaDependencies().failed())
    return failure();

  // Phase 3: detect loop-carried dependencies.
  detectLoopCarriedDeps();
  if (result.hasLoopCarriedDeps)
    return failure();

  // Phase 4: trace memref subnets for output registration.
  if (traceMemrefSubnets().failed())
    return failure();

  // Phase 5: construct the pure DDG with latencies and edges.
  if (constructDdg().failed())
    return failure();

  // Validate the DAG.
  if (result.ddg.size() == 0)
    return scopeOp->emitWarning("[ms-dag-builder] DAG is empty");

  sms::VerifyResult vr = sms::verifyDag(result.ddg);
  if (!vr.ok) {
    LLVM_DEBUG(dbgs() << "[ms-dag-builder] DAG verification failed: "
                      << vr.error << "\n");
    return scopeOp->emitWarning("[ms-dag-builder] DAG verification failed: ")
           << vr.error;
  }

  // Build the per-node dependency map (mirrors DDG edges but by index).
  for (const sms::DDGEdge &edge : result.ddg.getEdges())
    result.dependenceMap[edge.dst].insert(edge.src);

  return std::move(result);
}

} // namespace mlir
