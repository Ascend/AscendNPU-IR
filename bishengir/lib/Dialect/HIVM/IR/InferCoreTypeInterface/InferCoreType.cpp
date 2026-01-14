//===- InferCoreType.cpp - InferCoreType Interface Impl. ------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <numeric>

using namespace mlir;
using namespace mlir::hivm;

namespace {

static std::optional<TCoreType>
inferCoreTypeBasedOnPipes(ArrayRef<hivm::PIPE> pipes) {
  if (pipes.empty()) {
    return std::nullopt;
  }

  static std::set<hivm::PIPE> cubeOnyPipes = {
      hivm::PIPE::PIPE_FIX, hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_MTE1};
  static std::set<hivm::PIPE> vecOnyPipes = {hivm::PIPE::PIPE_MTE3,
                                             hivm::PIPE::PIPE_V};

  SmallVector<TCoreType> coreTypes(pipes.size(), TCoreType::CUBE_OR_VECTOR);
  std::transform(pipes.begin(), pipes.end(), coreTypes.begin(),
                 [](hivm::PIPE pipe) -> TCoreType {
                   TCoreType result = hivm::TCoreType::CUBE_OR_VECTOR;
                   if (cubeOnyPipes.count(pipe) != 0) {
                     result = TCoreType::CUBE;
                   } else if (vecOnyPipes.count(pipe) != 0) {
                     result = TCoreType::VECTOR;
                   }
                   return result;
                 });

  return std::accumulate(
      coreTypes.begin(), coreTypes.end(),
      std::optional<TCoreType>(TCoreType::CUBE_OR_VECTOR),
      [](std::optional<hivm::TCoreType> acc,
         hivm::TCoreType t2) -> std::optional<hivm::TCoreType> {
        if (!acc) {
          return std::nullopt;
        }

        hivm::TCoreType t1 = acc.value();

        hivm::TCoreType result = hivm::TCoreType::CUBE_OR_VECTOR;
        if (t1 == hivm::TCoreType::CUBE_OR_VECTOR) {
          result = t2;
        } else if (t2 == hivm::TCoreType::CUBE_OR_VECTOR) {
          result = t1;
        } else if (t1 == t2) {
          result = t1;
        } else {
          return std::nullopt;
        }

        return result;
      });
}

template <typename GlobalMixMatmulTy>
std::optional<TCoreType>
inferCoreTypeForGlobalMixMatmulOps(GlobalMixMatmulTy *mixMatmulOp) {
  TCoreType coreType = TCoreType::CUBE_AND_VECTOR;
  func::FuncOp enclosingFunc =
      (*mixMatmulOp)->template getParentOfType<func::FuncOp>();
  if (!enclosingFunc)
    return coreType;

  auto funcCoreTypeAttr =
      enclosingFunc->getAttrOfType<TFuncCoreTypeAttr>(TFuncCoreTypeAttr::name);
  if (!funcCoreTypeAttr)
    return coreType;

  // if the mix matmul is inside a aiv/aic function, then the op's core type
  // is consistent with the function type
  switch (funcCoreTypeAttr.getFuncCoreType()) {
  case TFuncCoreType::AIC:
    coreType = TCoreType::CUBE;
    break;
  case TFuncCoreType::AIV:
    coreType = TCoreType::VECTOR;
    break;
  default:
    break;
  }
  return coreType;
}

} // namespace

//===----------------------------------------------------------------------===//
// HIVM Ops
//===----------------------------------------------------------------------===//

std::optional<TCoreType> CustomOp::inferCoreType() {
  if (auto coreTypeAttr = getOperation()->template getAttrOfType<TCoreTypeAttr>(
          TCoreTypeAttr::name)) {
    return coreTypeAttr.getTcoretype();
  }

  return {};
}

std::optional<TCoreType> ConvertLayoutOp::inferCoreType() {
  BaseMemRefType srcMemRefTy = getSource().getType();
  hivm::AddressSpace addrSpace =
      static_cast<hivm::AddressSpace>(srcMemRefTy.getMemorySpaceAsInt());

  TCoreType result = TCoreType::CUBE_OR_VECTOR;
  if (addrSpace == hivm::AddressSpace::UB) {
    result = TCoreType::VECTOR;
  } else if (addrSpace == hivm::AddressSpace::L1 ||
             addrSpace == hivm::AddressSpace::L0A ||
             addrSpace == hivm::AddressSpace::L0B ||
             addrSpace == hivm::AddressSpace::L0C) {
    result = TCoreType::CUBE;
  }

  return result;
}

std::optional<TCoreType> DebugOp::inferCoreType() {
  std::optional<hivm::TCoreTypeAttr> maybeTCoreTypeAttr = this->getTcoretype();
  if (maybeTCoreTypeAttr.has_value() &&
      maybeTCoreTypeAttr.value().getTcoretype() !=
          hivm::TCoreType::CUBE_OR_VECTOR) {
    return maybeTCoreTypeAttr.value().getTcoretype();
  } else {
    mlir::Value arg = this->getArg();
    // first try the definingOp (TODO: change to tracing)
    Operation *definingOp = arg.getDefiningOp();
    if (definingOp) {
      auto res = getCoreType(definingOp);
      if (succeeded(res)) {
        this->setTcoretypeAttr(
            hivm::TCoreTypeAttr::get(this->getContext(), res.value()));
        return res.value();
      }
    }
    // finally if we cannot get a definite answer, just use CUBE_OR_VECTOR
    this->setTcoretypeAttr(hivm::TCoreTypeAttr::get(
        this->getContext(), hivm::TCoreType::CUBE_OR_VECTOR));
    return TCoreType::CUBE_OR_VECTOR;
  }
}

//===----------------------------------------------------------------------===//
// HIVM Synchronization Ops
//===----------------------------------------------------------------------===//

std::optional<TCoreType> SetFlagOp::inferCoreType() {
  hivm::PIPE p1 = getSetPipe().getPipe();
  hivm::PIPE p2 = getWaitPipe().getPipe();
  return inferCoreTypeBasedOnPipes({p1, p2});
}

std::optional<TCoreType> WaitFlagOp::inferCoreType() {
  hivm::PIPE p1 = getSetPipe().getPipe();
  hivm::PIPE p2 = getWaitPipe().getPipe();
  return inferCoreTypeBasedOnPipes({p1, p2});
}

std::optional<TCoreType> SyncBlockSetOp::inferCoreType() {
  return getTcoreTypeAttr().getTcoretype();
}

std::optional<TCoreType> PipeBarrierOp::inferCoreType() {
  hivm::PIPE pipe = getPipeAttr().getPipe();
  return inferCoreTypeBasedOnPipes({pipe});
}

std::optional<TCoreType> SyncBlockWaitOp::inferCoreType() {
  return getTcoreTypeAttr().getTcoretype();
}

std::optional<TCoreType> SyncBlockOp::inferCoreType() {
  hivm::SyncBlockMode mode = getSyncBlockModeAttr().getSyncMode();

  hivm::TCoreType result = TCoreType::CUBE_OR_VECTOR;
  if (mode == hivm::SyncBlockMode::BARRIER_CUBE ||
      mode == hivm::SyncBlockMode::ALL_CUBE) {
    result = TCoreType::CUBE;
  } else if (mode == hivm::SyncBlockMode::BARRIER_VECTOR ||
             mode == hivm::SyncBlockMode::ALL_VECTOR ||
             mode == hivm::SyncBlockMode::ALL_SUB_VECTOR) {
    result = TCoreType::VECTOR;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// HIVM DMA Ops
//===----------------------------------------------------------------------===//

std::optional<TCoreType> LoadOp::inferCoreType() {
  MemRefType srcMemRefTy = dyn_cast<MemRefType>(getSrc().getType());
  MemRefType dstMemRefTy = dyn_cast<MemRefType>(getDst().getType());
  if (srcMemRefTy && dstMemRefTy) {
    auto fromAddrSpace =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(srcMemRefTy.getMemorySpace());
    auto toAddrSpace =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(dstMemRefTy.getMemorySpace());
    if (fromAddrSpace && toAddrSpace) {
      bool isGMToUB =
          fromAddrSpace.getAddressSpace() == hivm::AddressSpace::GM &&
          toAddrSpace.getAddressSpace() == hivm::AddressSpace::UB;

      return isGMToUB ? TCoreType::VECTOR : TCoreType::CUBE;
    }
  }

  auto dstAllocVal =
      dstMemRefTy ? utils::tracebackMemRef(getDst()) : getResult(0);
  DenseSet<Value> visited;
  auto userAllCube = utils::checkUsersAllWithCondition(
      dstAllocVal, getOperation(), visited,
      [](Operation *op) {
        auto coreType = hivm::detail::queryCoreTypeHelper(op);
        return coreType == TCoreType::CUBE;
      },
      [](Operation *op) {
        auto coreType = hivm::detail::queryCoreTypeHelper(op);
        return !coreType;
      });
  if (userAllCube.has_value() && userAllCube.value()) {
    return TCoreType::CUBE;
  }
  visited.clear();
  auto userAllVec = utils::checkUsersAllWithCondition(
      dstAllocVal, getOperation(), visited,
      [](Operation *op) {
        auto coreType = hivm::detail::queryCoreTypeHelper(op);
        return coreType == TCoreType::VECTOR;
      },
      [](Operation *op) {
        auto coreType = hivm::detail::queryCoreTypeHelper(op);
        return !coreType;
      });
  visited.clear();
  if (userAllVec.has_value() && userAllVec.value()) {
    return TCoreType::VECTOR;
  }

  return TCoreType::CUBE_OR_VECTOR;
}

std::optional<TCoreType> StoreOp::inferCoreType() {
  // On 910B, fixpipe handles L0C to GM. Thus reaching here
  // means core type is VECTOR.
  return TCoreType::VECTOR;
}

// NOTICE: coretype inference for CopyOp never fail!
std::optional<TCoreType> CopyOp::inferCoreType() {
  if (hasPureTensorSemantics()) {
    return TCoreType::CUBE_OR_VECTOR;
  }

  MemRefType srcMemRefTy = dyn_cast<MemRefType>(getSrc().getType());
  MemRefType dstMemRefTy = dyn_cast<MemRefType>(getDst().getType());
  if (srcMemRefTy && dstMemRefTy) {
    auto fromAddrSpace =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(srcMemRefTy.getMemorySpace());
    auto toAddrSpace =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(dstMemRefTy.getMemorySpace());
    if (fromAddrSpace && toAddrSpace) {
      static const std::set<std::pair<hivm::AddressSpace, hivm::AddressSpace>>
          vectorTypeSet{
              std::pair{hivm::AddressSpace::GM, hivm::AddressSpace::UB},
              std::pair{hivm::AddressSpace::UB, hivm::AddressSpace::GM},
              std::pair{hivm::AddressSpace::UB, hivm::AddressSpace::UB},
              std::pair{hivm::AddressSpace::UB, hivm::AddressSpace::L1},
          };

      return vectorTypeSet.count(std::pair{fromAddrSpace.getAddressSpace(),
                                           toAddrSpace.getAddressSpace()})
                 ? TCoreType::VECTOR
                 : TCoreType::CUBE;
    }
  }

  return TCoreType::CUBE_OR_VECTOR;
}

//===----------------------------------------------------------------------===//
// HIVM Macro Ops
//===----------------------------------------------------------------------===//

std::optional<TCoreType> AtomicRMWOp::inferCoreType() { return TCoreType::VECTOR; }

std::optional<TCoreType> MatmulOp::inferCoreType() { return TCoreType::CUBE; }

std::optional<TCoreType> MixMatmulOp::inferCoreType() {
  return inferCoreTypeForGlobalMixMatmulOps<MixMatmulOp>(this);
}

std::optional<TCoreType> MixGroupMatmulOp::inferCoreType() {
  return inferCoreTypeForGlobalMixMatmulOps<MixGroupMatmulOp>(this);
}

std::optional<TCoreType> VBrcOp::inferCoreType() {
  Type dstType = this->getDst().getType();
  auto mayAddrSpace = getOptionalHIVMAddressSpace(dstType);
  if (!mayAddrSpace.has_value()) {
    // normally brc is vector op for L1 brc, it will only appear after load
    // decomposition and at that phase there is mem scope already
    return TCoreType::VECTOR;
  }

  if (mayAddrSpace.value() == hivm::AddressSpace::L1) {
    return TCoreType::CUBE;
  } else if (mayAddrSpace.value() == hivm::AddressSpace::UB) {
    return TCoreType::VECTOR;
  } else {
    llvm_unreachable("unsupport mem scope for vbrc");
  }
}
