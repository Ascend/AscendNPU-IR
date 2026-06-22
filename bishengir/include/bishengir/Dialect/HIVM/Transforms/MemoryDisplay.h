//===- MemoryDisplay.h ----Memory Display Tools ---------------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_MEMORY_DISPLAY_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_MEMORY_DISPLAY_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/PlanMemory.h"
#include "llvm/Support/JSON.h"

namespace mlir {
namespace hivm {

class MemoryInfo {
public:
  MemoryInfo(Location loc) : sourceLocation(loc) {}
  Value buffer{};
  SmallVector<uint64_t> offset{};
  uint64_t extent{};
  int64_t allocTimeInIR{};
  std::pair<int64_t, int64_t> lifeTimeInIR{};
  Location sourceLocation;
  bool isTmpBuf{};
};

class MemoryDisplayInfo {
public:
  std::string scope{};
  std::string status{};
  std::string errorInfo{};
  SmallVector<MemoryInfo> memoryInfoArray{};
};

// Memory visualization tool JSON generator
class MemoryInfoJsonBuilder {
private:
  llvm::json::Object root{};
  llvm::json::Array recordArray{};

public:
  MemoryInfoJsonBuilder() = default;

  void setHeader(const std::string &kernelName);

  void addRecord(const MemoryDisplayInfo &memoryDisplayInfo);

  std::string build();

  // write json to a local file
  bool writeToFile(const std::string &filename);

  // convert memory info to json
  llvm::json::Value convertMemoryInfoToJson(const MemoryInfo &memoryInfo);

  // convert smallvector to json
  llvm::json::Value convertSmallVectorToJson(const SmallVector<uint64_t> &vec);

  // convert mlir value to json
  llvm::json::Value convertValueToJson(mlir::Value value);

  // convert pair to json
  llvm::json::Value convertPairToJson(const std::pair<int64_t, int64_t> &value);

  // convert location to json
  llvm::json::Value convertLocationToJson(const mlir::Location &loc);
};

// collect the memory information required for memory display tools.
void collectMemoryInfoForDebug(
    SmallVector<MemoryDisplayInfo> &memoryDisplayInfoList,
    llvm::MapVector<hivm::AddressSpace, StorageEntry *> selectRootMap,
    DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets,
    DenseMap<hivm::AddressSpace, std::string> errorInfo, bool isFail);

// Generate a json file based on memory information.
void createJsonForMemoryDisplay(
    func::FuncOp funcOp, SmallVector<MemoryDisplayInfo> &memoryDisplayInfoList);
} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_MEMORY_DISPLAY_H
