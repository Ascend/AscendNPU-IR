//===- MemoryDisplay.cpp ----Memory Display Tools -------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/MemoryDisplay.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

namespace mlir {
namespace hivm {

void MemoryInfoJsonBuilder::setHeader(const std::string &kernelName) {
  llvm::json::Object header;
  header["KernelName"] = kernelName;
  root["Header"] = std::move(header);
}

void MemoryInfoJsonBuilder::addRecord(
    const MemoryDisplayInfo &memoryDisplayInfo) {
  llvm::json::Object record;
  record["scope"] = memoryDisplayInfo.scope;
  record["status"] = memoryDisplayInfo.status;
  record["error_info"] = memoryDisplayInfo.errorInfo;

  llvm::json::Array memoryInfoArray;
  for (const auto &info : memoryDisplayInfo.memoryInfoArray) {
    memoryInfoArray.push_back(convertMemoryInfoToJson(info));
  }
  record["memory_info_array"] = std::move(memoryInfoArray);
  recordArray.push_back(std::move(record));
}

llvm::json::Value
MemoryInfoJsonBuilder::convertMemoryInfoToJson(const MemoryInfo &memoryInfo) {
  llvm::json::Object record;
  record["buffer"] = convertValueToJson(memoryInfo.buffer);
  record["offset"] = convertSmallVectorToJson(memoryInfo.offset);
  record["extent"] = memoryInfo.extent;
  record["alloc_time_in_ir"] = memoryInfo.allocTimeInIR;
  record["life_time_in_ir"] = convertPairToJson(memoryInfo.lifeTimeInIR);
  record["source_location"] = convertLocationToJson(memoryInfo.sourceLocation);
  record["is_tmpbuf"] = memoryInfo.isTmpBuf;
  return record;
}

llvm::json::Value MemoryInfoJsonBuilder::convertSmallVectorToJson(
    const SmallVector<uint64_t> &vec) {
  llvm::json::Array jsonArray;
  for (uint64_t value : vec) {
    jsonArray.push_back(llvm::json::Value(static_cast<int64_t>(value)));
  }
  return jsonArray;
}

llvm::json::Value MemoryInfoJsonBuilder::convertValueToJson(mlir::Value value) {
  std::string printBuffer;
  llvm::raw_string_ostream os(printBuffer);
  value.print(os);
  return os.str();
}

llvm::json::Value MemoryInfoJsonBuilder::convertPairToJson(
    const std::pair<int64_t, int64_t> &value) {
  llvm::json::Array array;
  array.push_back(value.first);
  array.push_back(value.second);
  return llvm::json::Value(std::move(array));
}

llvm::json::Value
MemoryInfoJsonBuilder::convertLocationToJson(const mlir::Location &loc) {
  std::string locStr;
  llvm::raw_string_ostream os(locStr);
  loc.print(os);
  os.flush();
  return llvm::json::Value(locStr);
}

std::string MemoryInfoJsonBuilder::build() {
  root["Record"] = std::move(recordArray);
  llvm::json::Value jsonValue(std::move(root));
  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << llvm::formatv("{0:2}", jsonValue);
  return os.str();
}

bool MemoryInfoJsonBuilder::writeToFile(const std::string &filename) {
  std::string jsonContent = build();
  std::error_code errorCode;
  llvm::raw_fd_ostream os(filename, errorCode);
  if (errorCode) {
    llvm::errs() << "Failed to open file: " << filename << " - "
                 << errorCode.message() << "\n";
    return false;
  }
  os << jsonContent;
  return true;
}

void createJsonForMemoryDisplay(
    func::FuncOp funcOp,
    SmallVector<MemoryDisplayInfo> &memoryDisplayInfoList) {
  MemoryInfoJsonBuilder builder;
  builder.setHeader(funcOp.getName().str());
  for (auto info : memoryDisplayInfoList) {
    builder.addRecord(info);
  }
  std::string jsonPath = "./memory_info.json";
  std::optional<TFuncCoreType> funcCoreType = queryFuncCoreType(funcOp);
  if (funcCoreType.has_value()) {
    if (funcCoreType.value() == TFuncCoreType::AIC) {
      jsonPath = "./memory_info_aic.json";
    } else if (funcCoreType.value() == TFuncCoreType::AIV) {
      jsonPath = "./memory_info_aiv.json";
    }
  }
  builder.writeToFile(jsonPath);
}

MemoryInfo
generateMemoryInfo(const StorageEntry *storageEntry,
                   DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets,
                   Operation *oper) {
  Location loc = oper->getLoc();
  Value value = oper->getResult(0);
  MemoryInfo memoryInfo = MemoryInfo(loc);
  memoryInfo.buffer = value;
  memoryInfo.offset = buffer2Offsets[value];
  assert(storageEntry->bufInfo != nullptr);
  memoryInfo.extent =
      AlignUp(storageEntry->bufInfo->constBits, utils::kBitsToByte);
  assert(storageEntry->bufferLifeVec.size() == 1 &&
         "The final bufferlife corresponding to the buffer should be 1");
  assert(storageEntry->bufferLifeVec[0] != nullptr);
  memoryInfo.lifeTimeInIR =
      std::make_pair(storageEntry->bufferLifeVec[0]->allocTime,
                     storageEntry->bufferLifeVec[0]->freeTime);
  memoryInfo.allocTimeInIR = storageEntry->bufferLifeVec[0]->allocTime;
  memoryInfo.isTmpBuf = oper->hasAttr("is_tmpbuf") ? true : false;
  return memoryInfo;
}

void collectMemoryInfoForDebug(
    SmallVector<MemoryDisplayInfo> &memoryDisplayInfoList,
    DenseMap<hivm::AddressSpace, StorageEntry *> selectRootMap,
    DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets,
    DenseMap<hivm::AddressSpace, std::string> errorInfo, bool isFail) {
  for (auto &it : selectRootMap) {
    MemoryDisplayInfo memoryDisplayInfo;
    memoryDisplayInfo.scope = stringifyEnum(it.first);
    memoryDisplayInfo.status = isFail ? "fail" : "success";
    if (isFail) {
      memoryDisplayInfo.errorInfo = errorInfo[it.first];
    }
    auto rootStorageEntry = it.second;
    assert(rootStorageEntry && "Root StorageEntry should not be null");
    assert(rootStorageEntry->bufInfo != nullptr);
    Operation *rootOper = rootStorageEntry->bufInfo->operation;
    MemoryInfo memoryRootInfo =
        generateMemoryInfo(rootStorageEntry, buffer2Offsets, rootOper);
    memoryDisplayInfo.memoryInfoArray.emplace_back(memoryRootInfo);
    uint64_t needByte =
        (rootStorageEntry->alignedConstBits + utils::kBitsToByte - 1) /
        utils::kBitsToByte;
    uint64_t offsetByte =
        (rootStorageEntry->bitsOffset + utils::kBitsToByte - 1) /
        utils::kBitsToByte;
    for (auto &childrenStorageEntry : rootStorageEntry->mergedChildren) {
      for (auto &buffer : childrenStorageEntry->inplaceBuffers) {
        if (buffer.getDefiningOp()) {
          if (auto allocOp =
               dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
            MemoryInfo memoryLeafInfo = generateMemoryInfo(
                childrenStorageEntry, buffer2Offsets, allocOp);
            memoryDisplayInfo.memoryInfoArray.emplace_back(memoryLeafInfo);
          }
        }
      }
    }
    memoryDisplayInfoList.emplace_back(memoryDisplayInfo);
  }
}

} // namespace hivm
} // namespace mlir