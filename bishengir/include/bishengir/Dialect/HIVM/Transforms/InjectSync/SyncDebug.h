//===------------- SyncDebug.h ----Provide print syncIR -------------------===//
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
#ifndef BISHENGIR_SYNCDEBUG_H
#define BISHENGIR_SYNCDEBUG_H

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace hivm {

class SyncDebug {
public:
  SyncDebug(SyncIRs &syncIR) : syncIR(syncIR){};

  ~SyncDebug() = default;

  void PrintSyncIr();

private:
  SyncIRs &syncIR;

  /*! \brief The indentation level. */
  unsigned depth{0};

private:
  const unsigned kTabSize = 2;

  const DenseMap<hivm::PIPE, std::string> kPipeToName = {
      {hivm::PIPE::PIPE_S, "PIPE_S"},
      {hivm::PIPE::PIPE_V, "PIPE_V"},
      {hivm::PIPE::PIPE_M, "PIPE_M"},
      {hivm::PIPE::PIPE_MTE1, "PIPE_MTE1"},
      {hivm::PIPE::PIPE_MTE2, "PIPE_MTE2"},
      {hivm::PIPE::PIPE_MTE3, "PIPE_MTE3"},
      {hivm::PIPE::PIPE_FIX, "PIPE_FIX"},
      {hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A, "VIRTUAL_PIPE_MTE2_L1A"},
      {hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B, "VIRTUAL_PIPE_MTE2_L1B"},
  };

  const DenseMap<TCoreType, std::string> kCoreTypeToName = {
      {TCoreType::CUBE, "CoreType::CUBE"},
      {TCoreType::VECTOR, "CoreType::VECTOR"},
      {TCoreType::CUBE_OR_VECTOR, "CoreType::CUBE_OR_VECTOR"},
      {TCoreType::CUBE_AND_VECTOR, "CoreType::CUBE_AND_VECTOR"}};

  void PrintInstanceElement(const InstanceElement *e, raw_ostream &os);

  void PrintSyncOperation(const SyncOps &pipeType, raw_ostream &os);

  void PrintSync(const SyncOperation *s, raw_ostream &os);

  void PrintIndent(raw_ostream &os);

  void PrintPipeBarrierSync(const SyncOperation *s, raw_ostream &os);

  void PrintSetWaitSync(const SyncOperation *s, raw_ostream &os);

  void PrintCompoundIR(const InstanceElement *e, raw_ostream &os);

  void PrintForIR(const InstanceElement *e, raw_ostream &os);

  void PrintBranchIR(const InstanceElement *e, raw_ostream &os);

  std::string PrintSyncType(hivm::PIPE type) const;

  void printUnitFlag(const CompoundInstanceElement *e, raw_ostream &os) const;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_SYNCDEBUG_H
