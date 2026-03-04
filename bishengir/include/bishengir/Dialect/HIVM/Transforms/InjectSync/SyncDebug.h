//===------------- SyncDebug.h ----Provide print syncIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  void printPlaceHolderIR(const InstanceElement *e, raw_ostream &os);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_SYNCDEBUG_H
