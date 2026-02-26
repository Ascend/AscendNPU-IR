//===- GetOperandsTargetLayout.cpp - get operands target layout impls -----===//
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
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::hivm;

namespace mlir::hivm {
llvm::SmallVector<int64_t> getBlockSizes(mlir::Value oper) {
  llvm::SmallVector<int64_t> kBlockSizes;
  auto elementType = getElementTypeOrSelf(oper.getType());
  size_t kBlockSize =
      utils::INTR_BYTES_PER_BLOCK /
      (elementType.getIntOrFloatBitWidth() / utils::kBitsToByte);
  kBlockSizes.push_back(utils::FRACTAL_BLOCK_NUM);
  kBlockSizes.push_back(kBlockSize);
  return kBlockSizes;
}
} // namespace mlir::hivm

//===----------------------------------------------------------------------===//
// MmadL1Op
//===----------------------------------------------------------------------===//

llvm::SmallDenseMap<Value, DataLayoutAttr>
MmadL1Op::getOperandsCurrentLayout() {
  llvm::SmallDenseMap<Value, DataLayoutAttr> valLayoutMap;

  auto aLayoutAttr = getOperandALayout();
  assert(succeeded(aLayoutAttr) && "Cannot get layout for Matrix A");
  valLayoutMap[getDpsInputOperand(0)->get()] = *aLayoutAttr;

  auto bLayoutAttr = getOperandBLayout();
  assert(succeeded(bLayoutAttr) && "Cannot get layout for Matrix B");
  valLayoutMap[getDpsInputOperand(1)->get()] = *bLayoutAttr;

  auto cLayoutAttr = getOperandCLayout();
  assert(succeeded(cLayoutAttr) && "Cannot get layout for Matrix C");
  valLayoutMap[getDpsInitOperand(0)->get()] = *cLayoutAttr;

  if (getPerChannelBias()) {
    auto biasLayoutAttr = getOperandBiasLayout();
    assert(succeeded(biasLayoutAttr) && "Cannot get layout for bias");
    valLayoutMap[getDpsInputOperand(getNumDpsInputs() - 1)->get()] =
        *biasLayoutAttr;
  }
  return valLayoutMap;
}

llvm::SmallDenseMap<Value, DataLayoutAttr> MmadL1Op::getOperandsTargetLayout() {
  llvm::SmallDenseMap<Value, DataLayoutAttr> valLayoutMap;

  auto operA = getA();
  bool isATranspose = getATranspose().has_value();
  auto aBlockSizes = getBlockSizes(operA);
  auto mALayoutAttr = DataLayoutAttr::get(
      getContext(), isATranspose ? DataLayout::nZ : DataLayout::zN,
      nullptr,
      mlir::DenseI64ArrayAttr::get(getContext(), ArrayRef(aBlockSizes)));
  valLayoutMap[operA] = mALayoutAttr;

  auto operB = getB();
  bool isBTranspose = getBTranspose().has_value();
  auto bBlockSizes = getBlockSizes(operB);
  auto mBLayoutAttr = DataLayoutAttr::get(
      getContext(), isBTranspose ? DataLayout::nZ : DataLayout::zN,
      nullptr,
      mlir::DenseI64ArrayAttr::get(getContext(), ArrayRef(bBlockSizes)));
  valLayoutMap[operB] = mBLayoutAttr;

  llvm::SmallVector<int64_t> cBlockSizes;
  cBlockSizes.push_back(utils::FRACTAL_BLOCK_NUM);
  cBlockSizes.push_back(utils::FRACTAL_BLOCK_NUM);
  auto mCLayoutAttr = DataLayoutAttr::get(
      getContext(), DataLayout::zN, nullptr,
      mlir::DenseI64ArrayAttr::get(getContext(), ArrayRef(cBlockSizes)));
  valLayoutMap[getC()] = mCLayoutAttr;

  if (auto bias = getPerChannelBias()) {
    auto biasLayoutAttr = DataLayoutAttr::get(getContext(), DataLayout::ND,
                                              nullptr, nullptr);
    valLayoutMap[bias] = biasLayoutAttr;
  }
  return valLayoutMap;
}