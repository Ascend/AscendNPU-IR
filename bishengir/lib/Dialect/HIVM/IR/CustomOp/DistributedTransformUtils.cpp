//===- DistributedTransformUtils.cpp - Distributed HIVM op utilities ------===//
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

#include "bishengir/Dialect/HIVM/IR/CustomOp/DistributedTransformUtils.h"

namespace mlir {
namespace hivm {

const llvm::StringSet<> shmemIntp = {
    "dist.aclshmem_int64_p", "dist.aclshmem_int32_p", "dist.aclshmem_int16_p",
    "dist.aclshmem_int8_p"};

const std::string kShmemPtr = "aclshmem_ptr";
const std::string kShmemConsumeToken = "aclshmem_consume_token";

} // namespace hivm
} // namespace mlir
