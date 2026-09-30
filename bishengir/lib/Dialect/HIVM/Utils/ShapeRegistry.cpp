//===- ShapeRegistry.cpp - Loop-shape registry fingerprints ---------------===//
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

#include "bishengir/Dialect/HIVM/Utils/ShapeRegistry.h"

#include <cstddef>
#include <cstdint>

#include "SkewShapeTable.inc"

using llvm::StringRef;

namespace {
constexpr uint64_t kSpanSaltSeed = 0x9E3779B97F4A7C15ULL;

constexpr uint64_t advanceSpanSalt(uint64_t s) {
  return s * 6364136223846793005ULL + 1442695040888963407ULL;
}
} // namespace

bool mlir::hivm::isLoopShapeRegistered(StringRef name) {
  if (name.empty() || kSpanKeyCount == 0)
    return false;
  uint64_t salt[kSpanKeyCount] = {};
  uint64_t s = kSpanSaltSeed;
  for (unsigned k = 0; k < kSpanKeyCount; ++k) {
    s = advanceSpanSalt(s);
    salt[k] = s;
  }
  // Fingerprint every substring so decorated symbols resolve to the same shape.
  for (size_t i = 0, n = name.size(); i < n; ++i) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t j = i; j < n; ++j) {
      h ^= static_cast<unsigned char>(name[j]);
      h *= 0x100000001b3ULL;
      for (unsigned k = 0; k < kSpanKeyCount; ++k)
        if ((h ^ salt[k]) == kSpanKeys[k])
          return true;
    }
  }
  return false;
}
