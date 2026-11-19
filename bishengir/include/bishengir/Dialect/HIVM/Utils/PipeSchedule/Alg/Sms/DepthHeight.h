//===- DepthHeight.h - DDG depth and height analyses ------------*- C++ -*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include <vector>

namespace bishengir::hivm::sms {
std::vector<int> computeDepth(const DDG &ddg);
std::vector<int> computeHeight(const DDG &ddg);
} // namespace bishengir::hivm::sms
