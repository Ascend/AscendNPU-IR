//===- SwingOrder.h - Swing modulo scheduling order ------------*- C++ -*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include <vector>

namespace bishengir::hivm::sms {
struct OrderEntry {
  int nodeIdx;
  bool topDown;
};

std::vector<OrderEntry> buildScheduleOrder(const DDG &ddg,
                                           const std::vector<int> &asap,
                                           const std::vector<int> &mov,
                                           const std::vector<int> &depth,
                                           const std::vector<int> &height);
} // namespace bishengir::hivm::sms
