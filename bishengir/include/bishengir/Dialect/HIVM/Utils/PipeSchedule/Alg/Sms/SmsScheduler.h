//===- SmsScheduler.h - Swing modulo scheduler -----------------*- C++ -*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include <map>
#include <optional>

namespace bishengir::hivm::sms {
struct ScheduleResult {
  int ii;
  std::map<int, int> nodeToCycle;
  std::map<int, int> nodeToStage;
  int stageCount;
};

std::optional<ScheduleResult> smsSchedule(const DDG &ddg, int maxIISearch = 20);
} // namespace bishengir::hivm::sms
