//===- AsapAlap.h - ASAP, ALAP, and mobility analyses ----------*- C++ -*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include <vector>

namespace bishengir::hivm::sms {
std::vector<int> topoOrder(const DDG &ddg);
std::vector<int> computeASAP(const DDG &ddg);
std::vector<int> computeALAP(const DDG &ddg, int horizon);
std::vector<int> computeMOV(const std::vector<int> &asap,
                            const std::vector<int> &alap);
} // namespace bishengir::hivm::sms
