//===- ResMII.h - Resource minimum initiation interval -----------*- C++
//-*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"

namespace bishengir::hivm::sms {
int computeResMII(const DDG &ddg);
} // namespace bishengir::hivm::sms
