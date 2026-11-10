//===- DagVerifier.h - SMS DAG validation ----------------------*- C++ -*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include <string>

namespace bishengir::hivm::sms {
struct VerifyResult {
  bool ok;
  std::string error;
};

VerifyResult verifyDag(const DDG &ddg);
} // namespace bishengir::hivm::sms
