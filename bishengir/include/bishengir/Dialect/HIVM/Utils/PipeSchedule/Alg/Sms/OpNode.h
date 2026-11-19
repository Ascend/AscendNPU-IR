#pragma once
#include <string>

namespace bishengir::hivm::sms {
struct OpNode {
  int ordinal = 0;
  std::string name;
  std::string pipeline;
  int latency = 1;
  int issueTime = 1;
  bool isParallelUnit = false;
};
} // namespace bishengir::hivm::sms
