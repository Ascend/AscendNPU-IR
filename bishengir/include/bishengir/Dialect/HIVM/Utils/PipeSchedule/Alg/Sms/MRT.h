//===- MRT.h - Modulo reservation table ------------------------*- C++ -*-===//
#pragma once

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/Alg/Sms/DDG.h"
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace bishengir::hivm::sms {
class MRT {
public:
  explicit MRT(int ii);

  int getII() const { return ii; }
  bool isParallel(const std::string &pipeline) const;
  bool isIntervalFree(int cycle, const std::string &pipeline,
                      int duration) const;
  void reserve(int cycle, const std::string &pipeline, int nodeIdx,
               int duration);
  void rebuild(int newII, const std::map<int, int> &scheduled, const DDG &ddg);

private:
  int ii;
  std::map<std::string, std::vector<int>> slots;
  std::map<std::string, std::vector<std::tuple<int, int, int>>> parallelLog;
};
} // namespace bishengir::hivm::sms
