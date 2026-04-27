#ifndef ASCENDC_BISHENG_EVENT_UTILS_HPP
#define ASCENDC_BISHENG_EVENT_UTILS_HPP

#include "catlass/catlass.hpp"
#include "common_types.h"
#include "kernel_log.h"

namespace AscendCBisheng {

enum class Hardware : uint8_t { GM, UB, L1, L0A, L0B, L0C, BIAS, FIXBUF, MAX };

__aicore__ inline constexpr Hardware GetPhyType(TPosition pos)
{
    Hardware hard = Hardware::UB;
    if (pos == TPosition::GM) {
        hard = Hardware::GM;
    } else if (pos == TPosition::A1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::A2) {
        hard = Hardware::L0A;
    } else if (pos == TPosition::B1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::B2) {
        hard = Hardware::L0B;
    } else if (pos == TPosition::C1) {
        hard = Hardware::L1;
    } else if (pos == TPosition::C2) {
        hard = Hardware::BIAS;
    } else if (pos == TPosition::CO2) {
        hard = Hardware::GM;
    } else if (pos == TPosition::C2PIPE2GM) {
        hard = Hardware::FIXBUF;
    } else if (pos == TPosition::CO1) {
        hard = Hardware::L0C;
    } else if (pos == TPosition::SHM) {
        hard = Hardware::L1;
    } else if (pos == TPosition::TSCM) {
        hard = Hardware::L1;
    }
    return hard;
} // AscendCBisheng::Hardware

} // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_EVENT_UTILS_HPP
