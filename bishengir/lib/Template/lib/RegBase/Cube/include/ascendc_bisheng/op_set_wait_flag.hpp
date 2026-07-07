#ifndef ASCENDC_BISHENG_OP_SET_WAIT_FLAG_HPP
#define ASCENDC_BISHENG_OP_SET_WAIT_FLAG_HPP

#include "kernel_log.h"
#include "__clang_cce_types.h"

namespace AscendCBisheng {

enum class HardEvent : uint8_t {
    // src_dst
    MTE2_MTE1,
    MTE1_MTE2,
    MTE1_M,
    M_MTE1,
    MTE2_V,
    V_MTE2,
    MTE3_V,
    V_MTE3,
    M_V,
    V_M,
    V_V,
    MTE3_MTE1,
    MTE1_MTE3,
    MTE1_V,
    MTE2_M,
    M_MTE2,
    V_MTE1,
    M_FIX,
    FIX_M,
    MTE3_MTE2,
    MTE2_MTE3,
    S_V,
    V_S,
    S_MTE2,
    MTE2_S,
    S_MTE3,
    MTE3_S,
    MTE2_FIX,
    FIX_MTE2,
    FIX_S,
    M_S,
    FIX_MTE3,
    MTE1_FIX,
    FIX_MTE1,
    FIX_FIX,
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 5102))
    FIX_V,
#endif
    MAX,
}; // HardEvent

template <pipe_t pipe>
__aicore__ inline constexpr bool IsSplitVectorPipe()
{
    return pipe == PIPE_S || pipe == PIPE_V || pipe == PIPE_MTE2 || pipe == PIPE_MTE3;
}

template <pipe_t pipe>
__aicore__ inline constexpr bool IsSplitCubePipe()
{
    return pipe == PIPE_S || pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_FIX || pipe == PIPE_M;
}

template <pipe_t srcPipe, pipe_t dstPipe>
__aicore__ inline void WaitFlagInternal(event_t evt)
{
    if constexpr (IsSplitCubePipe<srcPipe>() && IsSplitCubePipe<dstPipe>()){
        wait_flag(srcPipe, dstPipe, evt);
    }
}

__aicore__ inline void WaitFlagImpl(const HardEvent event, int32_t eventID)
{
    ASCENDC_ASSERT((eventID >= 0 && eventID < QUE_MAX_EVENT),
                   { KERNEL_LOG(KERNEL_ERROR, "eventID %d should be in range [0, %d)", eventID, QUE_MAX_EVENT); });
    event_t e = static_cast<event_t>(eventID);
    switch (event) {
        case HardEvent::MTE2_MTE1:
            WaitFlagInternal<PIPE_MTE2, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE1_MTE2:
            WaitFlagInternal<PIPE_MTE1, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE1_M:
            WaitFlagInternal<PIPE_MTE1, PIPE_M>(e);
            break;
        case HardEvent::M_MTE1:
            WaitFlagInternal<PIPE_M, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE3_MTE1:
            WaitFlagInternal<PIPE_MTE3, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE1_MTE3:
            WaitFlagInternal<PIPE_MTE1, PIPE_MTE3>(e);
            break;
        case HardEvent::MTE2_M:
            WaitFlagInternal<PIPE_MTE2, PIPE_M>(e);
            break;
        case HardEvent::M_MTE2:
            WaitFlagInternal<PIPE_M, PIPE_MTE2>(e);
            break;
        case HardEvent::FIX_M:
            WaitFlagInternal<PIPE_FIX, PIPE_M>(e);
            break;
        case HardEvent::M_FIX:
            WaitFlagInternal<PIPE_M, PIPE_FIX>(e);
            break;
        case HardEvent::MTE2_FIX:
            WaitFlagInternal<PIPE_MTE2, PIPE_FIX>(e);
            break;
        case HardEvent::FIX_MTE2:
            WaitFlagInternal<PIPE_FIX, PIPE_MTE2>(e);
            break;
        case HardEvent::FIX_S:
            WaitFlagInternal<PIPE_FIX, PIPE_S>(e);
            break;
        case HardEvent::FIX_MTE3:
            WaitFlagInternal<PIPE_FIX, PIPE_MTE3>(e);
            break;
        case HardEvent::MTE1_FIX:
            WaitFlagInternal<PIPE_MTE1, PIPE_FIX>(e);
            break;
        case HardEvent::FIX_MTE1:
            WaitFlagInternal<PIPE_FIX, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE3_MTE2:
            WaitFlagInternal<PIPE_MTE3, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE2_MTE3:
            WaitFlagInternal<PIPE_MTE2, PIPE_MTE3>(e);
            break;
        case HardEvent::S_MTE2:
            WaitFlagInternal<PIPE_S, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE2_S:
            WaitFlagInternal<PIPE_MTE2, PIPE_S>(e);
            break;
        case HardEvent::S_MTE3:
            WaitFlagInternal<PIPE_S, PIPE_MTE3>(e);
            break;
        case HardEvent::MTE3_S:
            WaitFlagInternal<PIPE_MTE3, PIPE_S>(e);
            break;
        case HardEvent::M_S:
            WaitFlagInternal<PIPE_M, PIPE_S>(e);
            break;
        case HardEvent::S_V:
            WaitFlagInternal<PIPE_S, PIPE_V>(e);
            break;
        case HardEvent::V_S:
            WaitFlagInternal<PIPE_V, PIPE_S>(e);
            break;
        case HardEvent::V_V:
            return;
        case HardEvent::MAX:
            break;
        default:
            break;
    }
    return;
}

template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
{
    if constexpr (event == HardEvent::MTE2_V || event == HardEvent::V_MTE2 || event == HardEvent::MTE3_V
                  || event == HardEvent::V_MTE3 || event == HardEvent::V_V || event == HardEvent::S_V ||
                  event == HardEvent::V_S || event == HardEvent::MTE2_MTE3 || event == HardEvent::MTE3_MTE2
                  || event == HardEvent::MTE3_S || event == HardEvent::S_MTE3) {
        return;
    }
    WaitFlagImpl(event, eventID);
}

template <pipe_t srcPipe, pipe_t dstPipe>
__aicore__ inline void SetFlagInternal(event_t evt)
{
    if constexpr (IsSplitCubePipe<srcPipe>() && IsSplitCubePipe<dstPipe>()){
        set_flag(srcPipe, dstPipe, evt);
    }
}

template <HardEvent event>
__aicore__ inline void SetFlagImpl(int32_t eventID)
{
    ASCENDC_ASSERT((eventID >= 0 && eventID < QUE_MAX_EVENT),
                   { KERNEL_LOG(KERNEL_ERROR, "eventID %d should be in range [0, %d)", eventID, QUE_MAX_EVENT); });
    event_t e = static_cast<event_t>(eventID);
    switch (event) {
        case HardEvent::MTE2_MTE1:
            SetFlagInternal<PIPE_MTE2, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE1_MTE2:
            SetFlagInternal<PIPE_MTE1, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE2_MTE3:
            SetFlagInternal<PIPE_MTE2, PIPE_MTE3>(e);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlagInternal<PIPE_MTE3, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE1_M:
            SetFlagInternal<PIPE_MTE1, PIPE_M>(e);
            break;
        case HardEvent::M_MTE1:
            SetFlagInternal<PIPE_M, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE2_V:
            SetFlagInternal<PIPE_MTE2, PIPE_V>(e);
            break;
        case HardEvent::V_MTE2:
            SetFlagInternal<PIPE_V, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE3_V:
            SetFlagInternal<PIPE_MTE3, PIPE_V>(e);
            break;
        case HardEvent::V_MTE3:
            SetFlagInternal<PIPE_V, PIPE_MTE3>(e);
            break;
        case HardEvent::M_V:
            SetFlagInternal<PIPE_M, PIPE_V>(e);
            break;
        case HardEvent::M_S:
            SetFlagInternal<PIPE_M, PIPE_S>(e);
            break;
        case HardEvent::V_M:
            SetFlagInternal<PIPE_V, PIPE_M>(e);
            break;
        case HardEvent::S_V:
            SetFlagInternal<PIPE_S, PIPE_V>(e);
            break;
        case HardEvent::V_S:
            SetFlagInternal<PIPE_V, PIPE_S>(e);
            break;
        case HardEvent::MTE3_MTE1:
            SetFlagInternal<PIPE_MTE3, PIPE_MTE1>(e);
            break;
        case HardEvent::MTE1_MTE3:
            SetFlagInternal<PIPE_MTE1, PIPE_MTE3>(e);
            break;
        case HardEvent::MTE1_V:
            SetFlagInternal<PIPE_MTE1, PIPE_V>(e);
            break;
        case HardEvent::MTE2_M:
            SetFlagInternal<PIPE_MTE2, PIPE_M>(e);
            break;
        case HardEvent::M_MTE2:
            SetFlagInternal<PIPE_M, PIPE_MTE2>(e);
            break;
        case HardEvent::S_MTE2:
            SetFlagInternal<PIPE_S, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE2_S:
            SetFlagInternal<PIPE_MTE2, PIPE_S>(e);
            break;
        case HardEvent::V_MTE1:
            SetFlagInternal<PIPE_V, PIPE_MTE1>(e);
            break;
        case HardEvent::S_MTE3:
            SetFlagInternal<PIPE_S, PIPE_MTE3>(e);
            break;
        case HardEvent::MTE3_S:
            SetFlagInternal<PIPE_MTE3, PIPE_S>(e);
            break;
        case HardEvent::M_FIX:
            SetFlagInternal<PIPE_M, PIPE_FIX>(e);
            break;
        case HardEvent::FIX_M:
            SetFlagInternal<PIPE_FIX, PIPE_M>(e);
            break;
        case HardEvent::FIX_MTE3:
            SetFlagInternal<PIPE_FIX, PIPE_MTE3>(e);
            break;
        case HardEvent::FIX_MTE2:
            SetFlagInternal<PIPE_FIX, PIPE_MTE2>(e);
            break;
        case HardEvent::MTE2_FIX:
            SetFlagInternal<PIPE_MTE2, PIPE_FIX>(e);
            break;
        case HardEvent::FIX_S:
            SetFlagInternal<PIPE_FIX, PIPE_S>(e);
            break;
        case HardEvent::MTE1_FIX:
            SetFlagInternal<PIPE_MTE1, PIPE_FIX>(e);
            break;
        case HardEvent::FIX_MTE1:
            SetFlagInternal<PIPE_FIX, PIPE_MTE1>(e);
            break;
        case HardEvent::MAX:
            break;
        default:
            ASCENDC_ASSERT((0), { KERNEL_LOG(KERNEL_ERROR, "invalid event %d", static_cast<int32_t>(event)); });
            break;
    }
}

template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
{
    if constexpr (event == HardEvent::MTE2_V || event == HardEvent::V_MTE2 || event == HardEvent::MTE3_V
                  || event == HardEvent::V_MTE3 || event == HardEvent::V_V || event == HardEvent::S_V ||
                  event == HardEvent::V_S || event == HardEvent::MTE2_MTE3 || event == HardEvent::MTE3_MTE2
                  || event == HardEvent::MTE3_S || event == HardEvent::S_MTE3) {
        return;
    }
    SetFlagImpl<event>(eventID);
}

} // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_OP_SET_WAIT_FLAG_HPP
