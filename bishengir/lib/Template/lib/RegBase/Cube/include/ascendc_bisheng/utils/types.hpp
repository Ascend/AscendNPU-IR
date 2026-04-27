#ifndef ASCENDC_BISHENG_UTILS_TYPES_HPP
#define ASCENDC_BISHENG_UTILS_TYPES_HPP

namespace AscendCBisheng {

template <bool b> struct BoolInst {
    using Type = BoolInst<b>;
    static constexpr bool value = b;
};

using TrueType = BoolInst<true>;
using FalseType = BoolInst<false>;

template <typename T, typename U> struct IsSameType : public FalseType {};

template <typename T> struct IsSameType<T, T> : public TrueType {};

template <typename T, typename U, typename... Args>
__aicore__ constexpr bool SupportType()
{
    if constexpr (sizeof...(Args) > 0) {
        return IsSameType<T, U>::value || SupportType<T, Args...>();
    }
    return IsSameType<T, U>::value;
}

} // namespace AscendCBisheng

#endif // ASCENDC_BISHENG_UTILS_TYPES_HPP
