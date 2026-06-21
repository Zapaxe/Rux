#pragma once

#include "Rux/Rcu.h"
#include <charconv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Rux {

// Type utilities (mirrored from Asm.cpp)
inline int SizeOf(const TypeRef& t) {
    switch (t.kind) {
    case TypeRef::Kind::Bool8:
    case TypeRef::Kind::Char8:
    case TypeRef::Kind::Int8:
    case TypeRef::Kind::UInt8:
        return 1;
    case TypeRef::Kind::Bool16:
    case TypeRef::Kind::Char16:
    case TypeRef::Kind::Int16:
    case TypeRef::Kind::UInt16:
        return 2;
    case TypeRef::Kind::Bool32:
    case TypeRef::Kind::Char32:
    case TypeRef::Kind::Int32:
    case TypeRef::Kind::UInt32:
    case TypeRef::Kind::Float32:
        return 4;
    case TypeRef::Kind::Opaque:
        return 0;
    case TypeRef::Kind::Tuple: {
        const auto alignUp = [](int v, int a) {
            return (v + a - 1) & ~(a - 1);
        };
        int offset = 0;
        int maxAlign = 1;
        for (const auto& elem : t.inner) {
            const int sz = SizeOf(elem);
            const int al = sz > 0 ? std::min(sz, 8) : 1;
            if (al > 1) {
                offset = alignUp(offset, al);
            }
            offset += sz > 0 ? sz : 8;
            maxAlign = std::max(maxAlign, al);
        }
        return alignUp(offset, maxAlign);
    }
    case TypeRef::Kind::Named:
        if (!t.inner.empty()) {
            return SizeOf(t.inner[0]);
        }
        return 8;
    default:
        return 8;
    }
}

inline bool IsFloat(const TypeRef& t) {
    return t.kind == TypeRef::Kind::Float32 || t.kind == TypeRef::Kind::Float64;
}

inline std::string_view NumericLiteralSuffix(std::string_view text) {
    static constexpr std::string_view suffixes[] = {
        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
        "f32", "f64", "i", "u",
    };
    for (const auto suffix : suffixes) {
        if (text.size() > suffix.size() &&
            text.substr(text.size() - suffix.size()) == suffix) {
            return suffix;
        }
    }
    return {};
}

inline std::optional<std::uint64_t> ParseIntegerLiteralBits(std::string_view text) {
    const std::string_view suffix = NumericLiteralSuffix(text);
    if (!suffix.empty()) {
        text.remove_suffix(suffix.size());
    }
    bool negative = false;
    if (!text.empty() && (text.front() == '-' || text.front() == '+')) {
        negative = text.front() == '-';
        text.remove_prefix(1);
    }
    std::string cleaned;
    cleaned.reserve(text.size());
    for (const char c : text) {
        if (c != '_') {
            cleaned.push_back(c);
        }
    }
    int base = 10;
    std::string_view digits(cleaned);
    if (digits.size() > 2 && digits[0] == '0') {
        switch (digits[1]) {
        case 'x': case 'X': base = 16; digits.remove_prefix(2); break;
        case 'b': case 'B': base = 2;  digits.remove_prefix(2); break;
        case 'o': case 'O': base = 8;  digits.remove_prefix(2); break;
        default: break;
        }
    }
    if (digits.empty()) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    const auto* first = digits.data();
    const auto* last = first + digits.size();
    const auto [ptr, ec] = std::from_chars(first, last, value, base);
    if (ec != std::errc{} || ptr != last) {
        return std::nullopt;
    }
    if (!negative) {
        return value;
    }
    constexpr std::uint64_t maxNegativeMagnitude =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1;
    if (value > maxNegativeMagnitude) {
        return std::nullopt;
    }
    return std::uint64_t{0} - value;
}

inline int AlignUp(int v, int a) {
    return (v + a - 1) & ~(a - 1);
}

// String table
class RcuStringTable {
public:
    RcuStringTable() { data_.push_back('\0'); }
    uint32_t Intern(const std::string& s) {
        if (s.empty()) return 0;
        auto it = map_.find(s);
        if (it != map_.end()) return it->second;
        const auto off = static_cast<uint32_t>(data_.size());
        map_[s] = off;
        data_.insert(data_.end(), s.begin(), s.end());
        data_.push_back('\0');
        return off;
    }
    [[nodiscard]] uint32_t Size() const { return static_cast<uint32_t>(data_.size()); }
    [[nodiscard]] const char* Data() const { return data_.data(); }
    [[nodiscard]] std::string Get(const uint32_t off) const {
        if (off >= data_.size()) return {};
        return {data_.data() + off};
    }
private:
    std::vector<char> data_;
    std::unordered_map<std::string, uint32_t> map_;
};

} // namespace Rux
