#ifndef DMT_RULES_H
#define DMT_RULES_H

#include <unordered_map>

template <typename T=Parents::type>
using AreaRules = std::unordered_map<T, T>;

template <typename T, typename U=Parents::type>
using Root = std::pair<T, U>;

template <typename T, typename U=Parents::type>
using RootRules = std::unordered_map<U, Root<T, U>>;

#endif //DMT_RULES_H
