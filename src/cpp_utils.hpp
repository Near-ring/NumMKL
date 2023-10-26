//
// Created by Semigroup on 2023/10/26.
//

#ifndef NUMMKL_CPP_UTILS_HPP
#define NUMMKL_CPP_UTILS_HPP

#include <chrono>
#include <iostream>

template <typename Func>
auto measure_exec_time_ms(Func lambda)
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    lambda();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    return duration.count();
}

void print() {
    std::cout << "\n";  // print a newline when we reach the end
}

template <typename T, typename... Args>
void print(const T& str, const Args&... args) {
    std::cout << str << " ";
    print(args...);
}

#endif // NUMMKL_CPP_UTILS_HPP
