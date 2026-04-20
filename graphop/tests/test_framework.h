/**
 * test_framework.h
 *
 * Minimal, zero-dependency C++ test framework for graphop.
 *
 * Usage
 * -----
 *
 *   #include "test_framework.h"
 *
 *   void test_something() {
 *       EXPECT_TRUE(1 + 1 == 2);
 *       EXPECT_EQ(42, 42);
 *       EXPECT_NEAR(3.14, M_PI, 0.01);
 *       EXPECT_THROWS(throw std::runtime_error("boom"));
 *       EXPECT_NO_THROW(int x = 1 + 1; (void)x);
 *   }
 *
 *   int main() {
 *       RUN_TEST(test_something);
 *       return test::summary();   // returns 0 if all pass, nonzero otherwise
 *   }
 *
 * Design goals
 * ------------
 *  - No external libraries (no gtest / Catch2).
 *  - Each check throws std::runtime_error on failure; RUN_TEST() catches it.
 *  - summary() prints a final tally and returns the failure count (suitable
 *    as the main() return value and as a CTest exit code).
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

// ── Global pass/fail counters ─────────────────────────────────────────────────

inline int& pass_count() { static int n = 0; return n; }
inline int& fail_count() { static int n = 0; return n; }

// ── summary ───────────────────────────────────────────────────────────────────

/**
 * Print a final tally line and return the failure count.
 * Use as the return value of main():
 *
 *   return test::summary();
 */
inline int summary()
{
    int total = pass_count() + fail_count();
    std::cout << "\n──────────────────────────────────────\n";
    if (fail_count() == 0) {
        std::cout << "  ALL " << total << " tests PASSED\n";
    } else {
        std::cout << "  " << pass_count() << "/" << total
                  << " tests passed,  "
                  << fail_count() << " FAILED\n";
    }
    std::cout << "──────────────────────────────────────\n";
    return fail_count();
}

// ── Internal check primitive ──────────────────────────────────────────────────

inline void _check(bool cond,
                   const char* expr,
                   const char* file,
                   int         line)
{
    if (!cond) {
        std::ostringstream oss;
        oss << "assertion failed: " << expr
            << "  (" << file << ":" << line << ")";
        throw std::runtime_error(oss.str());
    }
}

} // namespace test

// ── Public macros ─────────────────────────────────────────────────────────────

/**
 * RUN_TEST(fn)
 * Run a void() test function, catch any exception, and record pass/fail.
 */
#define RUN_TEST(fn)                                                  \
    do {                                                              \
        std::cout << "  [ RUN  ] " #fn "\n";                         \
        try {                                                         \
            fn();                                                     \
            std::cout << "  [ PASS ] " #fn "\n";                     \
            ::test::pass_count()++;                                   \
        } catch (const std::exception& _e) {                         \
            std::cout << "  [ FAIL ] " #fn "\n"                      \
                      << "           " << _e.what() << "\n";         \
            ::test::fail_count()++;                                   \
        } catch (...) {                                               \
            std::cout << "  [ FAIL ] " #fn "\n"                      \
                      << "           (unknown exception)\n";         \
            ::test::fail_count()++;                                   \
        }                                                             \
    } while (0)

/** Assert that condition is true. */
#define EXPECT_TRUE(cond) \
    ::test::_check((cond), #cond, __FILE__, __LINE__)

/** Assert that condition is false. */
#define EXPECT_FALSE(cond) \
    ::test::_check(!(cond), "!" #cond, __FILE__, __LINE__)

/** Assert a == b (uses operator==). */
#define EXPECT_EQ(a, b) \
    ::test::_check((a) == (b), #a " == " #b, __FILE__, __LINE__)

/** Assert a != b. */
#define EXPECT_NE(a, b) \
    ::test::_check((a) != (b), #a " != " #b, __FILE__, __LINE__)

/** Assert |a - b| <= tol. */
#define EXPECT_NEAR(a, b, tol)                                           \
    ::test::_check(std::abs(double(a) - double(b)) <= double(tol),       \
                   "|" #a " - " #b "| <= " #tol, __FILE__, __LINE__)

/** Assert that expr throws any exception. */
#define EXPECT_THROWS(expr)                                              \
    do {                                                                 \
        bool _threw = false;                                             \
        try { (void)(expr); } catch (...) { _threw = true; }            \
        ::test::_check(_threw, "throws: " #expr, __FILE__, __LINE__);   \
    } while (0)

/** Assert that expr does NOT throw. */
#define EXPECT_NO_THROW(expr)                                            \
    do {                                                                 \
        bool _threw = false;                                             \
        try { (void)(expr); } catch (...) { _threw = true; }            \
        ::test::_check(!_threw, "no throw: " #expr, __FILE__, __LINE__);\
    } while (0)