cmake_minimum_required(VERSION 3.14)
project(graphop VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────

# CGAL (header-only kernel + Surface_mesh_deformation)
find_package(CGAL REQUIRED)

# Eigen3 (required by CGAL's Eigen solver backend)
find_package(Eigen3 REQUIRED NO_MODULE)

# pybind11 (try cmake dir first, then system)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fall back to the Python package location
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED HINTS "${PYBIND11_CMAKE_DIR}")
endif()

# ── Demo executable (graphop_demo) ────────────────────────────────────────────
# Kept for quick sanity checks; does not depend on pybind11.
add_executable(graphop_demo graphop/main.cpp)
target_link_libraries(graphop_demo PRIVATE CGAL::CGAL Eigen3::Eigen)

# ── Shared library: deformation backend ───────────────────────────────────────
# Must be compiled with -fPIC so it can be linked into the Python extension (.so)
add_library(deformation_backend STATIC
    graphop/deformation.cpp
)
set_target_properties(deformation_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(deformation_backend PUBLIC graphop)
target_link_libraries(deformation_backend PUBLIC CGAL::CGAL Eigen3::Eigen)
target_compile_definitions(deformation_backend PUBLIC CGAL_EIGEN3_ENABLED)

# ── C++ test battery (no pybind11 dependency) ─────────────────────────────────
# Build:  make test_graphop
# Run:    ./test_graphop          (or: ctest --test-dir build -V)
#
# TEST_DATA_DIR is injected at compile time so the tests can always locate
# the sample meshes in data/ regardless of the working directory.
add_executable(test_graphop
    graphop/tests/test_deformation.cpp
)
target_include_directories(test_graphop PRIVATE graphop graphop/tests)
target_link_libraries(test_graphop PRIVATE deformation_backend)
target_compile_definitions(test_graphop PRIVATE
    CGAL_EIGEN3_ENABLED
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

enable_testing()
add_test(NAME graphop_tests COMMAND test_graphop)

# ── Python extension: graphop module ──────────────────────────────────────────
pybind11_add_module(graphop graphop/bindings.cpp)
target_link_libraries(graphop PRIVATE deformation_backend)
target_compile_definitions(graphop PRIVATE CGAL_EIGEN3_ENABLED)

# Install the extension next to the Python package so it can be imported
# without any additional PYTHONPATH manipulation:
#   build/ -> python3 -c "import graphop"
install(TARGETS graphop LIBRARY DESTINATION ${CMAKE_SOURCE_DIR})