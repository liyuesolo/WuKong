include(CMakePrintHelpers)

set(DEPS tbb tbbmalloc tbbmalloc_proxy igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis)
set(TEST_DEPS tbb tbbmalloc tbbmalloc_proxy igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis gtest gtest_main)

file(GLOB HEADERS "include/*.h" "include/*.hpp" )
file(GLOB SOURCES "src/*.cpp"  "src/potentials/*.cpp")
add_executable(Foam2D ${HEADERS} ${SOURCES})

# file(GLOB TEST_HEADERS "include/*.h" "include/*.hpp" "tests/*.h" )
# file(GLOB TEST_SOURCES "src/*.cpp" )
file(GLOB TEST_HEADERS "include/*.h" "include/*.hpp" "tests/*.h" )
file(GLOB TEST_SOURCES "src/*.cpp" "tests/*.cpp" )
list(FILTER TEST_SOURCES EXCLUDE REGEX "^.*(/src/main.cpp)$")
cmake_print_variables(TEST_SOURCES)

add_executable(cellsim_test ${TEST_HEADERS} ${TEST_SOURCES})

target_compile_options(Foam2D PRIVATE  -Wall -Wreorder -Wextra)

target_link_libraries(Foam2D ${DEPS})
target_link_libraries(cellsim_test ${TEST_DEPS})
