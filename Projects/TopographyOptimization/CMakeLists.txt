set(DEPS spgrid ziran partio openvdb tbbmalloc tbbmalloc_proxy solver igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png)

file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB SOURCES "*.c" "*.cpp")

# add_library(TopographyOptimization STATIC ${HEADERS} ${SOURCES})
add_executable(TopographyOptimization ${HEADERS} ${SOURCES})
target_link_libraries(TopographyOptimization ${DEPS})