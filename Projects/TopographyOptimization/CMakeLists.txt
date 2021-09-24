set(DEPS partio openvdb tbb tbbmalloc tbbmalloc_proxy cholmod_solver igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png)

file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB SOURCES "*.c" "*.cpp")


add_executable(TopographyOptimization ${HEADERS} ${SOURCES})
target_link_libraries(TopographyOptimization ${DEPS})