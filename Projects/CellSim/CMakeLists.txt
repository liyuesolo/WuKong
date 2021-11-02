set(DEPS partio openvdb tbb tbbmalloc tbbmalloc_proxy cholmod_solver mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png)

file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB SOURCES "*.c" "*.cpp")


add_executable(CellSim ${HEADERS} ${SOURCES})
target_link_libraries(CellSim ${DEPS})