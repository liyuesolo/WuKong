set(DEPS partio openvdb tbb tbbmalloc tbbmalloc_proxy cholmod_solver mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png amgcl VexCL::CUDA)

file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB SOURCES "*.c" "*.cpp")


add_executable(TopographyOptimization ${HEADERS} ${SOURCES})
target_include_directories(TopographyOptimization PUBLIC ../../Libs/spectra/include)
target_include_directories(TopographyOptimization PUBLIC ../../build/amgcl-src/)
target_link_libraries(TopographyOptimization ${DEPS})