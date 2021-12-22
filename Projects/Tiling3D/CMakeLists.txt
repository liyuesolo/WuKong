set(DEPS tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png mkl_intel_lp64 mkl_sequential mkl_core)

file(GLOB HEADERS "include/*.h" "tactile/.*h" "tactile/.*hpp")
file(GLOB SOURCES "src/*.cpp" "tactile/*.cpp")

add_executable(Tiling3D ${HEADERS} ${SOURCES})  
target_link_libraries(Tiling3D ${DEPS})
