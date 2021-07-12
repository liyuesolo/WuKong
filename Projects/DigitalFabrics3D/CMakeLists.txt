set(DEPS tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png)

file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB SOURCES "*.c" "*.cpp")

add_executable(DigitalFabrics3D ${HEADERS} ${SOURCES})
target_link_libraries(DigitalFabrics3D PUBLIC ${DEPS})
