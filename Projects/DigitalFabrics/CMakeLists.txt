file(GLOB HEADERS "*.h" "*.hpp")
file(GLOB SOURCES "*.c" "*.cpp")

add_executable(DigitalFabrics ${HEADERS} ${SOURCES})
target_link_libraries(DigitalFabrics PUBLIC Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::tetgen igl::triangle)