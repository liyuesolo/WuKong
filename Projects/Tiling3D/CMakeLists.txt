set(DEPS tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png)

file(GLOB HEADERS "*.h" "*.hpp" "IsohedraTiling/*.h" "tactile/.*h")
file(GLOB SOURCES "*.c" "*.cpp" "IsohedraTiling/*.cpp" "tactile/*.cpp")

add_executable(Tiling3D ${HEADERS} ${SOURCES})  
target_link_libraries(Tiling3D PUBLIC ${DEPS})
