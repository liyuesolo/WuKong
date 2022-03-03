set(DEPS tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::tetgen igl::triangle igl::png mkl_intel_lp64 mkl_sequential mkl_core cholmod_solver)

file(GLOB HEADERS "include/*.h" "include/autodiff/*.h")
file(GLOB SOURCES "src/*.cpp" "src/autodiff/*.cpp")

add_executable(FEM ${HEADERS} ${SOURCES})  
target_link_libraries(FEM /home/yueli/Documents/ETH/ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(FEM /home/yueli/Documents/ETH/ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(FEM /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(FEM /home/yueli/Documents/ETH/ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(FEM /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)

target_include_directories(FEM PUBLIC ../../Libs/spectra/include)
target_include_directories(FEM PUBLIC /home/yueli/Documents/ETH/ipc-toolkit/build/include)
target_link_libraries(FEM ${DEPS})
