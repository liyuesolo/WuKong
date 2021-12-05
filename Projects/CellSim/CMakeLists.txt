set(DEPS partio openvdb tbb tbbmalloc tbbmalloc_proxy cholmod_solver mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png)

file(GLOB HEADERS "include/*.h" "include/*.hpp")
file(GLOB SOURCES "*.c" "*.cpp" "src/*.cpp" "src/autodiff/*.cpp")


add_executable(CellSim ${HEADERS} ${SOURCES})
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)
target_include_directories(CellSim PUBLIC ../../Libs/spectra/include)
target_include_directories(CellSim PUBLIC /home/yueli/Documents/ETH/ipc-toolkit/build/include)

target_link_libraries(CellSim ${DEPS})