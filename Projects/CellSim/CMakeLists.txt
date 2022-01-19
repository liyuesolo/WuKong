set(DEPS partio openvdb tbb tbbmalloc tbbmalloc_proxy mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis cholmod amd camd ccolamd colamd spqr)

file(GLOB HEADERS "include/*.h" "include/*.hpp" "solver/*.hpp")
file(GLOB SOURCES "*.c" "*.cpp" "src/*.cpp" "src/autodiff/*.cpp" "solver/*.cpp")


add_executable(CellSim ${HEADERS} ${SOURCES})

set(SuiteSparse_ROOT "/home/yueli/Documents/SuiteSparse-5.9.0")
include_directories(${SuiteSparse_ROOT})
include_directories(${SuiteSparse_ROOT}/include)
link_directories(${SuiteSparse_ROOT}/lib)
# target_link_libraries(CellSim suitesparseconfig metis cholmod amd camd ccolamd colamd spqr)

target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)
target_include_directories(CellSim PUBLIC ../../Libs/spectra/include)
target_include_directories(CellSim PUBLIC /home/yueli/Documents/ETH/ipc-toolkit/build/include)

target_link_libraries(CellSim ${DEPS})