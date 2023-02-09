set(DEPS lbfgsb_solver cholmod_solver solver tbb tbbmalloc tbbmalloc_proxy mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis cholmod amd camd ccolamd colamd spqr igl::mosek igl::tetgen)

file(GLOB HEADERS "include/*.h" "include/*.hpp" )
file(GLOB SOURCES "*.c" "*.cpp" "src/*.cpp" "src/autodiff/*.cpp")


add_executable(CellSimExplicit ${HEADERS} ${SOURCES})


target_link_libraries(CellSimExplicit /home/yueli/Documents/ETH/ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(CellSimExplicit /home/yueli/Documents/ETH/ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(CellSimExplicit /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(CellSimExplicit /home/yueli/Documents/ETH/ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(CellSimExplicit /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)

target_include_directories(CellSimExplicit PUBLIC /home/yueli/Documents/nanoflann/include)
target_include_directories(CellSimExplicit PUBLIC ../../Libs/spectra/include)
target_include_directories(CellSimExplicit PUBLIC /home/yueli/Documents/mosek/9.2/tools/platform/linux64x86/h)
target_include_directories(CellSimExplicit PUBLIC /home/yueli/Documents/ETH/ipc-toolkit/build/include)

target_link_libraries(CellSimExplicit ${DEPS})
