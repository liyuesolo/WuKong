set(DEPS cholmod_solver solver partio Half openvdb tbb tbbmalloc tbbmalloc_proxy mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis cholmod amd camd ccolamd colamd spqr igl::mosek)

file(GLOB HEADERS "include/*.h" "include/*.hpp" )
file(GLOB SOURCES "*.c" "*.cpp" "src/*.cpp" "src/autodiff/*.cpp" "src/objectives/*.cpp")


add_executable(CellSim2D ${HEADERS} ${SOURCES})

set(SuiteSparse_ROOT "/home/yueli/Documents/SuiteSparse-5.9.0")
include_directories(${SuiteSparse_ROOT})
include_directories(${SuiteSparse_ROOT}/include)
link_directories(${SuiteSparse_ROOT}/lib)


target_link_libraries(CellSim2D /home/yueli/Documents/ETH/ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(CellSim2D /home/yueli/Documents/ETH/ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(CellSim2D /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(CellSim2D /home/yueli/Documents/ETH/ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(CellSim2D /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)
# target_link_libraries(CellSim /home/yueli/Documents/mosek/9.2/tools/platform/linux64x86/src/fusion_cxx/libmosek64.a)

target_include_directories(CellSim2D PUBLIC /home/yueli/Documents/nanoflann/include)
target_include_directories(CellSim2D PUBLIC ../../Libs/spectra/include)
target_include_directories(CellSim2D PUBLIC ../../Libs/lbfgs)
target_include_directories(CellSim2D PUBLIC /home/yueli/Documents/mosek/9.2/tools/platform/linux64x86/h)
target_include_directories(CellSim2D PUBLIC /home/yueli/Documents/ETH/ipc-toolkit/build/include)


find_package (IPOPT REQUIRED)
if(IPOPT_FOUND)
  message("IPOPT Found!")
  message("ipopt include dirs:" ${IPOPT_INCLUDE_DIRS})
  message("ipopt libraries:"  ${IPOPT_LIBRARIES})
  add_definitions("-DUSE_IPOPT")
  include_directories(${IPOPT_INCLUDE_DIRS})
  if(IPOPT_FOUND)
    target_link_libraries(CellSim2D ${IPOPT_LIBRARIES})
  endif()
else()
  message("IPOPT Not Found! install by \n sudo apt-get install coinor-libipopt-dev")
endif()

target_link_libraries(CellSim2D ${DEPS})