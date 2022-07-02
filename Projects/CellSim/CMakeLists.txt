set(DEPS lbfgsb_solver cholmod_solver solver partio Half openvdb tbb tbbmalloc tbbmalloc_proxy mkl_intel_lp64 mkl_sequential mkl_core igl::core igl::opengl_glfw igl::triangle igl::opengl_glfw_imgui igl::png suitesparseconfig metis cholmod amd camd ccolamd colamd spqr igl::mosek igl::tetgen)

file(GLOB HEADERS "include/*.h" "include/*.hpp" "icp/*.h")
file(GLOB SOURCES "*.c" "*.cpp" "src/*.cpp" "src/autodiff/*.cpp" "icp/*.cpp" "src/objectives/*.cpp")


add_executable(CellSim ${HEADERS} ${SOURCES})



target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(CellSim /home/yueli/Documents/ETH/ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)
# target_link_libraries(CellSim /home/yueli/Documents/mosek/9.2/tools/platform/linux64x86/src/fusion_cxx/libmosek64.a)

target_include_directories(CellSim PUBLIC /home/yueli/Documents/nanoflann/include)
target_include_directories(CellSim PUBLIC ../../Libs/spectra/include)
target_include_directories(CellSim PUBLIC /home/yueli/Documents/mosek/9.2/tools/platform/linux64x86/h)
target_include_directories(CellSim PUBLIC /home/yueli/Documents/ETH/ipc-toolkit/build/include)

target_include_directories(CellSim PUBLIC ../../Libs/lbfgsb-Jonas)

find_package (IPOPT REQUIRED)
if(IPOPT_FOUND)
  message("IPOPT Found!")
  message("ipopt include dirs:" ${IPOPT_INCLUDE_DIRS})
  message("ipopt libraries:"  ${IPOPT_LIBRARIES})
  add_definitions("-DUSE_IPOPT")
  include_directories(${IPOPT_INCLUDE_DIRS})
  if(IPOPT_FOUND)
    target_link_libraries(CellSim ${IPOPT_LIBRARIES})
  endif()
else()
  message("IPOPT Not Found! install by \n sudo apt-get install coinor-libipopt-dev")
endif()


target_link_libraries(CellSim ${DEPS})