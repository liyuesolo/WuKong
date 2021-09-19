find_package(Eigen3 REQUIRED)
add_library(eigen INTERFACE IMPORTED)

find_package(Threads)

add_library(ziran STATIC
  CS/DataStructure/Box.cpp
  CS/DataStructure/DataArrayBase.cpp
  CS/DataStructure/KdTree.cpp
  CS/Util/Active.cpp
  CS/Util/ErrorContext.cpp
  CS/Util/FloatingPointExceptions.cpp
  CS/Util/Logging.cpp
  CS/Util/Profiler.cpp
  CS/Util/PrettyPrinting.cpp
  CS/Util/RandomNumber.cpp
  CS/Util/SignalHandler.cpp
  CS/Util/Signals.cpp
  CS/Util/SharedLibrary.cpp
  CS/Util/StackTrace.cpp
  CS/Util/Timer.cpp
  Math/Geometry/AnalyticLevelSet.cpp
  Math/Geometry/ObjIO.cpp
  Math/Geometry/VdbLevelSet.cpp
  Physics/ConstitutiveModel/LinearCorotated.cpp
  Physics/ConstitutiveModel/NeoHookean.cpp
)
target_compile_options(ziran PRIVATE ${ZIRAN_CXXFLAGS_LTO})
target_include_directories(ziran PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(ziran SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_options(ziran PUBLIC "-Wno-unused-but-set-parameter")
target_compile_options(ziran PUBLIC "-Wno-return-type")

target_link_libraries(ziran PUBLIC eigen half kdtree Tick tbb ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${ZIRAN_CXXFLAGS_LTO})

if(OPENVDB_FOUND)
  target_link_libraries(ziran PRIVATE openvdb)
endif()

set(ZIRAN_DATA_DIR ${ZIRAN_SOURCE_DIR}/Data)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CS/Util/DataDir.h.in ${CMAKE_CURRENT_SOURCE_DIR}/CS/Util/DataDir.h @ONLY)
