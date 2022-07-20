
add_library(lbfgsb_solver STATIC lbfgsb-Jonas/lbfgsb/LBFGSB-fable.h lbfgsb-Jonas/BoxConstraints.cpp lbfgsb-Jonas/LBFGSB.cpp lbfgsb-Jonas/lbfgsb/LBFGSB-fable.cpp)
target_include_directories(lbfgsb_solver PUBLIC lbfgsb-Jonas/lbfgsb)
target_link_libraries(lbfgsb_solver ${DEPS})
set(DEPS ${DEPS} lbfgsb_solver)

add_library(clipper STATIC clipper/clipper.cpp)
target_include_directories(clipper PUBLIC clipper/clipper.h)
target_link_libraries(clipper ${DEPS})
set(DEPS ${DEPS} clipper)