cmake_minimum_required(VERSION 3.4)
project(visual_slam3d)
add_subdirectory(libs/g2o)
add_subdirectory(libs/pybind11)
pybind11_add_module(module_name pybind_test.cpp)

#our main file to implement SLAP
add_executable(visual_slam3d_node
src/main.cpp
src/visual_slam3d.h
)
set_target_properties(visual_slam3d_node PROPERTIES OUTPUT_NAME visual_slam3d_node${EXE_POSTFIX})
target_link_libraries(visual_slam3d_node types_sba types_slam3d types_slam3d_addons core solver_dense solver_cholmod)


#example
add_executable(visual_slam3d_example
src/example.cpp
src/visual_slam3d.h
)

set_target_properties(visual_slam3d_example PROPERTIES OUTPUT_NAME visual_slam3d_example${EXE_POSTFIX})
target_link_libraries(visual_slam3d_example types_sba types_slam3d types_slam3d_addons core solver_dense solver_cholmod)