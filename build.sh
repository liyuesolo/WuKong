mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# make -j4 DigitalFabrics3D
# make -j4 DigitalFabrics
# make -j4 TopographyOptimization
# make -j8 Tiling3D
make -j8 Tiling2D
# make -j8 CellSim
# make -j4 FEM