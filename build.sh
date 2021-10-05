mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# make -j4 DigitalFabrics3D
# make -j4 DigitalFabrics
make -j4 TopographyOptimization
# make -j4 Tiling3D