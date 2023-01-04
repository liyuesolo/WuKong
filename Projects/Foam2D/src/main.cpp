#include "../include/App.h"
#include "../include/CellSim.h"
#include <cmath>
#include <fenv.h>

using Eigen::Vector2d;


int main() {
	feenableexcept(FE_INVALID);
	//feenableexcept(FE_ALL_EXCEPT);

	using namespace cs2d;
	CellSim2D cs(23);
	std::cout << "CellSim2D initialized" << std::endl;

	const int cols = 4, rows = 3;
	cs2d::initialize_cells(cs, cols, rows, 6, 4);

	std::vector<double> central_area_verts = {
		0.25, 0,
		0.2, 0.15,
		0.1, 0.3,
		0,   0.35,
		-0.1, 0.33,
		-0.2, 0.2,
		-0.25, 0.1,
		-0.28, 0,
		-0.2, -0.1,
		-0.15, -0.15,
		-0.1, -.2,
		-0.05, -0.22,
		0.1,  -0.2,
		0.2, -0.15
	};

    Cell2DApp app(cs);
	app.initViewer(3840, 2160);

    return 0;
}
