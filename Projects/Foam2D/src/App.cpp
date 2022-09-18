#include <igl/triangle/triangulate.h>
//#include <igl/triangle_fan.h>

#include "../include/App.h"
#include "../include/Cell.h"
#include "../include/Triangulate.h"

void Cell2DApp::setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu)
{
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {

        }
    };

    viewer.callback_key_pressed = 
        [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
    {
        switch(key)
        {
		case ' ':
			std::cout << "Simulation step =" << cellSim.t << std::endl;
			cellSim.step();
			//std::cout << cellSim.vertices << std::endl;
			break;
        default: 
            return false;
        }
		updateScreen(viewer);
        
    };
    
    //foam.createRectangleScene();

    updateScreen(viewer);

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 20.0;
    viewer.data().line_width = 4.0;
    //viewer.data().set_colors(C);

	// draw two cells to test disconnected meshes

}

void Cell2DApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	//cellSim.triangulate_cell(cellSim.cells[0], V, F);
	auto x = cellSim.triangulate_all_cells();
    viewer.data().set_mesh(x.first, x.second);
	//viewer.data().set_data(Eigen::MatrixXd::Random(6, 1));
    viewer.data().add_points(cellSim.vertices_state, Eigen::RowVector3d(.7, .5, 0));

	/*
	Eigen::MatrixXd lineVertsA(cellSim.edges.size(), 3);
	Eigen::MatrixXd lineVertsB(cellSim.edges.size(), 3);
	int i = 0;
	for (auto edgeIterator: cellSim.edges) {
		int a, b;
		std::tie(a, b) = edgeIterator.first;
		lineVertsA.row(i) << cellSim.vertices.row(a);
		lineVertsB.row(i) << cellSim.vertices.row(b);
		i++;
	}

    viewer.data().add_edges(lineVertsA, lineVertsB, Eigen::RowVector3d(.7, .5, 0));
	*/
}
