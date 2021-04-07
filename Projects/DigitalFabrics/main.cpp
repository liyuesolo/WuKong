#include <igl/opengl/glfw/Viewer.h>

#include "EoLRodSim.h"

EoLRodSim<float> eol_sim;



bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    eol_sim.buildMeshFromRodNetwork(V, F);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    return false;
}

int main()
{
    eol_sim.buildRodNetwork(2, 2);

    igl::opengl::glfw::Viewer viewer;

    viewer.callback_key_down = &key_down;
    key_down(viewer,' ',0);
    viewer.launch();
    return 0;
}