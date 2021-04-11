#include <igl/opengl/glfw/Viewer.h>

#include "EoLRodSim.h"

EoLRodSim<double, 3> eol_sim;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        eol_sim.advanceOneStep();
        eol_sim.buildMeshFromRodNetwork(V, F);
    }
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    return false;
}

int main()
{
    eol_sim.buildRodNetwork(10, 10);
    eol_sim.buildMeshFromRodNetwork(V, F);

    igl::opengl::glfw::Viewer viewer;

    viewer.callback_key_down = &key_down;
    key_down(viewer,'0',0);
    viewer.launch();
    return 0;
}