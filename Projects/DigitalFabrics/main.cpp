#include <igl/opengl/glfw/Viewer.h>

#include "EoLRodSim.h"


#define T double
#define dim 3



EoLRodSim<T, dim> eol_sim;


Eigen::MatrixXd V;
Eigen::MatrixXi F;

// bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
// {
//     if (key == ' ')
//     {
//         eol_sim.advanceOneStep();
//         eol_sim.buildMeshFromRodNetwork(V, F);
//     }
//     viewer.data().clear();
//     viewer.data().set_mesh(V, F);
//     viewer.data().set_face_based(true);
//     return false;
// }

int main()
{
    int test = 1;
    if(test == 0)
    {
        assert(dim == 2);
        eol_sim.build5NodeTestScene();
    }
    else if (test == 1)
    {
        assert(dim == 3);
        eol_sim.buildRodNetwork(4, 4);    
        // eol_sim.addBCStretchingTest();
        eol_sim.addBCShearingTest();    
    }
    eol_sim.advanceOneStep();
    // eol_sim.buildMeshFromRodNetwork(V, F);

    // igl::opengl::glfw::Viewer viewer;

    // viewer.callback_key_down = &key_down;
    // key_down(viewer,'0',0);
    // viewer.launch();
    return 0;
}