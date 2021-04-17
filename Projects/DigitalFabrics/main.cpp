#include <igl/opengl/glfw/Viewer.h>

#include "EoLRodSim.h"


#define T double
#define dim 2

EoLRodSim<T, dim> eol_sim;


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

enum TestCase{
    FiveNodes, Bending, Stretching, Shearing, GridScene, DerivativeCheck
};

int main()
{
    TestCase test = Shearing;
    if(test == FiveNodes)
    {
        assert(dim == 2);
        eol_sim.build5NodeTestScene();
    }
    else if (test == GridScene)
    {
        assert(dim == 3);
        eol_sim.buildRodNetwork(2, 2);    
        // eol_sim.addBCStretchingTest();
        eol_sim.addBCShearingTest();    
    }
    else if (test == Bending)
    {
        assert(dim == 2);
        eol_sim.buildLongRodForBendingTest();
    }
    else if (test == DerivativeCheck)
    {
        eol_sim.build5NodeTestScene();
        eol_sim.runDerivativeTest();
    } 
    else if (test == Shearing)
    {
        eol_sim.buildShearingTest();
    }
    // eol_sim.advanceOneStep();
    eol_sim.buildMeshFromRodNetwork(V, F);

    igl::opengl::glfw::Viewer viewer;

    viewer.callback_key_down = &key_down;
    key_down(viewer,'0',0);
    viewer.launch();
    return 0;
}