#ifndef EOL_ROD_SIM_H
#define EOL_ROD_SIM_H

#include "EoLNode.h"

template<class T>
class RodNetwork
{
public:
    int width, height;
    std::vector<EoLNode<T>*> nodes;
    std::vector<std::pair<Vector<int, 2>, bool>> rods;
    
public:
    RodNetwork(int w, int h) : width(w), height(h) {}
    RodNetwork() : width(5), height(5) {}
    ~RodNetwork() {}
};


template<class T>
class EoLRodSim
{
    using TV3 = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using IV2 = Vector<int, 2>;

public:
    RodNetwork<T> rod_net;

public:
    EoLRodSim() {}
    ~EoLRodSim() {}

    void buildRodNetwork(int width, int height)
    {
        rod_net = RodNetwork<T>(width, height);
        int cnt = 0;
        for(int i = 0; i < height; i++)
        {
            if(i < height - 1)
            {
                rod_net.rods.push_back(std::make_pair(IV2(i*width, (i + 1)*width + 1), true));
            }
            for(int j = 0; j < width; j++)
            {
                EoLNode<T>* node = new EoLNode<T>(
                    TV3(i/T(width), j/T(height), 0),
                    TV2(i/T(width), j/T(height)),
                    cnt++, false, 0.01
                );
                rod_net.nodes.push_back(node);
                if(j < width - 1)
                    rod_net.rods.push_back(std::make_pair(IV2(i*width + j, i*width + j + 1), false));
            }
        }
        std::cout << "# node " << rod_net.nodes.size() << " " << rod_net.rods.size() << std::endl;
    }

    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
    {
        V.resize(4, 3);
        V.row(0) = Eigen::Vector3d(0, 0, 0);
        V.row(1) = Eigen::Vector3d(1, 0, 0);
        V.row(2) = Eigen::Vector3d(1, 1, 0);
        V.row(3) = Eigen::Vector3d(0, 1, 0);
        F.resize(2, 3);
        F.row(0) = Eigen::Vector3i(0, 1, 2);
        F.row(1) = Eigen::Vector3i(0, 2, 3);
    }

    void advanceOneStep()
    {

    }


};

#endif