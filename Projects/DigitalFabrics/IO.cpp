#include "EoLRodSim.h"

template<class T>
void EoLRodSim<T>::buildRodNetwork(int width, int height)
    {
        
        n_nodes = width * height;
        n_rods = (width - 1) * height + (height - 1) * width;
        
        q = TV5Stack(5, n_nodes);
        dq = TV5Stack(5, n_nodes);
        rods = IV3Stack(3, n_rods);
        
        q.setZero();
        dq.setZero();
        rods.setZero();

        int cnt = 0;
        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                
                int idx = i * width + j;

                // q.col(idx) = TV5(i/T(width), T(0), j/T(height), i/T(width), j/T(height));
                q.col(idx).segment(0, 3) = TV3(i/T(width), T(0), j/T(height));
                q.col(idx).segment(3, 5) = TV2(i/T(width), j/T(height));
                if (i < height - 1)
                    rods.col(cnt++) = IV3(i*width + j, (i + 1)*width + j, WARP);
                if (j < width - 1)
                    rods.col(cnt++) = IV3(i*width + j, i*width + j + 1, WEFT);

                // rod_net.nodes.push_back(node);
                // if(i < height - 1 &&  j != 0 && j != width-1)
                //     rod_net.rods.push_back(Rod(i*width + j, (i + 1)*width + j, 1));
                // if(j < width - 1 && i != 0 && i != height-1)
                //     rod_net.rods.push_back(Rod(i*width + j, i*width + j + 1, 0));
            }
        }
        
    }

template<class T>
void EoLRodSim<T>::buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    int n_div = 10;
    
    T theta = 2.0 * EIGEN_PI / T(n_div);
    TV3Stack points = TV3Stack::Zero(3, n_div);

    // bottom face vertices
    for(int i = 0; i < n_div; i++)
        points.col(i) = TV3(R * std::cos(theta * T(i)), 0.0, R*std::sin(theta*T(i)));
    
    int rod_offset_v = n_div * 2 + 2;
    int rod_offset_f = n_div * 4;
    V.resize(n_rods * rod_offset_v, 3);
    V.setZero();
    F.resize(n_rods * rod_offset_f, 3);
    F.setZero();
    int rod_cnt = 0;
    tbb::parallel_for(0, n_rods, [&](int rod_cnt){
        int rov = rod_cnt * rod_offset_v;
        int rof = rod_cnt * rod_offset_f;

        TV3 vtx_from = q.col(rods.col(rod_cnt)[0]).segment(0, 3);
        TV3 vtx_to = q.col(rods.col(rod_cnt)[1]).segment(0, 3);
        
        TV3 axis_world = vtx_to - vtx_from;
        TV3 axis_local(0, axis_world.norm(), 0);

        // rotate local (0, 1, 0) aligned cyliner to world
        TM3 R = Eigen::Quaternion<T>().setFromTwoVectors(axis_local, axis_world).toRotationMatrix();
        
        V(rov + n_div*2+1, 1) = axis_world.norm();
        
        V.row(rov + n_div*2+1) = (V.row(rov + n_div*2+1) * R).transpose() - vtx_from;
        V.row(rov + n_div*2) = -vtx_from;
        
        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                V(rov + i, d) = points.col(i)[d];
                V(rov + i+n_div, d) = points.col(i)[d];
                if (d == 1)
                    V(rov + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            V.row(rov + i) = (V.row(rov + i) * R).transpose() - vtx_from;
            V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() - vtx_from;
            
            //top faces of the cylinder
            F.row(rof + i) = IV3(rov + n_div*2, rov + i, rov + (i+1)%(n_div));
            //bottom faces of the cylinder
            F.row(rof + i+n_div) = IV3(rov + n_div*2+1, rov + n_div + (i+1)%(n_div), rov + i + n_div);
            
            //side faces of the cylinder
            F.row(rof + i*2 + 2 * n_div) = IV3(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            F.row(rof + i*2 + 1 + 2 * n_div) = IV3(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);
        }

    });
}


template class EoLRodSim<double>;

//not working for float yet
// template class EoLRodSim<float>;