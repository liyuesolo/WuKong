#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::build5NodeTestScene()
{
    n_nodes = 5;
    n_rods = 4;

    q = DOFStack(dof, n_nodes); q.setZero();
    rods = IV3Stack(3, n_rods); rods.setZero();
    connections = IV4Stack(4, n_nodes);

    normal = TV3Stack(3, n_rods);
    
    if constexpr (dim == 2)
    {
        q.col(0).template segment<dim>(0) = TV2(0, 0.5);
        q.col(0).template segment<2>(dim) = TV2(0, 0.5);

        q.col(1).template segment<dim>(0) = TV2(1, 0.5);
        q.col(1).template segment<2>(dim) = TV2(1, 0.5);

        q.col(2).template segment<dim>(0) = TV2(0.5, 0);
        q.col(2).template segment<2>(dim) = TV2(0.5, 0);

        q.col(3).template segment<dim>(0) = TV2(0.5, 1);
        q.col(3).template segment<2>(dim) = TV2(0.5, 1);

        q.col(4).template segment<dim>(0) = TV2(0.5, 0.5);
        q.col(4).template segment<2>(dim) = TV2(0.5, 0.5);

        rods.col(0) = IV3(0, 4, WARP);
        rods.col(1) = IV3(4, 1, WARP);
        rods.col(2) = IV3(2, 4, WEFT);
        rods.col(3) = IV3(4, 3, WEFT);

        normal.col(0) = TV3(0, 1, 0);
        normal.col(1) = TV3(0, 1, 0);
        normal.col(2) = TV3(0, -1, 0);
        normal.col(3) = TV3(0, -1, 0);

        TVDOF target, mask;
        mask.setOnes();
        target.setZero();
        
        //fully fix three nodes
        dirichlet_data[0] = std::make_pair(target, mask);
        dirichlet_data[1] = std::make_pair(target, mask);
        dirichlet_data[2] = std::make_pair(target, mask);
        
        //move top vertex
        target.template segment<dim>(0) = TV2(-0.1, 0);
        dirichlet_data[3] = std::make_pair(target, mask);

        // uncomment this to fix uv of the middle node
        // target.setZero();
        // mask.template segment<dim>(0) = TV::Zero();
        // dirichlet_data[4] = std::make_pair(target, mask);

        connections(2, 0) = -1; connections(3, 0) = -1; connections(0, 0) = -1; connections(1, 0) = 4; 
        connections(2, 1) = -1; connections(3, 1) = -1; connections(0, 1) = 4; connections(1, 1) = -1; 
        connections(2, 2) = -1; connections(3, 2) = 4; connections(0, 2) = -1; connections(1, 2) = -1; 
        connections(2, 3) = 4; connections(3, 3) = -1; connections(0, 3) = -1; connections(1, 3) = -1; 
        connections(2, 4) = 2; connections(3, 4) = 3; connections(0, 4) = 0; connections(1, 4) = 1; 
    }
    else
    {
        std::cout << "3D version this is not implemented" << std::endl;
        std::exit(0);
    }
}


template<class T, int dim>
void EoLRodSim<T, dim>::buildRodNetwork(int width, int height)
{
    
    n_nodes = (width + 1) * (height + 1);
    n_rods = (width + 1) * height + (height + 1) * width;
    
    q = DOFStack(dof, n_nodes);
    rods = IV3Stack(3, n_rods);
    normal = TV3Stack(3, n_rods);
    q.setZero();
    rods.setZero();

    connections = IV4Stack(4, n_nodes);

    int cnt = 0;
    for(int i = 0; i < height+1; i++)
    {
        for(int j = 0; j < width+1; j++)
        {
            
            int idx = i * (width+1) + j;
            
            if constexpr (dim == 3)
                q.col(idx).template segment<dim>(0) = TV3(i/T(width), T(0), j/T(height));
            else if constexpr (dim == 2)
                q.col(idx).template segment<dim>(0) = TV2(i/T(width), j/T(height));

            q.col(idx).template segment<2>(dim) = TV2(i/T(width), j/T(height));
            if (i < height)
            {
                normal.col(cnt) = TV3(0, 1, 0);
                rods.col(cnt++) = IV3(i*(width+1) + j, (i + 1)*(width+1) + j, WARP);
            }
            if (j < width)
            {
                normal.col(cnt) = TV3(0, -1, 0);
                rods.col(cnt++) = IV3(i*(width+1) + j, i*(width+1) + j + 1, WEFT);
            }
            IV4 neighbor = IV4::Zero();
            neighbor[2] = (j - 1) < 0 ? -1 : i*(width+1) + j - 1;
            neighbor[3] = (j + 1) > width  ? -1 : i*(width+1) + j + 1;

            neighbor[0] = (i - 1) < 0 ? -1 : (i - 1)*(width+1) + j;
            neighbor[1] = (i + 1) > height  ? -1 : (i + 1)*(width+1) + j;
            connections.col(idx) = neighbor;
        }
    }
    
}



template<class T, int dim>
void EoLRodSim<T, dim>::buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
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

        TV vtx_from_TV = q.col(rods.col(rod_cnt)[0]).template segment<dim>(0);
        TV vtx_to_TV = q.col(rods.col(rod_cnt)[1]).template segment<dim>(0);

        TV3 vtx_from = TV3::Zero();
        TV3 vtx_to = TV3::Zero();
        if constexpr (dim == 3)
        {
            vtx_from = vtx_from_TV;
            vtx_to = vtx_to_TV;
        }
        else
        {
            vtx_from = TV3(vtx_from_TV[0], 0, vtx_from_TV[1]);
            vtx_to = TV3(vtx_to_TV[0], 0, vtx_to_TV[1]);
        }

        
        TV3 normal_offset = TV3::Zero();
        if (rods.col(rod_cnt)[2] == WARP)
            normal_offset = normal.col(rod_cnt);
        else
            normal_offset = normal.col(rod_cnt);

        vtx_from += normal_offset * R;
        vtx_to += normal_offset * R;
        
        TV3 axis_world = vtx_to - vtx_from;
        TV3 axis_local(0, axis_world.norm(), 0);

        
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


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;

//not working for float yet
// template class EoLRodSim<float>;