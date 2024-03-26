#include "../include/FEMSolver.h"
//#include <igl/readOBJ.h>
#include <fstream>

template <int dim>
void FEMSolver<dim>::initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    
    num_nodes = V.rows();
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed(i*dim+0) = V(i,0);
        undeformed(i*dim+1) = V(i,1);
        if constexpr (dim == 3)
            undeformed(i*dim+2) = V(i,2);
    });
    deformed = undeformed;

    u = VectorXT::Zero(num_nodes * dim);
    f = VectorXT::Zero(num_nodes * dim);

    num_ele = 0;
    
    num_surface_faces = F.rows();
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = Face(F(i, 0), F(i, 1), F(i, 2));
    });
}

template <int dim>
void FEMSolver<dim>::initializeElementData(Eigen::MatrixXd& TV, 
    const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT, const Eigen::MatrixXi& TF_quad, const Eigen::MatrixXi& TT_quad)
{
    num_nodes = TV.rows();
    VS = TV;
    FS = TF;
    undeformed.resize(num_nodes * dim);
    if(USE_NEW_FORMULATION)
    {
        additional_dof = slave_nodes_3d[0].size();
        undeformed.resize((additional_dof+num_nodes) * dim);
    } 

    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<dim>(i * dim) = TV.row(i);
    });

    if(USE_NEW_FORMULATION)
    {
        for(auto it = slave_nodes_3d[0].begin(); it!=slave_nodes_3d[0].end(); it++)
        {
            undeformed.segment<dim>((it->second-1+num_nodes) * dim) = TV.row(it->first);
            //undeformed((it->second-1+num_nodes) * dim) += 0.001;
           
           	Vector3a xi = undeformed.segment<3>(3*(it->second+num_nodes-1));
            Vector3a ck = undeformed.segment<3>(3*(it->first));
            Eigen::VectorXi valence_indices;

            // std::cout<<it->first<<" "<<it->second<<std::endl;
            // std::cout<<xi.transpose()<<" ";
            // std::cout<<ck.transpose()<<std::endl;
        }
        
        //undeformed((i+num_nodes) * dim) += 0.001;
    }

    deformed = undeformed;
    std::cout<<"deformed: "<<num_nodes <<" "<<dim<<std::endl;
    std::cout<<"additional: "<<additional_dof <<" "<<dim<<std::endl;
    std::cout<<"total: "<<undeformed.size() <<std::endl;
        
    u = VectorXT::Zero(num_nodes * dim);
    f = VectorXT::Zero(num_nodes * dim);
    if(USE_NEW_FORMULATION)
    {
        u = VectorXT::Zero((additional_dof+num_nodes) * dim);
        f = VectorXT::Zero((additional_dof+num_nodes) * dim);
    }

    #if USE_QUAD_ELEMENT
    num_ele_quad = TT_quad.rows();
    indices_quad.resize(num_ele_quad * 4);
    tbb::parallel_for(0, num_ele_quad, [&](int i)
    { 
        indices_quad.segment<4>(i * 4) = TT_quad.row(i);
    });


    num_surface_faces_quad = TF_quad.rows();
    surface_indices_quad.resize(num_surface_faces_quad * 4);
    tbb::parallel_for(0, num_surface_faces_quad,  [&](int i)
    {
        surface_indices_quad.segment<4>(i * 4) = FaceQuad(TF_quad(i, 0), TF_quad(i, 1), TF_quad(i, 2), TF_quad(i, 3));
    });
    #endif

    num_ele = TT.rows();
    indices.resize(num_ele * (dim+1));
    tbb::parallel_for(0, num_ele, [&](int i)
    {
        indices.segment<dim+1>(i * (dim+1)) = TT.row(i);
        // indices[i * 4 + 0] = TT(i, 1);
        // indices[i * 4 + 1] = TT(i, 2);
        // indices[i * 4 + 2] = TT(i, 3);
        // indices[i * 4 + 3] = TT(i, 0);
    });
    // tbb::parallel_for(0, num_ele, [&](int i)
    // {
    //     if constexpr(dim == 2){
    //         indices.segment<dim+1>(i * (dim+1)) = Vector<int, 3>(TT(i, 1), TT(i, 0), TT(i, 2));
    //     }
    // });

    std::vector<Eigen::Vector3i> surface_faces;
    for(int i=0; i<TF.rows(); ++i)
    {
        bool is_surface = true;
        for(int j=0; j<3; ++j)
        {
            if(!is_surface_vertex[TF(i,j)])
            {
                is_surface = false;
                break;
            }
        }
        if(is_surface) surface_faces.push_back(TF.row(i));
    }
    

    num_surface_faces = surface_faces.size(); 
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = surface_faces[i];
    });

    std::cout<<"Surface faces: "<<surface_faces.size()<<std::endl;

    computeBoundingBox();
    center = 0.5 * (max_corner + min_corner);
    // std::cout << max_corner.transpose() << " " << min_corner.transpose() << std::endl;
    // std::getchar();

    // use_ipc = true;

    initializeBoundaryInfo();
    barrier_distance = 1e-3;
    //barrier_distance = 1;
    barrier_weight = 1;
    if(USE_DYNAMICS)
    {
        std::vector<Entry> entries;
        constructMassMatrix(entries);
        x_prev = undeformed;
        v_prev = x_prev;
        v_prev.setZero();
        if(USE_FRICTION)
        {
            prev_contact_force = undeformed;
            prev_contact_force.setZero();
        }
    } 

    //computeIPC2DRestData(1);
    
    // if(USE_IPC_3D)
    //     computeIPC3DRestData();
    // else if(USE_TRUE_IPC_2D)
    //     computeIPC2DtrueRestData();
    

    //E = 1e4;
    E = 1 * 1e4;
    E_2 = 5 * 1e5;
    nu = 0.3;
    
    penalty_weight = 1e6;
    use_penalty = true;
    USE_SHELL = true;

    if(USE_SHELL)
    {
        gravitional_energy = false;
        updateLameParameters();
        buildHingeStructure();
        computeRestShape();
    }

    computeIPC3DRestData();

    if(USE_MORTAR_METHOD)
        initializeMortarInformation();
}




template <int dim>
void FEMSolver<dim>::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    V.resize(num_nodes, 3);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V(i,0) = deformed(i*dim+0);
        V(i,1) = deformed(i*dim+1);
        if constexpr (dim == 3)
            V(i,2) = deformed(i*dim+2);
        else 
            V(i,2) = 0;
    });


    F.resize(num_surface_faces, 3);
    C.resize(num_surface_faces, 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        if(F(i,0) == 2660 || F(i,1) == 2660 || F(i,2) == 2660)
            C.row(i) = Vector<T, 3>(1,1,1);
        else if(i == 10449)
            C.row(i) = Vector<T, 3>(1,1,0);
        else
            C.row(i) = Vector<T, 3>(0.3, 223./255., 229./255.);
        //C.row(i) = Vector<T, 3>(1,1,1);
        //std::cout<<C.row(i)<<std::endl;
    });
}

template <int dim>
void FEMSolver<dim>::generateMeshForRenderingStress(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, double max_color_val)
{
    V.resize(num_nodes, 3);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V(i,0) = deformed(i*dim+0);
        V(i,1) = deformed(i*dim+1);
        if constexpr (dim == 3)
            V(i,2) = deformed(i*dim+2);
        else 
            V(i,2) = 0;
    });

    F.resize(num_surface_faces, 3);
    C.resize(num_surface_faces, 3);
    Eigen::VectorXd min_color(3);
    min_color<<0, 0.0, 1.0;
    Eigen::VectorXd max_color(3);
    max_color<<1.0, 0, 0;

    VectorXT residual(deformed.rows());
    residual.setZero();
    VectorXT contactforce(deformed.rows());
    contactforce.setZero();
    addIPC2DtrueForceEntries(contactforce);
    //computeResidual(u,residual);

    // Eigen::Vector2d fs,fs2;
    // fs.setZero(); fs2.setZero();
    // for(int i=0;i<193; ++i)
    // {
    //     fs2+=residual.segment<2>(2*i);
    // }
    // for(int i=193;i<num_nodes; ++i)
    // {
    //     fs+=residual.segment<2>(2*i);
    // }
    // std::cout<<fs2.transpose()<<std::endl;
    // std::cout<<fs.transpose()<<std::endl;
    std::cout<<"contact force: "<<contactforce.segment<2>(2*(15+RES_2)).transpose()<<std::endl;

    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
    });

    compute1PKStress();
    
    double max_val = OnePKStress.maxCoeff();
    double min_val = 0;
    std::cout<<"Max Stress: "<<max_val<<std::endl;

    #if USE_QUAD_ELEMENT
    std::cout<<max_val<<std::endl;
    assert(2*num_surface_faces_quad == num_surface_faces);
    tbb::parallel_for(0, num_surface_faces_quad,  [&](int i)
    {
        F.row(2*i) = surface_indices.segment<3>(2* i * 3);
        F.row(2*i+1) = surface_indices.segment<3>((2*i+1) * 3);
        double lambda = (fabs(OnePKStress(i))-min_val)/(max_val-min_val);
        //double lambda = 1;
        C.row(2*i) = lambda*max_color + (1-lambda)*min_color;
        C.row(2*i+1) = lambda*max_color + (1-lambda)*min_color;
    });
    #else
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        double lambda = (fabs(OnePKStress(i))-min_val)/(max_val-min_val);
        //double lambda = 1;
        C.row(i) = lambda*max_color + (1-lambda)*min_color;
    });
    #endif
}

template <int dim>
void FEMSolver<dim>::computeBoundingBox()
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            max_corner[d] = std::max(max_corner[d], deformed[i * dim + d]);
            min_corner[d] = std::min(min_corner[d], deformed[i * dim + d]);
        }
    }
}


template class FEMSolver<2>;
template class FEMSolver<3>;