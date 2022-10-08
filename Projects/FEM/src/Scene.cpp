#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/edges.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include "../include/FEMSolver.h"
#include <fstream>
#include <gmsh.h>

template <int dim>
void FEMSolver<dim>::intializeSceneFromTriMesh(const std::string& filename)
{
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);

    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TF;
    
    igl::copyleft::tetgen::tetrahedralize(V,F, "pq1.414Y", TV,TT,TF);
    initializeElementData(TV, TF, TT);
}

template <int dim>
void FEMSolver<dim>::generatePeriodicMesh(const std::string& filename)
{
    Eigen::MatrixXd V, C, N;
    Eigen::MatrixXi F;
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/FEM/data/simulationTest.obj", V, F);
    T eps = 1e-5;
    gmsh::initialize();
    
    TV min_corner, max_corner;
    computeBBox(V, min_corner, max_corner);
    std::cout << min_corner.transpose() << " " << max_corner.transpose() << std::endl;
    gmsh::model::add("SpacerFarbic");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    TV t1(max_corner[0], min_corner[1], min_corner[2]);
    TV t2(min_corner[0], max_corner[1], min_corner[2]);
    TV t3(min_corner[0], min_corner[1], max_corner[2]);
    
    std::vector<double> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, t1[2], 0, 0, 0, 1});
    std::vector<double> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, t2[2], 0, 0, 0, 1});

    gmsh::model::occ::synchronize();

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(
        std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, std::min(0.0,t2[2])-eps, 
        std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, std::max(0.0,t2[2])+eps, sleft, 2);
    std::cout << sleft.size() << std::endl;
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 2);
        std::cout << sright.size() << std::endl;
        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) {
            gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_hor);
            }
        }
    }
    
    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();
    
    gmsh::model::mesh::generate(3);
    gmsh::write("spacer_fabric.vtk");
    gmsh::finalize();
	std::exit(0);
}

template <int dim>
void FEMSolver<dim>::initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    num_nodes = V.rows();
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<3>(i * dim) = V.row(i);
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
    const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT)
{
    num_nodes = TV.rows();
    
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<3>(i * dim) = TV.row(i);
    });
    deformed = undeformed;    
    u = VectorXT::Zero(num_nodes * dim);
    f = VectorXT::Zero(num_nodes * dim);

    num_ele = TT.rows();
    indices.resize(num_ele * 4);
    tbb::parallel_for(0, num_ele, [&](int i)
    {
        indices.segment<4>(i * 4) = TT.row(i);
    });

    num_surface_faces = TF.rows();
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = Face(TF(i, 1), TF(i, 0), TF(i, 2));
    });

    computeBoundingBox();
    center = 0.5 * (max_corner + min_corner);
    // std::cout << max_corner.transpose() << " " << min_corner.transpose() << std::endl;
    // std::getchar();

    use_ipc = true;
    if (use_ipc)
    {
        add_friction = false;
        barrier_distance = 1e-3;
        barrier_weight = 1e10;
        computeIPCRestData();
    }

    // E = 1e4;
    E = 2.6 * 1e5;
    nu = 0.48;
    
    penalty_weight = 1e8;
    use_penalty = false;
}

template <int dim>
void FEMSolver<dim>::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    V.resize(num_nodes, 3);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V.row(i) = deformed.segment<3>(i * dim);
    });

    F.resize(num_surface_faces, 3);
    C.resize(num_surface_faces, 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        C.row(i) = TV(0, 0.3, 1.0);
    });

}

template <int dim>
void FEMSolver<dim>::computeBoundingBox()
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], deformed[i * 3 + d]);
            min_corner[d] = std::min(min_corner[d], deformed[i * 3 + d]);
        }
    }
}


// template class FEMSolver<2>;
template class FEMSolver<3>;