#include <igl/readOBJ.h>
#include "../include/FEMSolver.h"
#include <fstream>
#include <gmsh.h>
#include <unordered_set>

void loadMeshFromVTKFile2D(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    using TV3 = Vector<T, 3>;
    using IV3 = Vector<int, 3>;

    std::ifstream in(filename);
    std::string token;

	while(token != "POINTS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    
    int n_points;
	in >> n_points;

	in >> token; //double

	V.resize(n_points, 3);

	for(int i=0; i<n_points; i++)
        for (int  j = 0; j < 3; j++)
            in >> V(i, j);
        
    while(token != "CELLS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    int n_cells, n_entries;
	in >> n_cells;
	in >> n_entries;
    
    int cell_type;
	std::vector<Vector<int, 3>> faces;
	for(int i=0; i<n_cells; ++i)
	{
		in >> cell_type;

		if(cell_type == 3)
		{
            IV3 face;
			for(int j = 0; j < 3; j++)
				in >> face[j];
            faces.push_back(face);
		}
		else
		{
			// type 1 2
			for(int j = 0; j < cell_type; j++)
				in >> token;
		}
	}
    int n_faces = faces.size();
    F.resize(n_faces, 3);
    tbb::parallel_for(0, n_faces, [&](int i)
    {
        TV3 ei = V.row(faces[i][0]) - V.row(faces[i][1]);
        TV3 ej = V.row(faces[i][2]) - V.row(faces[i][1]);
        if (ej.cross(ei).dot(TV3(0, 0, 1)) < 0)
            F.row(i) = IV3(faces[i][1], faces[i][0], faces[i][2]);
        else
            F.row(i) = faces[i];
    });
    in.close();
}

void loadMeshFromVTKFile3D(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& T)
{
    using TV3 = Vector<double, 3>;
    using IV3 = Vector<int, 3>;
    using IV4 = Vector<int, 4>;

    struct IV3Hash
    {
        size_t operator()(const IV3& a) const{
            std::size_t h = 0;
            for (int d = 0; d < 3; ++d) {
                h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
            }
            return h;
        }
    };
    
    std::ifstream in(filename);
    std::string token;

	while(token != "POINTS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    
    int n_points;
	in >> n_points;

	in >> token; //double

	V.resize(n_points, 3);

	for(int i=0; i<n_points; i++)
        for (int  j = 0; j < 3; j++)
            in >> V(i, j);
        
    while(token != "CELLS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    int n_cells, n_entries;
	in >> n_cells;
	in >> n_entries;
    
    int cell_type;
	std::vector<IV4> tets;
	for(int i=0; i<n_cells; ++i)
	{
		in >> cell_type;

		if(cell_type == 4)
		{
            IV4 tet;
			for(int j = 0; j < 4; j++)
				in >> tet[j];
            tets.push_back(tet);
		}
		else
		{
			// type 1 2
			for(int j = 0; j < cell_type; j++)
				in >> token;
		}
	}
    int n_tets = tets.size();
    T.resize(n_tets, 4);
    tbb::parallel_for(0, n_tets, [&](int i)
    {
        T.row(i) = tets[i];
    });
    in.close();

    std::unordered_set<IV3, IV3Hash> surface;
    for (IV4& tet : tets)
    {
        for (IV3 face : { IV3(tet[0], tet[1], tet[2]), 
                    IV3(tet[0], tet[2], tet[3]), 
                    IV3(tet[0], tet[3], tet[1]),
                    IV3(tet[1], tet[3], tet[2])})
        {
            int previous_size = surface.size();
            IV3 fi = IV3(face[0], face[1], face[2]);
            bool find = surface.find(fi) != surface.end();
            if (find) surface.erase(fi);
            fi = IV3(face[0], face[2], face[1]);
            find = surface.find(fi) != surface.end();
            if (find) surface.erase(fi);
            fi = IV3(face[1], face[0], face[2]);
            find = surface.find(fi) != surface.end();
            if (find) surface.erase(fi);
            fi = IV3(face[1], face[2], face[0]);
            find = surface.find(fi) != surface.end();
            if (find) surface.erase(fi);
            fi = IV3(face[2], face[0], face[1]);
            find = surface.find(fi) != surface.end();
            if (find) surface.erase(fi);
            fi = IV3(face[2], face[1], face[0]);
            find = surface.find(fi) != surface.end();
            if (find) surface.erase(fi);

            if (previous_size == surface.size())
            {
                surface.insert(IV3(face[2], face[1], face[0]));
            }
            
        }
    }
    
    F.resize(surface.size(), 3);
    int face_cnt = 0;
    for (const auto& face : surface)
    {
        F.row(face_cnt++) = face;
    }
}

void FEMSolver::generate3DHomogenousMesh(const std::string& prefix)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    int bounding_cube = gmsh::model::occ::addBox(0, 0, 0, 1.0, 1.0, 1.0);

   
    gmsh::model::occ::synchronize();
    TV t1(1, 0, 0), t2(0, 1, 0), t3(0, 0, 1);

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, t1[2], 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, t2[2], 0, 0, 0, 1});
    std::vector<T> translation_dep({1, 0, 0, t3[0], 0, 1, 0, t3[1], 0, 0, 1, t3[2], 0, 0, 0, 1});

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(-eps, -eps, -eps, 
        t1[0] + eps, t2[1]+eps, t3[2]+eps, sleft, 2);
    // std::cout << "sleft size : " << sleft.size() << std::endl;
    
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps+t1[2], 
            xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps+t1[2], sright, 2);
        // std::cout << "sright size : " << sright.size() << std::endl;
        // std::getchar();
        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            zmax2 -= t1[2];
            zmin2 -= t1[2];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_hor);
                // std::cout << "X " << j.second << " " << i.second << std::endl;
                // std::getchar();
            }
        }
    }

    std::vector<std::pair<int, int>> sbottom;
    gmsh::model::getEntitiesInBoundingBox(-eps, -eps, -eps, 
        t1[0] + eps, +eps, t3[2]+eps, sbottom, 2);
    
    for(auto i : sbottom) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > stop;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t2[0], ymin-eps+t2[1], zmin - eps +t2[2], 
            xmax+eps+t2[0], ymax+eps+t2[1], zmax + eps + t2[2], stop, 2);

        for(auto j : stop) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t2[0];
            ymin2 -= t2[1];
            zmin2 -= t2[2];
            xmax2 -= t2[0];
            ymax2 -= t2[1];
            zmax2 -= t2[2];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_ver);
                // std::cout << "Y " << j.second << " " << i.second << std::endl;
            }
        }
    }

    std::vector<std::pair<int, int>> sback;
    gmsh::model::getEntitiesInBoundingBox(-eps, -eps, -eps, 
        t1[0] + eps, t2[1]+eps, +eps, sback, 2);
    
    for(auto i : sback) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sfront;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t3[0], ymin-eps+t3[1], zmin - eps +t3[2], 
            xmax+eps+t3[0], ymax+eps+t3[1], zmax + eps + t3[2], sfront, 2);

        for(auto j : sfront) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t3[0];
            ymin2 -= t3[1];
            zmin2 -= t3[2];
            xmax2 -= t3[0];
            ymax2 -= t3[1];
            zmax2 -= t3[2];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_dep);
                // std::cout << "Z " << j.second << " " << i.second << std::endl;
            }
        }
    }


    gmsh::model::occ::synchronize();

    gmsh::model::mesh::field::add("Distance", 1);
    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.05);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.02);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(3);
    gmsh::write(prefix + ".vtk");
    gmsh::finalize();
}

void FEMSolver::generate3DUnitCell(const std::string& prefix, T width, T alpha)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);


    // T width = 0.05; // 0.05 to 0.15
    // T alpha = 0.0;

    TV center = TV(0.5, 0.5, 0.5);

    VectorXT structure_vertices(16 * 3);
    structure_vertices << 0, 0, 0, //0
                            1, 0, 0, //1
                            1, 1, 0, //2
                            0, 1, 0, //3
                            0, 0, 1, //4
                            1, 0, 1, //5
                            1, 1, 1, //6
                            0, 1, 1, //7
                            0, 0, 0, 
                            1, 0, 0,
                            1, 1, 0,
                            0, 1, 0,
                            0, 0, 1,
                            1, 0, 1,
                            1, 1, 1,
                            0, 1, 1;
    
    for (int i = 0; i < 8; i++)
    {
        TV vi = structure_vertices.segment<3>(i * 3);
        structure_vertices.segment<3>(8 * 3 + i * 3) = vi
            + (center - vi) * std::abs(alpha);
    }

    auto vtx = [&](int idx) ->TV 
    { 
        return structure_vertices.segment<3>(idx * 3); 
    };

    auto addThickEdge = [&](const TV& from, const TV& to, T thickness) -> int
    {
        // add a x aligned box
        T length  = (to - from).norm();
        // int idx = gmsh::model::occ::addBox(-width / 2.0, -width / 2.0, -width / 2.0, 
        //         length + width, width, width);
        int idx = gmsh::model::occ::addBox(-width / 2.0, -width / 2.0, -width / 2.0, 
                length+width/2.0, width, width);
        
        Matrix<T, 3, 3> rotation_matrix = Eigen::Quaternion<T>().setFromTwoVectors(TV(1, 0, 0), (to - from).normalized()).toRotationMatrix();
        Eigen::AngleAxisd angle_axis(rotation_matrix);
        gmsh::model::occ::rotate({{3, idx}}, 0, 0, 0, angle_axis.axis()[0], angle_axis.axis()[1], angle_axis.axis()[2], angle_axis.angle());
        gmsh::model::occ::translate({{3, idx}}, from[0], from[1], from[2]);
        return idx;
    };

    std::vector<int> thick_edge_handles;
    for (int i = 0; i < 8; i++) thick_edge_handles.push_back(addThickEdge(vtx(i), vtx(i+8), width));
    for (int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        thick_edge_handles.push_back(addThickEdge(vtx(i+8), vtx(j+8), width));//8+i
    }
    for (int i = 0; i < 4; i++)
    {
        int j = (i + 1) % 4;
        thick_edge_handles.push_back(addThickEdge(vtx(i+12), vtx(j+12), width)); // 12+i
    }
    for (int i = 0; i < 4; i++) thick_edge_handles.push_back(addThickEdge(vtx(i+8), vtx(i+12), width));//16+i
    std::vector<std::vector<std::pair<int, int>>> ov(thick_edge_handles.size()+6*2);
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{3, thick_edge_handles[8]}}, {{3, thick_edge_handles[9]}}, ov[0], _dummy);
    gmsh::model::occ::fragment(ov[0], {{3, thick_edge_handles[10]}}, ov[1], _dummy);
    gmsh::model::occ::fragment(ov[1], {{3, thick_edge_handles[11]}}, ov[2], _dummy);
    for (int i = 0; i < 4; i++) gmsh::model::occ::fragment(ov[2+i], {{3, thick_edge_handles[16+i]}}, ov[3+i], _dummy);
    for (int i = 0; i < 4; i++) gmsh::model::occ::fragment(ov[6+i], {{3, thick_edge_handles[12+i]}}, ov[7+i], _dummy);
    for (int i = 0; i < 8; i++) gmsh::model::occ::fragment(ov[10+i], {{3, thick_edge_handles[i]}}, ov[11+i], _dummy);
    
    int bounding_cube2 = gmsh::model::occ::addBox(0, 0, 0, 1.0, 1.0, 1.0);
    gmsh::model::occ::intersect( ov[18], {{3, bounding_cube2}}, ov[19], _dummy);
    gmsh::model::occ::synchronize();

    // std::vector<std::pair<int, int> > f;
    // gmsh::model::getBoundary(ov[19], f);
    // std::vector<std::pair<int, int> > e;
    // gmsh::model::getBoundary(f, e, false);
    // std::vector<int> c;
    // for(auto i : f) c.push_back(abs(i.second));
    // gmsh::model::occ::fillet({ov[19][0].second}, c, {0.05}, ov[20]);

   
    gmsh::model::occ::synchronize();
    TV t1(1, 0, 0), t2(0, 1, 0), t3(0, 0, 1);

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, t1[2], 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, t2[2], 0, 0, 0, 1});
    std::vector<T> translation_dep({1, 0, 0, t3[0], 0, 1, 0, t3[1], 0, 0, 1, t3[2], 0, 0, 0, 1});

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(-eps, -eps, -eps, 
        t1[0] + eps, t2[1]+eps, t3[2]+eps, sleft, 2);
    // std::cout << "sleft size : " << sleft.size() << std::endl;
    
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps+t1[2], 
            xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps+t1[2], sright, 2);
        // std::cout << "sright size : " << sright.size() << std::endl;
        // std::getchar();
        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            zmax2 -= t1[2];
            zmin2 -= t1[2];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_hor);
                // std::cout << "X " << j.second << " " << i.second << std::endl;
                // std::getchar();
            }
        }
    }

    std::vector<std::pair<int, int>> sbottom;
    gmsh::model::getEntitiesInBoundingBox(-eps, -eps, -eps, 
        t1[0] + eps, +eps, t3[2]+eps, sbottom, 2);
    
    for(auto i : sbottom) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > stop;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t2[0], ymin-eps+t2[1], zmin - eps +t2[2], 
            xmax+eps+t2[0], ymax+eps+t2[1], zmax + eps + t2[2], stop, 2);

        for(auto j : stop) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t2[0];
            ymin2 -= t2[1];
            zmin2 -= t2[2];
            xmax2 -= t2[0];
            ymax2 -= t2[1];
            zmax2 -= t2[2];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_ver);
                // std::cout << "Y " << j.second << " " << i.second << std::endl;
            }
        }
    }

    std::vector<std::pair<int, int>> sback;
    gmsh::model::getEntitiesInBoundingBox(-eps, -eps, -eps, 
        t1[0] + eps, t2[1]+eps, +eps, sback, 2);
    
    for(auto i : sback) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sfront;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t3[0], ymin-eps+t3[1], zmin - eps +t3[2], 
            xmax+eps+t3[0], ymax+eps+t3[1], zmax + eps + t3[2], sfront, 2);

        for(auto j : sfront) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t3[0];
            ymin2 -= t3[1];
            zmin2 -= t3[2];
            xmax2 -= t3[0];
            ymax2 -= t3[1];
            zmax2 -= t3[2];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(2, {j.second}, {i.second}, translation_dep);
                // std::cout << "Z " << j.second << " " << i.second << std::endl;
            }
        }
    }


    gmsh::model::occ::synchronize();

    gmsh::model::mesh::field::add("Distance", 1);
    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.02);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.03);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.06);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(3);
    gmsh::write(prefix + ".vtk");
    gmsh::finalize();
    
}

bool FEMSolver::initializeSimulationDataFromFiles(const std::string& filename)
{
    if (filename.substr(filename.find_last_of(".") + 1) == "vtk")
    {
        
        MatrixXd V; MatrixXi F;
        Eigen::MatrixXi tets;
        loadMeshFromVTKFile3D(filename, V, F, tets);
        initializeElementData(V, F, tets);
        // loadMeshFromVTKFile2D(filename, V, F);
        // initializeSurfaceData(V, F);
    }
    return true;
}

void FEMSolver::initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
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

void FEMSolver::initializeElementData(Eigen::MatrixXd& _TV, 
    const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT)
{
    num_nodes = _TV.rows();
    
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<3>(i * dim) = _TV.row(i);
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
        surface_indices.segment<3>(i * 3) = Face(TF(i, 0), TF(i, 1), TF(i, 2));
    });

    computeBoundingBox();
    center = 0.5 * (max_corner + min_corner);
    std::cout << "BBOX: " << max_corner.transpose() << " " << min_corner.transpose() << std::endl;
    // std::getchar();

    use_ipc = true;
    if (use_ipc)
    {
        add_friction = false;
        barrier_distance = 1e-3;
        barrier_weight = 1e6;
        computeIPCRestData();
    }
    add_pbc = true;
    if (add_pbc)
    {
        pbc_w = 1e8;
        pbc_strain_w = 1e10;
        addPBCPairs3D();
        // theta = 0.0; phi = 0.0; // along Z
        // theta = M_PI * 0.5; phi = 0.0; // along X
        // theta = M_PI * 0.5; phi = M_PI * 0.5; // along Y
        theta = M_PI * 0.25; phi = M_PI * 0.25; // along 45

        // theta = M_PI * 0.05; phi = M_PI * 0.3; // along 45
        // strain_magnitudes = TV(1.1, 0.0, 0.0); 
        loading_type = UNI_AXIAL;
        strain_magnitudes = TV(1.2, 1.05, 1.2);
        // loading_type = TRI_AXIAL;
    }

    for (int i = 0; i < 3; i++)
    {
        dirichlet_data[i] = 0.0;
    }
    

    E = 2.6 * 1e3;
    nu = 0.48;
    
    verbose = false;
    max_newton_iter = 500;
    project_block_PD = false;
    std::cout << "Initialization Done" << std::endl;
}

void FEMSolver::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, bool rest)
{
    V.resize(num_nodes, 3);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V.row(i) = rest ? undeformed.segment<3>(i * dim) : deformed.segment<3>(i * dim);
    });

    F.resize(num_surface_faces, 3);
    C.resize(num_surface_faces, 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        C.row(i) = TV(0, 0.3, 1.0);
    });

    // if (three_point_bending_with_cylinder)
    // {
    //     MatrixXd cylinder_color(cylinder_faces.rows(), 3);
    //     cylinder_color.col(0).setConstant(0); cylinder_color.col(1).setConstant(1); cylinder_color.col(2).setConstant(0);
    //     appendMesh(V, F, C, cylinder_vertices, cylinder_faces, cylinder_color);
    // }
}

void FEMSolver::computeBoundingBox()
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

void FEMSolver::appendSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, T scale, const TV& center)
{
    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere.obj", v_sphere, f_sphere);
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);

    v_sphere = v_sphere * scale;

    tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
        v_sphere.row(row_idx) += center;
    });

    int n_vtx_prev = V.rows();
    int n_face_prev = F.rows();

    tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
        f_sphere.row(row_idx) += Eigen::Vector3i(n_vtx_prev, n_vtx_prev, n_vtx_prev);
    });

    V.conservativeResize(V.rows() + v_sphere.rows(), 3);
    F.conservativeResize(F.rows() + f_sphere.rows(), 3);

    V.block(n_vtx_prev, 0, v_sphere.rows(), 3) = v_sphere;
    F.block(n_face_prev, 0, f_sphere.rows(), 3) = f_sphere;
}

void FEMSolver::appendCylinder(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
        const TV& _center, const TV& direction, T R, T length)
{
    T visual_R = R;
    int n_div = 30;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV(visual_R * std::cos(theta * T(i)), 
        0.0, visual_R*std::sin(theta*T(i)));
    
    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    V.conservativeResize(n_row_V + rod_offset_v, 3);
    F.conservativeResize(n_row_F + rod_offset_f, 3);
    C.conservativeResize(n_row_F + rod_offset_f, 3);

    TV vtx_from = _center - direction * 0.5 * length;
    TV vtx_to = _center + direction * 0.5 * length;

    TV axis_world = vtx_to - vtx_from;
    TV axis_local(0, axis_world.norm(), 0);

    Matrix<T, 3, 3> rotation_matrix = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();
    
    for(int i = 0; i < n_div; i++)
    {
        for(int d = 0; d < 3; d++)
        {
            V(n_row_V + i, d) = points[i * 3 + d];
            V(n_row_V + i+n_div, d) = points[i * 3 + d];
            if (d == 1)
                V(n_row_V + i+n_div, d) += axis_world.norm();
        }

        // central vertex of the top and bottom face
        V.row(n_row_V + i) = (V.row(n_row_V + i) * rotation_matrix).transpose() + vtx_from;
        V.row(n_row_V + i + n_div) = (V.row(n_row_V + i + n_div) * rotation_matrix).transpose() + vtx_from;

        F.row(n_row_F + i*2 ) = IV(n_row_V + i, n_row_V + i+n_div, n_row_V + (i+1)%(n_div));
        F.row(n_row_F + i*2 + 1) = IV(n_row_V + (i+1)%(n_div), n_row_V + i+n_div, n_row_V + (i+1)%(n_div) + n_div);

        C.row(n_row_F + i*2 ) = TV(1.0, 0.0, 0.0);
        C.row(n_row_F + i*2 + 1) = TV(1.0, 0.0, 0.0);
    }
}

void FEMSolver::appendCylinderMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
    const TV& _center, const TV& direction, T R, T length, int sub_div_R, int sub_div_L)
{
    // std::ofstream out("cylinder_closed.obj");
    T theta = 2.0 * EIGEN_PI / T(sub_div_R);
    VectorXT points = VectorXT::Zero(sub_div_R * 3);
    for(int i = 0; i < sub_div_R; i++)
        points.segment<3>(i * 3) = TV(R * std::cos(theta * T(i)), 
        0.0, R * std::sin(theta*T(i)));
    
    int offset_v = sub_div_R * (1 + sub_div_L) + 2;
    int offset_f = sub_div_R * sub_div_L * 2 + sub_div_R * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    V.conservativeResize(n_row_V + offset_v, 3);
    F.conservativeResize(n_row_F + offset_f, 3);

    TV vtx_from = _center - direction * 0.5 * length;
    TV vtx_to = _center + direction * 0.5 * length;

    TV axis_world = vtx_to - vtx_from;
    TV axis_local(0, axis_world.norm(), 0);

    Matrix<T, 3, 3> rotation_matrix = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();
    
    for (int j = 0; j < sub_div_L + 1; j++)
    {
        for(int i = 0; i < sub_div_R; i++)
        {
            for(int d = 0; d < dim; d++)
            {
                V(n_row_V + i + sub_div_R * j, d) = points[i * 3 + d];
                if (d == 1)
                    V(n_row_V + i + sub_div_R * j, d) += axis_world.norm() / T(sub_div_L) * j;
            }
            V.row(n_row_V + j * sub_div_R + i) = (V.row(n_row_V + j * sub_div_R + i) * rotation_matrix).transpose() + vtx_from;
            if (j < sub_div_L)
            {
                F.row(n_row_F + j * sub_div_R * 2 + i*2 ) = 
                    IV(n_row_V + i + sub_div_R * j, 
                        n_row_V + i + sub_div_R * (j + 1), 
                        n_row_V + (i+1)%(sub_div_R) + sub_div_R * j);
                F.row(n_row_F + j * sub_div_R * 2 + i*2 + 1) = 
                    IV(n_row_V + (i+1)%(sub_div_R) + sub_div_R * j,
                     n_row_V + i+ sub_div_R * (j + 1), 
                     n_row_V + (i+1)%(sub_div_R) + sub_div_R * (j + 1));
            }
        }
    }
    V.row(n_row_V + offset_v - 2) = vtx_from;
    V.row(n_row_V + offset_v - 1) = vtx_to;
    for(int i = 0; i < sub_div_R; i++)
    {
        F.row(n_row_F + offset_f - 2 * sub_div_R + i) = IV(n_row_V + offset_v - 2, n_row_V + i + sub_div_L * R, n_row_V + (i + 1) % sub_div_R);
        F.row(n_row_F + offset_f - 1 * sub_div_R + i) = 
            IV(n_row_V + offset_v - 1, 
                n_row_V + (i + 1) % sub_div_R + sub_div_L * sub_div_R,
                n_row_V + i + sub_div_L * sub_div_R); 
    }

    // for (int i = 0; i < V.rows(); i++)
    //     out << "v " << V.row(i) << std::endl;
    // for (int i =0 ;i < F.rows(); i++)
    //     out << "f " << F.row(i) + IV::Ones().transpose() << std::endl;
    // out.close();
    // std::exit(0);
}