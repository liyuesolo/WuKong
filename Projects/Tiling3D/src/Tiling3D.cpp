#include "../include/Tiling3D.h"

#include <random>
#include <cmath>
#include <fstream>

#include <igl/copyleft/tetgen/tetrahedralize.h>

#include "../include/Util.h"

std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );

static double zeta()
{
	return dis(gen);
}

void Tiling3D::fetchOneFamilyFillRegion(int IH, T* params, 
    std::vector<PointLoops>& raw_points, T width, T height)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );

    a_tiling.setParameters( params );

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        // Start by making a random Bezier segment.
        ej.push_back( dvec2( 0, 0 ) );
        // ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
        // ej.push_back( 
        //     dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
        // ej.push_back( dvec2( 0.25, 0 ) );
        // ej.push_back( dvec2( 0.75, 0) );
        ej.push_back( dvec2( 1, 0 ) );

        // Now, depending on the edge shape class, enforce symmetry 
        // constraints on edges.
        switch( a_tiling.getEdgeShape( idx ) ) {
        case J: 
            break;
        case U:
            ej[2].x = 1.0 - ej[1].x;
            ej[2].y = ej[1].y;
            break;
        case S:
            ej[2].x = 1.0 - ej[1].x;
            ej[2].y = -ej[1].y;
            break;
        case I:
            ej[1].y = 0.0;
            ej[2].y = 0.0;
            break;
        }
        edges[idx] = ej;
    }

    // Use a vector to hold the control points of the final tile outline.
    vector<dvec2> shape;

    // Iterate over the edges of a single tile, asking the tiling to
    // tell you about the geometric information needed to transform 
    // the edge shapes into position.  Note that this iteration is over
    // whole tiling edges.  It's also to iterator over partial edges
    // (i.e., halves of U and S edges) using t.parts() instead of t.shape().
    for( auto i : a_tiling.shape() ) {
        // Get the relevant edge shape created above using i->getId().
        const vector<dvec2>& ed = edges[ i->getId() ];
        // Also get the transform that maps to the line joining consecutive
        // tiling vertices.
        const glm::dmat3& TT = i->getTransform();

        // If i->isReversed() is true, we need to run the parameterization
        // of the path backwards.
        if( i->isReversed() ) {
            for( size_t idx = 1; idx < ed.size(); ++idx ) {
                shape.push_back( TT * dvec3( ed[ed.size()-1-idx], 1.0 ) );
            }
        } else {
            for( size_t idx = 1; idx < ed.size(); ++idx ) {
                shape.push_back( TT * dvec3( ed[idx], 1.0 ) );
            }
        }
    }

    // for( auto i : a_tiling.fillRegion(-5, -4, 2, 4) ) {
    // for( auto i : a_tiling.fillRegion(0, -1.5, 7, 5) ) {
    for( auto i : a_tiling.fillRegion(0, -1.5, 5, 3) ) {

        std::vector<TV2> data_points;
        dmat3 TT = i->getTransform();
        
        dvec2 p = TT * dvec3( shape.back(), 1.0 );

        data_points.push_back(TV2(p[0], p[1]));
        // std::cout << p[0] << " " << p[1] << std::endl;
        for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
            dvec2 p1 = TT * dvec3( shape[idx], 1.0 );
            dvec2 p2 = TT * dvec3( shape[idx+1], 1.0 );
            dvec2 p3 = TT * dvec3( shape[idx+2], 1.0 );

            data_points.push_back(TV2(p1[0], p1[1]));
            data_points.push_back(TV2(p2[0], p2[1]));
            data_points.push_back(TV2(p3[0], p3[1]));
        }

        if (TT[0][0] != TT[1][1])
            std::reverse(data_points.begin(), data_points.end());

        raw_points.push_back(data_points);   

    }
}

void Tiling3D::fetchOneFamily(int IH, T* params, TV2& T1, TV2& T2, 
    PointLoops& raw_points, T width, T height)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    
    size_t num_params = a_tiling.numParameters();
    if( num_params > 1 ) {
            double params[ num_params ];
        // Get the parameters out of the tiling
        a_tiling.getParameters( params );
        // Change a parameter
        for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) {
            // if (random)
                params[idx] += zeta()*0.2 - 0.1;
        }
        // Send the parameters back to the tiling
        a_tiling.setParameters( params );
    }

    // a_tiling.setParameters( params );

    T1 = TV2(a_tiling.getT1().x, a_tiling.getT1().y);
    T2 = TV2(a_tiling.getT2().x, a_tiling.getT2().y);

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        // Start by making a random Bezier segment.
        ej.push_back( dvec2( 0, 0 ) );
        ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
        ej.push_back( 
            dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
        // ej.push_back( dvec2( 0.25, 0 ) );
        // ej.push_back( dvec2( 0.75, 0) );
        ej.push_back( dvec2( 1, 0 ) );

        // Now, depending on the edge shape class, enforce symmetry 
        // constraints on edges.
        switch( a_tiling.getEdgeShape( idx ) ) {
        case J: 
            break;
        case U:
            ej[2].x = 1.0 - ej[1].x;
            ej[2].y = ej[1].y;
            break;
        case S:
            ej[2].x = 1.0 - ej[1].x;
            ej[2].y = -ej[1].y;
            break;
        case I:
            ej[1].y = 0.0;
            ej[2].y = 0.0;
            break;
        }
        edges[idx] = ej;
    }

    // Use a vector to hold the control points of the final tile outline.
    vector<dvec2> shape;

    // Iterate over the edges of a single tile, asking the tiling to
    // tell you about the geometric information needed to transform 
    // the edge shapes into position.  Note that this iteration is over
    // whole tiling edges.  It's also to iterator over partial edges
    // (i.e., halves of U and S edges) using t.parts() instead of t.shape().
    for( auto i : a_tiling.shape() ) {
        // Get the relevant edge shape created above using i->getId().
        const vector<dvec2>& ed = edges[ i->getId() ];
        // Also get the transform that maps to the line joining consecutive
        // tiling vertices.
        const glm::dmat3& TT = i->getTransform();

        // If i->isReversed() is true, we need to run the parameterization
        // of the path backwards.
        if( i->isReversed() ) {
            for( size_t idx = 1; idx < ed.size(); ++idx ) {
                shape.push_back( TT * dvec3( ed[ed.size()-1-idx], 1.0 ) );
            }
        } else {
            for( size_t idx = 1; idx < ed.size(); ++idx ) {
                shape.push_back( TT * dvec3( ed[idx], 1.0 ) );
            }
        }
    }

    dvec2 p = dvec3( shape.back(), 1.0 );
    raw_points.push_back(TV2(p[0], p[1]));
    // std::cout << p[0] << " " << p[1] << std::endl;
    for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
        dvec2 p1 = dvec3( shape[idx], 1.0 );
        dvec2 p2 = dvec3( shape[idx+1], 1.0 );
        dvec2 p3 = dvec3( shape[idx+2], 1.0 );

        raw_points.push_back(TV2(p1[0], p1[1]));
        raw_points.push_back(TV2(p2[0], p2[1]));
        raw_points.push_back(TV2(p3[0], p3[1]));        
    }


    // for( auto i : a_tiling.fillRegion(0.0, 0.0, width, height) ) {

    //     std::vector<TV2> data_points;
    //     dmat3 TT = i->getTransform();
        
    //     dvec2 p = TT * dvec3( shape.back(), 1.0 );

    //     data_points.push_back(TV2(p[0], p[1]));
    //     // std::cout << p[0] << " " << p[1] << std::endl;
    //     for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
    //         dvec2 p1 = TT * dvec3( shape[idx], 1.0 );
    //         dvec2 p2 = TT * dvec3( shape[idx+1], 1.0 );
    //         dvec2 p3 = TT * dvec3( shape[idx+2], 1.0 );

    //         data_points.push_back(TV2(p1[0], p1[1]));
    //         data_points.push_back(TV2(p2[0], p2[1]));
    //         data_points.push_back(TV2(p3[0], p3[1]));
    //     }

    //     if (TT[0][0] != TT[1][1])
    //         std::reverse(data_points.begin(), data_points.end());

    //     raw_points.push_back(data_points);   

    // }

}

void Tiling3D::extrudeToMesh(const std::vector<PointLoops>& raw_points, 
        T width, T height, std::string filename)
{
    std::vector<TV2> unique_points;
    std::vector<IdList> polygon_ids;

    for (const PointLoops& pl : raw_points)
    {
        TV2 center = TV2::Zero();
        IdList id_list;
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 pt = pl[i];
            center += pt;
            auto find_iter = std::find_if(unique_points.begin(), unique_points.end(), 
                            [&pt](const TV2 x)->bool
                               { return (x - pt).norm() < 1e-6; }
                             );
            if (find_iter == unique_points.end())
            {
                unique_points.push_back(pt);
                id_list.push_back(unique_points.size() - 1);
            }
            else
                id_list.push_back(std::distance(unique_points.begin(), find_iter));
        }

        polygon_ids.push_back(id_list);
        center /= T(pl.size() - 1);
        id_list.clear();
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 inner = center + (pl[i] - center) * 0.9;
            unique_points.push_back(inner);
            id_list.push_back(unique_points.size() - 1);
        }
        polygon_ids.push_back(id_list);
    }

    std::vector<Face> faces;
    for (int i = 0; i < polygon_ids.size() / 2; i++)
    {
        auto outer = polygon_ids[i * 2];
        auto inner = polygon_ids[i * 2 + 1];

        for (int j = 0; j < outer.size(); j++)
        {
            faces.push_back(Face(outer[j] + 1, inner[j] + 1, inner[(j + 1) % outer.size()] + 1));
            faces.push_back(Face(outer[j] + 1, inner[(j + 1) % outer.size()] + 1, outer[(j + 1) % outer.size()] + 1));
        }
    }

    std::vector<TV> vertices;
    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], 0));

    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], height));

    int nv = unique_points.size();
    int nf = faces.size();

    for (int i = 0; i < nf; i++)
    {
        faces.push_back(Face(
          faces[i][2] + nv, faces[i][1] + nv, faces[i][0] + nv
        ));
    }
    
    for (IdList id_list : polygon_ids)
    {
        for (int i = 0; i < id_list.size(); i++)
        {
            faces.push_back(Face(
                id_list[i] + 1, id_list[(i + 1)%id_list.size()] + 1, id_list[i] + nv + 1
            ));
            faces.push_back(Face(
                id_list[(i + 1)%id_list.size()] + 1, id_list[(i + 1)%id_list.size()] + nv + 1, id_list[i] + nv + 1 
            ));
        }
    }
    
    
    std::ofstream out(filename);
    for (const TV& pt : vertices)
        out << "v " << pt.transpose() * 10.0 << std::endl;
    for (auto face : faces)
        out << "f " << face.transpose() << std::endl;
    out.close();
}


void Tiling3D::test()
{
    std::vector<PointLoops> raw_points;
    // this is in cm
    int IH = 5;
    T params[] = {0.1224, 0.4979, 0.0252, 0.4131, 0.4979}; //Isohedral 5
    // fetchOneFamily(IH, params, raw_points, 10, 10);
    // extrudeToMesh(raw_points, 0.1, 1.0, "IH" + std::to_string(IH) + "_10cmx10cmx10mm.obj");
}

void Tiling3D::getMeshForPrinting(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    std::vector<PointLoops> raw_points;
    // int IH = 5;
    // T params[] = {0.1224, 0.4979, 0.0252, 0.4131, 0.4979}; //Isohedral 5

    // int IH = 0;
    // T params[] = {0.1161, 0.5464, 0.4313, 0.5464}; //Isohedral 0

    // int IH = 13;
    // T params[] = {0.1, 0.2}; //Isohedral 7

    // int IH = 29;
    // T params[] = {0}; //Isohedral 29

    int IH = 6;
    T params[] = {0.5, 0.5, 0.5, 0.5, 0.5}; //Isohedral 06

    // int IH = 1;
    // T params[] = {0.207, 0.7403, 0.304, 1.2373}; //Isohedral 01

    // int IH = 2;
	// T params[] = {0.3767, 0.5949, 0, 0}; //Isohedral 02

    TV2 T1, T2;

    fetchOneFamilyFillRegion(IH, params, raw_points, 10, 10);

    T height = 1.0;

    std::vector<TV2> unique_points;
    std::vector<IdList> polygon_ids;

    for (const PointLoops& pl : raw_points)
    {
        TV2 center = TV2::Zero();
        IdList id_list;
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 pt = pl[i];
            center += pt;
            auto find_iter = std::find_if(unique_points.begin(), unique_points.end(), 
                            [&pt](const TV2 x)->bool
                               { return (x - pt).norm() < 1e-6; }
                             );
            if (find_iter == unique_points.end())
            {
                unique_points.push_back(pt);
                id_list.push_back(unique_points.size() - 1);
            }
            else
                id_list.push_back(std::distance(unique_points.begin(), find_iter));
        }

        polygon_ids.push_back(id_list);
        center /= T(pl.size() - 1);
        id_list.clear();
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 inner = center + (pl[i] - center) * 0.85;
            unique_points.push_back(inner);
            id_list.push_back(unique_points.size() - 1);
        }
        polygon_ids.push_back(id_list);
    }

    std::vector<Face> faces;
    for (int i = 0; i < polygon_ids.size() / 2; i++)
    {
        auto outer = polygon_ids[i * 2];
        auto inner = polygon_ids[i * 2 + 1];

        for (int j = 0; j < outer.size(); j++)
        {
            faces.push_back(Face(outer[j] + 1, inner[j] + 1, inner[(j + 1) % outer.size()] + 1));
            faces.push_back(Face(outer[j] + 1, inner[(j + 1) % outer.size()] + 1, outer[(j + 1) % outer.size()] + 1));
        }
    }

    std::vector<TV> vertices;
    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], 0));

    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], height));

    int nv = unique_points.size();
    int nf = faces.size();

    for (int i = 0; i < nf; i++)
    {
        faces.push_back(Face(
          faces[i][2] + nv, faces[i][1] + nv, faces[i][0] + nv
        ));
    }
    
    for (IdList id_list : polygon_ids)
    {
        for (int i = 0; i < id_list.size(); i++)
        {
            faces.push_back(Face(
                id_list[i] + 1, id_list[(i + 1)%id_list.size()] + 1, id_list[i] + nv + 1
            ));
            faces.push_back(Face(
                id_list[(i + 1)%id_list.size()] + 1, id_list[(i + 1)%id_list.size()] + nv + 1, id_list[i] + nv + 1 
            ));
        }
    }

    V.resize(vertices.size(), 3);
    tbb::parallel_for(0, (int)vertices.size(), [&](int i){
        V.row(i) = vertices[i];
    });

    F.resize(faces.size(), 3);
    C.resize(faces.size(), 3);

    tbb::parallel_for(0, (int)faces.size(), [&](int i){
        F.row(i) = faces[i];
        C.row(i) = Eigen::Vector3d(0, 0.3, 1.0);
    });
    
    std::ofstream out("test_mesh.obj");
    for (const TV& pt : vertices)
        out << "v " << pt.transpose() * 10.0 << std::endl;
    for (auto face : faces)
        out << "f " << face.transpose() << std::endl;
    out.close();

}

void Tiling3D::buildSimulationMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    std::vector<PointLoops> raw_points;
    // int IH = 5;
    // T params[] = {0.1224, 0.4979, 0.0252, 0.4131, 0.4979}; //Isohedral 5

    int IH = 0;
    T params[] = {0.1161, 0.5464, 0.4313, 0.5464}; //Isohedral 0

    // int IH = 13;
    // T params[] = {0.1, 0.2}; //Isohedral 7

    // int IH = 29;
    // T params[] = {0}; //Isohedral 29

    // int IH = 6;
    // T params[] = {0.5, 0.5, 0.5, 0.5, 0.5}; //Isohedral 06

    // int IH = 1;
    // T params[] = {0.207, 0.7403, 0.304, 1.2373}; //Isohedral 01

    // int IH = 2;
	// T params[] = {0.3767, 0.5949, 0, 0}; //Isohedral 02

    TV2 T1, T2;

    fetchOneFamilyFillRegion(IH, params, raw_points, 10, 10);

    T height = 1.0;
    T thickness = 0.04;
    int sub_divide_width = 5;
    int sub_divide_height = 5;
    std::vector<TV> mesh_vertices;
    std::vector<Face> mesh_faces;
    std::vector<TV2> unique_points;
    std::vector<IdList> polygon_ids;

    for (const PointLoops& pl : raw_points)
    {
        TV2 center = TV2::Zero();
        IdList id_list;
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 pt = pl[i];
            auto find_iter = std::find_if(unique_points.begin(), unique_points.end(), 
                            [&pt](const TV2 x)->bool
                               { return (x - pt).norm() < 1e-6; }
                             );
            if (find_iter == unique_points.end())
            {
                unique_points.push_back(pt);
                id_list.push_back(unique_points.size() - 1);
            }
            else
                id_list.push_back(std::distance(unique_points.begin(), find_iter));
        }
        polygon_ids.push_back(id_list);
    }

    struct Vertex
    {
        TV x;
        std::vector<TV> neighbors;
        std::vector<Edge> edges;
        Vertex(const TV& _x) : x(_x) {}
    };
    
    std::vector<Vertex> vertices;
    for (TV2& pt : unique_points)
    {
        vertices.push_back(Vertex(TV(pt[0], pt[1], 0.0)));
    }

    T width = 0.04; // nozzle diameter
    
    std::vector<Edge> edges;

    for (const auto& id_list : polygon_ids)
    {
        for (int i = 0; i < id_list.size(); i++)
        {
            int j = (i + 1) % id_list.size();
            Edge ei(id_list[i], id_list[j]);
            
            auto find_iter = std::find_if(edges.begin(), edges.end(), 
                [&ei](const Edge e)->bool {return (ei[0] == e[0] && ei[1] == e[1] ) 
                    || (ei[0] == e[1] && ei[1] == e[0]); });
            if (find_iter == edges.end())
            {
                edges.push_back(ei);

                vertices[id_list[i]].edges.push_back(ei);
                vertices[id_list[j]].edges.push_back(Edge(ei[1], ei[0]));
                vertices[id_list[i]].neighbors.push_back(vertices[id_list[j]].x);
                vertices[id_list[j]].neighbors.push_back(vertices[id_list[i]].x);
            }   
        }
    }

    std::unordered_map<Edge, IdList, VectorHash<2>> edge_vtx_tracker;
    
    auto appendVtxToEdge = [&](const Vertex& vtx, const Edge& edge, int vtx_idx)
    {
        if (edge_vtx_tracker.find(edge) == edge_vtx_tracker.end())
            edge_vtx_tracker[edge] = {vtx_idx};
        else
            edge_vtx_tracker[edge].push_back(vtx_idx);
    };

    // insert bisecting vector vtx
    for (const Vertex& vtx : vertices)
    {
        IdList ixn_vtx_ids;
        for (int i = 0; i < vtx.edges.size(); i++)
        {
            int j = (i + 1) % vtx.edges.size();
            
            TV vi = vtx.x, vj = vertices[vtx.edges[i][1]].x, vk = vertices[vtx.edges[j][1]].x;
            
            TV bisec;
            // only two edges here and this is the larger angle
            if (j == 0 && i == 1)
                bisec = vi - ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;
            else
                bisec = vi + ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;
            mesh_vertices.push_back(bisec);
            int vtx_idx = mesh_vertices.size() - 1;            
            ixn_vtx_ids.push_back(vtx_idx);
            appendVtxToEdge(vtx, vtx.edges[i], vtx_idx);
            appendVtxToEdge(vtx, vtx.edges[j], vtx_idx);
        }
        if (ixn_vtx_ids.size() == 3)
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
        else if (ixn_vtx_ids.size() > 3)
            std::cout << "ixn_vtx_ids.size() > 3, add more faces" << std::endl;
    }

    // thicken line
    std::vector<Edge> boundary_edges;

    for (const Edge& edge : edges)
    {
        IdList left_vtx = edge_vtx_tracker[edge];
        IdList right_vtx = edge_vtx_tracker[Edge(edge[1], edge[0])];
        TV v0 = mesh_vertices[left_vtx[0]], v1 = mesh_vertices[left_vtx[1]];
        TV v2 = mesh_vertices[right_vtx[0]], v3 = mesh_vertices[right_vtx[1]];
        TV from = vertices[edge[0]].x, to = vertices[edge[1]].x;

        TV2 intersection;
        bool insec = lineSegementsIntersect2D<T>(v0.head<2>(), v2.head<2>(), from.head<2>(), to.head<2>(), intersection);

        if (insec) // v0->v3, v1->v2
        {
            // int n_sub_div = std::floor(std::min((v3 - v0).norm(), (v2-v1).norm()) / thickness);
            int n_sub_div = sub_divide_width;
            TV delta1 = (v3 - v0) / T(n_sub_div);
            TV delta2 = (v2 - v1) / T(n_sub_div);
            
            int loop0 = left_vtx[0], loop1 = left_vtx[1];

            for (int i = 1; i < n_sub_div; i++)
            {
                mesh_vertices.push_back(v0 + i * delta1);
                int idx0 = mesh_vertices.size() - 1;
                mesh_vertices.push_back(v1 + i * delta2);    
                int idx1 = mesh_vertices.size() - 1;
                mesh_faces.push_back(Face(loop0, loop1, idx0));
                mesh_faces.push_back(Face(loop1, idx1, idx0));
                boundary_edges.push_back(Edge(loop0, idx0));
                boundary_edges.push_back(Edge(loop1, idx1));
                loop0 = idx0;
                loop1 = idx1;
            }
            mesh_faces.push_back(Face(loop0, loop1, right_vtx[0]));
            mesh_faces.push_back(Face(loop0, right_vtx[0], right_vtx[1]));
            boundary_edges.push_back(Edge(loop0, right_vtx[1]));
            boundary_edges.push_back(Edge(loop1, right_vtx[0]));
        }
        else // v0->v2, v1->v3
        {
            // int n_sub_div = std::floor(std::min((v2 - v0).norm(), (v3-v1).norm()) / thickness);
            int n_sub_div = sub_divide_width;
            TV delta1 = (v2 - v0) / T(n_sub_div);
            TV delta2 = (v3 - v1) / T(n_sub_div);
            
            int loop0 = left_vtx[0], loop1 = left_vtx[1];

            for (int i = 1; i < n_sub_div; i++)
            {
                mesh_vertices.push_back(v0 + i * delta1);
                int idx0 = mesh_vertices.size() - 1;
                mesh_vertices.push_back(v1 + i * delta2);    
                int idx1 = mesh_vertices.size() - 1;
                mesh_faces.push_back(Face(loop0, loop1, idx0));
                mesh_faces.push_back(Face(loop1, idx1, idx0));
                boundary_edges.push_back(Edge(loop0, idx0));
                boundary_edges.push_back(Edge(loop1, idx1));
                loop0 = idx0;
                loop1 = idx1;
            }
            mesh_faces.push_back(Face(loop0, loop1, right_vtx[0]));
            mesh_faces.push_back(Face(right_vtx[0], loop1, right_vtx[1]));
            boundary_edges.push_back(Edge(loop0, right_vtx[0]));
            boundary_edges.push_back(Edge(loop1, right_vtx[1]));
        }
    }

    // unify face normal due to different edge orientation
    for (Face& face : mesh_faces)
    {
        TV v0 = mesh_vertices[face[0]];
        TV v1 = mesh_vertices[face[1]];
        TV v2 = mesh_vertices[face[2]];

        if ((v2 - v0).cross(v1 - v0).dot(TV(0, 0, 1)) < 0)
            face = Face(face[1], face[0], face[2]);
    }
    

    int nv = mesh_vertices.size();
    for (const TV& vtx : mesh_vertices)
        mesh_vertices.push_back(vtx + TV(0, 0, height));
    
    int nf = mesh_faces.size();

    for (int i = 0; i < nf; i++)
    {
        mesh_faces.push_back(Face(
          mesh_faces[i][2] + nv, mesh_faces[i][1] + nv, mesh_faces[i][0] + nv
        ));
    }

    auto normal = [&](const Face& face)
    {
        TV v0 = mesh_vertices[face[0]];
        TV v1 = mesh_vertices[face[1]];
        TV v2 = mesh_vertices[face[2]];

        return (v2 - v0).cross(v1 - v0).normalized();
    };

    auto inverseNormal = [&](const Face& face)
    {
        return Face(face[1], face[0], face[2]);
    };

    for (int i = 0; i < boundary_edges.size() / 2; i++)
    {
        Edge ei = boundary_edges[i * 2 + 0];
        Edge ei_oppo = boundary_edges[i * 2 + 1];
        TV v0 = mesh_vertices[ei[0]], v1 = mesh_vertices[ei[1]];
        TV v2 = mesh_vertices[ei_oppo[0]], v3 = mesh_vertices[ei_oppo[1]];
        
        Face f0 = Face(ei[1], ei[0], ei[0] + nv);
        Face f1 = Face(ei[1], ei[0] + nv, ei[1] + nv);
        
        TV n0 = normal(f0), n1 = normal(f1);
        TV edge_vec = (v2 - v0 + TV(1e-4, 1e-4, 1e-4)).normalized();

        if (edge_vec.dot(n0) < 0)
        {
            f0 = inverseNormal(f0);
            f1 = inverseNormal(f1);
        }
        mesh_faces.push_back(f0); mesh_faces.push_back(f1);

        Face f2 = Face(ei_oppo[0], ei_oppo[1], ei_oppo[0] + nv);
        Face f3 = Face(ei_oppo[0] + nv, ei_oppo[1], ei_oppo[1] + nv);
        TV n2 = normal(f2), n3 = normal(f3);
        if (-edge_vec.dot(n2) < 0)
        {
            f2 = inverseNormal(f2);
            f3 = inverseNormal(f3);
        }
        mesh_faces.push_back(f2);
        mesh_faces.push_back(f3);
    }
    

    // std::ofstream out("test_tiling.obj");
    
    // for (const TV& vtx : mesh_vertices)
    // {
    //     out << "v " << vtx.transpose() << std::endl;
    // }
    // for (const Face& face : mesh_faces)
    //     out << "f " << face.transpose() + IV::Ones().transpose() << std::endl;

    // out.close();
    V.resize(mesh_vertices.size(), 3);
    F.resize(mesh_faces.size(), 3);
    C.resize(mesh_faces.size(), 3);

    for (int i = 0; i < mesh_vertices.size(); i++)
    {
        V.row(i) = mesh_vertices[i];    
    }
    for (int i = 0; i < mesh_faces.size(); i++)
    {
        F.row(i) = mesh_faces[i];
        C.row(i) = TV(0, 0.3, 1.0);
    }

    
}

void Tiling3D::initializeSimulationData()
{
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    buildSimulationMesh(V, F, C);
    
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TF;

    igl::copyleft::tetgen::tetrahedralize(V,F, "pq1.414Y", TV,TT,TF);
    
    solver.initializeElementData(TV, TF, TT);

}
