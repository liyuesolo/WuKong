#include "Tiling3D.h"

#include <random>
#include <cmath>
#include<fstream>

std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );

static double zeta()
{
	return dis(gen);
}

void Tiling3D::fetchOneFamily(std::vector<PointLoops>& raw_points, T width, T height)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    int IH = 0;

    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    
    size_t num_params = a_tiling.numParameters();
    // if( num_params > 1 ) {
    //     double params[ num_params ];
    //     // Get the parameters out of the tiling
    //     a_tiling.getParameters( params );
    //     // Change a parameter
    //     for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) {
    //         // if (random)
    //             // params[idx] += zeta()*0.2 - 0.1;
    //     }
    //     // Send the parameters back to the tiling
    //     a_tiling.setParameters( params );
    // }

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        // Start by making a random Bezier segment.
        ej.push_back( dvec2( 0, 0 ) );
        // ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
        // ej.push_back( 
        //     dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
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


    for( auto i : a_tiling.fillRegion(0.0, 0.0, width, height) ) {

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
        raw_points.push_back(data_points);   
    }

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
            TV2 inner = center + (pl[i] - center) * 0.95;
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
    fetchOneFamily(raw_points, 5, 5);
    extrudeToMesh(raw_points, 0.1, 0.5, "hexagon10cmx10cmx5mm.obj");
}