#include "../include/Tiling2D.h"
#include <time.h>
std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );

static double zeta()
{
    return dis(gen);
}

void Tiling2D::generate3DSandwichMesh(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename)
{
    T eps = 1e-5;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

 
    //Points
    int acc = 1;
    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2, acc++);
    
    // sandwich boxes bottom 5 6 the other two points already exist
    T dx = 0.02 * (pbc_corners[1][0] - pbc_corners[0][0]);
    gmsh::model::occ::addPoint(pbc_corners[0][0],  pbc_corners[0][1] - dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0], pbc_corners[1][1] - dx, 0, 2, acc++);
    
    // sandwich boxes top 7 8 
    gmsh::model::occ::addPoint(pbc_corners[2][0], pbc_corners[2][1] + dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0], pbc_corners[3][1] + dx, 0, 2, acc++);

    // inner lattice
    for (int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    
    //Lines
    acc = 1;

    int acc_line = 1;
    
    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);
    
    // bottom box
    gmsh::model::occ::addLine(5, 6, acc++); 
    gmsh::model::occ::addLine(6, 2, acc++); 
    gmsh::model::occ::addLine(2, 1, acc++); 
    gmsh::model::occ::addLine(1, 5, acc++);

    // top box
    gmsh::model::occ::addLine(4, 3, acc++); 
    gmsh::model::occ::addLine(3, 7, acc++); 
    gmsh::model::occ::addLine(7, 8, acc++); 
    gmsh::model::occ::addLine(8, 4, acc++);

    acc_line = 9;

    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }
    
    acc = 1;
    int acc_loop = 1;
    // clipping box
    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
    gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
    acc_loop = 13;
    // std::cout << "#polygons " << polygons.size() << std::endl;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }
    
    for (int i = 0; i < polygons.size()+3; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }
    

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+4));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);

    std::vector<std::pair<int, int>> fuse_bottom_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);

    std::vector<std::pair<int, int>> fuse_top_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy2;
    gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);

    std::vector<std::pair<int, int> > ext;
    T depth = (pbc_corners[1] - pbc_corners[0]).norm() * 0.1;
	gmsh::model::occ::extrude(fuse_top_block, 0, 0, depth, ext);

    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 5.0);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();

    gmsh::model::occ::synchronize();
    gmsh::model::mesh::generate(3);
    

    
    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
}

void Tiling2D::loadTilingStructureFromTxt(const std::string& filename,
    std::vector<std::vector<TV2>>& eigen_polygons,
    std::vector<TV2>& eigen_base)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    std::ifstream in(filename);
    int IH;
    std::string token;
    in >> token;
    in >> IH;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    in >> token;
    int num_edge_shape;
    in >> num_edge_shape;
    std::vector<dvec2> edges[num_edge_shape];
    for (int i = 0; i < num_edge_shape; i++)
    {
        std::vector<dvec2> ej;
        for (int j = 0; j < 4; j++)
        {
            T x, y;
            in >> x >> y;
            ej.push_back(dvec2(x, y));
        }
        edges[i] = ej;
    }
    
    in.close();
    std::vector<dvec2> shape;
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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<dvec2> _data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        _data_points.push_back(dvec2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            _data_points.push_back(dvec2(p1[0], p1[1]));
            _data_points.push_back(dvec2(p2[0], p2[1]));
            _data_points.push_back(dvec2(p3[0], p3[1]));
        }

        return _data_points;
    };

    std::vector<std::vector<dvec2>> polygons_v;

    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    T width = 5.0;
	T depth = 5.0;
    int ii=0;
    int extension = 4;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) 
    {
        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);

        polygons_v.push_back(outv);
    }

    int chosen_x = (max_x+min_x)/2;
    int chosen_y = (max_y+min_y)/2;

    TV2 xy, x1y, xy1, x1y1;

    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) {

        dmat3 T = M * i->getTransform();
        vector<dvec2> outv = outShapeVec( shape, T );
    
        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        if(i->getT1() == chosen_x && i->getT2() == chosen_y && i->getAspect()==0)
            xy << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y && i->getAspect()==0)
            x1y << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x && i->getT2() == chosen_y+1 && i->getAspect()==0)
            xy1 << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y+1 && i->getAspect()==0)
            x1y1 << outv[0].x, outv[0].y;		

    }

    Vector<T, 4> transf;
    transf.head<2>() = x1y - xy;
    transf.tail<2>() = xy1 - xy;
    
    T temp1 = transf[0];
    T temp2 = transf[3];

    transf[0] = transf[2];
    transf[3] = transf[1];

    transf[1] = temp2;
    transf[2] = temp1;

    if(transf[0]<0)
        transf.head<2>() *= -1;
    if(transf[3]<0)
        transf.tail<2>() *= -1;

    T temp = transf[2];
    transf[2] = transf[1];
    transf[1] = temp;

    Vector<T, 8> periodic;
    int n_unit = 2;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    TV t1 = periodic.segment<2>(2);
    TV t2 = periodic.segment<2>(6);
    TV t1_unit = t1.normalized();
    TV x_axis(1, 0);

    T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
        theta_to_x *= -1.0;
    TM2 R;
    R << std::cos(theta_to_x), -std::sin(theta_to_x), std::sin(theta_to_x), std::cos(theta_to_x);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    // periodic.segment<2>(6) = TV(0, 1) * (R * periodic.segment<2>(6)).dot(TV(0, 1));
    periodic.segment<2>(6) = TV(0, 1) * periodic.segment<2>(2).norm();
    periodic[4] = periodic[2]; periodic[5] = periodic[7];


    T distance = -2.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    // std::ofstream out2("tiling_unit_clip_in_x.obj");
    // for (int i = 0; i < 4; i++)
    //     out2 << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // for (auto polygon : final_shape)
    // {
    //     for (auto vtx : polygon)
    //     {
    //         out2 << "v " << vtx.X / mult << " " << vtx.Y / mult << " 0" << std::endl;
    //     }
    // }
    // out2 << "l 1 2" << std::endl;
    // out2 << "l 2 3" << std::endl;
    // out2 << "l 3 4" << std::endl;
    // out2 << "l 4 1" << std::endl;
    // int cnt = 5;
    // // int cnt = 1;
    // for (auto polygon : final_shape)
    // {
    //     for (int i = 0; i < polygon.size(); i++)
    //     {
    //         int j = (i + 1) % polygon.size();
    //         out2 << "l " << cnt + i << " " << cnt + j << std::endl;
    //     }
    //     cnt += polygon.size();
    // }
    // out2.close();
    
    eigen_polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-eigen_polygons[i].front()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-eigen_polygons[i].back()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else
                eigen_polygons[i].push_back(cur_point);	

            //eigen_polygons[i].push_back(Vector2a(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);

    eigen_base.push_back(TV2(periodic[0], periodic[1]));
    eigen_base.push_back(TV2(periodic[2], periodic[3]));
    eigen_base.push_back(TV2(periodic[4], periodic[5]));
    eigen_base.push_back(TV2(periodic[6], periodic[7]));
}

void Tiling2D::sampleRegion(int IH, 
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, 
        const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    std::ofstream out(filename);
    out << "IH " << IH << std::endl;
    out << "num_params " << int(num_params) << " ";
    double new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];
    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        ej.push_back( dvec2( 0, 0.0 ) );
        ej.push_back( dvec2( eij[0], eij[1] ) );
        ej.push_back( dvec2( eij[2], eij[3] ) );
        ej.push_back( dvec2( 1.0, 0.0 ) );
        
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
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<dvec2> _data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        _data_points.push_back(dvec2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            _data_points.push_back(dvec2(p1[0], p1[1]));
            _data_points.push_back(dvec2(p2[0], p2[1]));
            _data_points.push_back(dvec2(p3[0], p3[1]));
        }

        return _data_points;
    };

    std::vector<std::vector<dvec2>> polygons_v;

    T width = 5.0;
	T depth = 5.0;
    int ii=0;
    int extension = 2;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) 
    {
        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        polygons_v.push_back(outv);
    }

    T min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            max_x = std::max(max_x, polygons_v[i][j][0]);
            min_x = std::min(min_x, polygons_v[i][j][0]);
            min_y = std::min(min_y, polygons_v[i][j][1]);
            max_y = std::max(max_y, polygons_v[i][j][1]);
        }
    }
    T dx = max_x - min_x, dy = max_y - min_y;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            polygons_v[i][j][0] = polygons_v[i][j][0] - min_x - 0.5 * dx;
            polygons_v[i][j][1] = polygons_v[i][j][1] - min_y - 0.5 * dy;
        }
    }

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = TV(polygons_v[i][j][0], polygons_v[i][j][1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }

    T distance = -2.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    eigen_polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-eigen_polygons[i].front()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-eigen_polygons[i].back()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else
                eigen_polygons[i].push_back(cur_point);	

            //eigen_polygons[i].push_back(Vector2a(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);
    T box_length = 0.1;
    eigen_base.push_back(TV2(-box_length * dx, -box_length * dy));
    eigen_base.push_back(TV2(box_length * dx, -box_length * dy));
    eigen_base.push_back(TV2(box_length * dx, box_length * dy));
    eigen_base.push_back(TV2(-box_length * dx, box_length * dy));

    // std::ofstream out2("tiling_unit_clip_in_x.obj");
    // // for (int i = 0; i < 4; i++)
    // //     out2 << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // for (auto polygon : final_shape)
    // {
    //     for (auto vtx : polygon)
    //     {
    //         out2 << "v " << vtx.X / mult << " " << vtx.Y / mult << " 0" << std::endl;
    //     }
    // }
    // // out2 << "l 1 2" << std::endl;
    // // out2 << "l 2 3" << std::endl;
    // // out2 << "l 3 4" << std::endl;
    // // out2 << "l 4 1" << std::endl;
    // // int cnt = 5;
    // int cnt = 1;
    // for (auto polygon : final_shape)
    // {
    //     for (int i = 0; i < polygon.size(); i++)
    //     {
    //         int j = (i + 1) % polygon.size();
    //         out2 << "l " << cnt + i << " " << cnt + j << std::endl;
    //     }
    //     cnt += polygon.size();
    // }
    // out2.close();
}

void Tiling2D::sampleSandwichFromOneFamilyFromDiffParamsDilation(int IH, 
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    std::ofstream out(filename);
    out << "IH " << IH << std::endl;
    out << "num_params " << int(num_params) << " ";
    double new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];
    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        ej.push_back( dvec2( 0, 0.0 ) );
        ej.push_back( dvec2( eij[0], eij[1] ) );
        ej.push_back( dvec2( eij[2], eij[3] ) );
        ej.push_back( dvec2( 1.0, 0.0 ) );
        
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
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<dvec2> _data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        _data_points.push_back(dvec2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            _data_points.push_back(dvec2(p1[0], p1[1]));
            _data_points.push_back(dvec2(p2[0], p2[1]));
            _data_points.push_back(dvec2(p3[0], p3[1]));
        }

        return _data_points;
    };

    std::vector<std::vector<dvec2>> polygons_v;

    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    T width = 5.0;
	T depth = 5.0;
    int ii=0;
    int extension = 4;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) 
    {
        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);

        polygons_v.push_back(outv);
    }

    int chosen_x = (max_x+min_x)/2;
    int chosen_y = (max_y+min_y)/2;

    TV2 xy, x1y, xy1, x1y1;

    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) {

        dmat3 T = M * i->getTransform();
        vector<dvec2> outv = outShapeVec( shape, T );
    
        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        if(i->getT1() == chosen_x && i->getT2() == chosen_y && i->getAspect()==0)
            xy << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y && i->getAspect()==0)
            x1y << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x && i->getT2() == chosen_y+1 && i->getAspect()==0)
            xy1 << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y+1 && i->getAspect()==0)
            x1y1 << outv[0].x, outv[0].y;		

    }

    Vector<T, 4> transf;
    transf.head<2>() = x1y - xy;
    transf.tail<2>() = xy1 - xy;
    
    T temp1 = transf[0];
    T temp2 = transf[3];

    transf[0] = transf[2];
    transf[3] = transf[1];

    transf[1] = temp2;
    transf[2] = temp1;

    if(transf[0]<0)
        transf.head<2>() *= -1;
    if(transf[3]<0)
        transf.tail<2>() *= -1;

    T temp = transf[2];
    transf[2] = transf[1];
    transf[1] = temp;

    Vector<T, 8> periodic;
    int n_unit = 1;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);
    

    TV t1 = periodic.segment<2>(2);
    TV t2 = periodic.segment<2>(6);

    TV t1_unit = t1.normalized();
    TV x_axis(1, 0);

    T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
        theta_to_x *= -1.0;
    TM2 R;
    R << std::cos(theta_to_x), -std::sin(theta_to_x), std::sin(theta_to_x), std::cos(theta_to_x);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            // TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            TV curr = TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    
    // periodic.segment<2>(2) = R * periodic.segment<2>(2);
    // periodic.segment<2>(6) = TV(0, 1) * periodic.segment<2>(2).norm();
    // periodic[4] = periodic[2]; periodic[5] = periodic[7];   


    // T distance = -2.0;//default
    T distance = -2.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    std::ofstream out2("tiling_unit_clip_in_x.obj");
    for (int i = 0; i < 4; i++)
        out2 << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    for (auto polygon : final_shape)
    {
        for (auto vtx : polygon)
        {
            out2 << "v " << vtx.X / mult << " " << vtx.Y / mult << " 0" << std::endl;
        }
    }
    out2 << "l 1 2" << std::endl;
    out2 << "l 2 3" << std::endl;
    out2 << "l 3 4" << std::endl;
    out2 << "l 4 1" << std::endl;
    int cnt = 5;
    // int cnt = 1;
    for (auto polygon : final_shape)
    {
        for (int i = 0; i < polygon.size(); i++)
        {
            int j = (i + 1) % polygon.size();
            out2 << "l " << cnt + i << " " << cnt + j << std::endl;
        }
        cnt += polygon.size();
    }
    out2.close();
    
    eigen_polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-eigen_polygons[i].front()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-eigen_polygons[i].back()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else
                eigen_polygons[i].push_back(cur_point);	

            //eigen_polygons[i].push_back(Vector2a(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);

    eigen_base.push_back(TV2(periodic[0], periodic[1]));
    eigen_base.push_back(TV2(periodic[2], periodic[3]));
    eigen_base.push_back(TV2(periodic[4], periodic[5]));
    eigen_base.push_back(TV2(periodic[6], periodic[7]));
}

void Tiling2D::sampleSandwichFromOneFamilyFromParamsDilation(int IH, 
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, const Vector<T, 4>& eij,
        bool save_to_file, std::string filename)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    std::ofstream out(filename);
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    out << "IH " << IH << std::endl;
    size_t num_params = a_tiling.numParameters();
    
    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";

    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        ej.push_back( dvec2( 0, 0.0 ) );
        ej.push_back( dvec2( eij[0], eij[1] ) );
        ej.push_back( dvec2( eij[2], eij[3] ) );
        ej.push_back( dvec2( 1.0, 0.0 ) );
        
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
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<dvec2> _data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        _data_points.push_back(dvec2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            _data_points.push_back(dvec2(p1[0], p1[1]));
            _data_points.push_back(dvec2(p2[0], p2[1]));
            _data_points.push_back(dvec2(p3[0], p3[1]));
        }

        return _data_points;
    };

    std::vector<std::vector<dvec2>> polygons_v;

    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    T width = 5.0;
	T depth = 5.0;
    int ii=0;
    int extension = 4;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) 
    {
        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);

        polygons_v.push_back(outv);
    }

    int chosen_x = (max_x+min_x)/2;
    int chosen_y = (max_y+min_y)/2;

    TV2 xy, x1y, xy1, x1y1;

    for( auto i : a_tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) {

        dmat3 T = M * i->getTransform();
        vector<dvec2> outv = outShapeVec( shape, T );
    
        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        if(i->getT1() == chosen_x && i->getT2() == chosen_y && i->getAspect()==0)
            xy << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y && i->getAspect()==0)
            x1y << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x && i->getT2() == chosen_y+1 && i->getAspect()==0)
            xy1 << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y+1 && i->getAspect()==0)
            x1y1 << outv[0].x, outv[0].y;		

    }

    Vector<T, 4> transf;
    transf.head<2>() = x1y - xy;
    transf.tail<2>() = xy1 - xy;
    
    T temp1 = transf[0];
    T temp2 = transf[3];

    transf[0] = transf[2];
    transf[3] = transf[1];

    transf[1] = temp2;
    transf[2] = temp1;

    if(transf[0]<0)
        transf.head<2>() *= -1;
    if(transf[3]<0)
        transf.tail<2>() *= -1;

    T temp = transf[2];
    transf[2] = transf[1];
    transf[1] = temp;

    Vector<T, 8> periodic;
    int n_unit = 2;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    TV t1 = periodic.segment<2>(2);
    TV t2 = periodic.segment<2>(6);
    TV t1_unit = t1.normalized();
    TV x_axis(1, 0);

    T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
        theta_to_x *= -1.0;
    TM2 R;
    R << std::cos(theta_to_x), -std::sin(theta_to_x), std::sin(theta_to_x), std::cos(theta_to_x);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    // periodic.segment<2>(6) = TV(0, 1) * (R * periodic.segment<2>(6)).dot(TV(0, 1));
    periodic.segment<2>(6) = TV(0, 1) * periodic.segment<2>(2).norm();
    periodic[4] = periodic[2]; periodic[5] = periodic[7];


    // T distance = -2.0;//default
    T distance = -1.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    // std::ofstream out2("tiling_unit_clip_in_x.obj");
    // for (int i = 0; i < 4; i++)
    //     out2 << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // for (auto polygon : final_shape)
    // {
    //     for (auto vtx : polygon)
    //     {
    //         out2 << "v " << vtx.X / mult << " " << vtx.Y / mult << " 0" << std::endl;
    //     }
    // }
    // out2 << "l 1 2" << std::endl;
    // out2 << "l 2 3" << std::endl;
    // out2 << "l 3 4" << std::endl;
    // out2 << "l 4 1" << std::endl;
    // int cnt = 5;
    // // int cnt = 1;
    // for (auto polygon : final_shape)
    // {
    //     for (int i = 0; i < polygon.size(); i++)
    //     {
    //         int j = (i + 1) % polygon.size();
    //         out2 << "l " << cnt + i << " " << cnt + j << std::endl;
    //     }
    //     cnt += polygon.size();
    // }
    // out2.close();
    
    eigen_polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-eigen_polygons[i].front()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-eigen_polygons[i].back()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else
                eigen_polygons[i].push_back(cur_point);	

            //eigen_polygons[i].push_back(Vector2a(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);

    eigen_base.push_back(TV2(periodic[0], periodic[1]));
    eigen_base.push_back(TV2(periodic[2], periodic[3]));
    eigen_base.push_back(TV2(periodic[4], periodic[5]));
    eigen_base.push_back(TV2(periodic[6], periodic[7]));
}

void Tiling2D::fetchSandwichFromOneFamilyFromParamsDilation(int IH, 
        std::vector<T> params,
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, bool random,
        bool save_to_file, std::string filename)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    std::ofstream out(filename);
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    out << "IH " << IH << std::endl;
    size_t num_params = a_tiling.numParameters();
    out << "num_params " << num_params << " ";
    if (num_params != params.size() && !random)
        return;    
    if( num_params > 1 ) 
    {
        double new_params[ num_params ];
        
        a_tiling.getParameters( new_params );
        // Change a parameter
        for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) {
            if (random)
                new_params[idx] += zeta()*0.2 - 0.1;
            else
                new_params[idx] = params[idx];
            out << new_params[idx] << " ";
        }
        a_tiling.setParameters( new_params );
    }
    out << std::endl;
    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        // Start by making a random Bezier segment.
        if (random)
        {
            ej.push_back( dvec2( 0, 0 ) );
            T x0_ran = zeta(); T y0_ran = zeta();
            T x1_ran = 1.0 - x0_ran; T y1_ran = 1.0 - y0_ran;
            ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
            ej.push_back( 
                dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
            ej.push_back( dvec2( 1, 0 ) );
        }
        else
        {
            ej.push_back( dvec2( 0, 0.0 ) );
            ej.push_back( dvec2( 0.25, 0.0 ) );
            ej.push_back( dvec2( 0.75, 0.0 ) );
            ej.push_back( dvec2( 1.0, 0.0 ) );
        }
        
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
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<dvec2> _data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        _data_points.push_back(dvec2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            _data_points.push_back(dvec2(p1[0], p1[1]));
            _data_points.push_back(dvec2(p2[0], p2[1]));
            _data_points.push_back(dvec2(p3[0], p3[1]));
        }

        return _data_points;
    };



    std::vector<std::vector<dvec2>> polygons_v;

    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    T width = 5.0;
	T depth = 5.0;
    int ii=0;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : a_tiling.fillRegion( -2 * width, -2 * depth, 2 * width, 2 * depth ) ) 
    {
        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);

        polygons_v.push_back(outv);
    }

    int chosen_x = (max_x+min_x)/2;
    int chosen_y = (max_y+min_y)/2;

    TV2 xy, x1y, xy1, x1y1;

    for( auto i : a_tiling.fillRegion( -2 * width, -2 * depth, 2 * width, 2 * depth ) ) {

        dmat3 T = M * i->getTransform();
        vector<dvec2> outv = outShapeVec( shape, T );
    
        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        if(i->getT1() == chosen_x && i->getT2() == chosen_y && i->getAspect()==0)
            xy << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y && i->getAspect()==0)
            x1y << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x && i->getT2() == chosen_y+1 && i->getAspect()==0)
            xy1 << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y+1 && i->getAspect()==0)
            x1y1 << outv[0].x, outv[0].y;		

    }

    Vector<T, 4> transf;
    transf.head<2>() = x1y - xy;
    transf.tail<2>() = xy1 - xy;
    
    T temp1 = transf[0];
    T temp2 = transf[3];

    transf[0] = transf[2];
    transf[3] = transf[1];

    transf[1] = temp2;
    transf[2] = temp1;

    if(transf[0]<0)
        transf.head<2>() *= -1;
    if(transf[3]<0)
        transf.tail<2>() *= -1;

    T temp = transf[2];
    transf[2] = transf[1];
    transf[1] = temp;

    Vector<T, 8> periodic;
    int n_unit = 3;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    TV t1 = periodic.segment<2>(2);
    TV t2 = periodic.segment<2>(6);
    TV t1_unit = t1.normalized();
    TV x_axis(1, 0);

    T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
        theta_to_x *= -1.0;
    TM2 R;
    R << std::cos(theta_to_x), -std::sin(theta_to_x), std::sin(theta_to_x), std::cos(theta_to_x);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    periodic.segment<2>(6) = TV(0, 1) * (R * periodic.segment<2>(6)).dot(TV(0, 1));
    periodic[4] = periodic[2]; periodic[5] = periodic[7];


    T distance = -2.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);


    // std::ofstream out("tiling_unit_clip_in_x.obj");
    // for (int i = 0; i < 4; i++)
    //     out << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // for (auto polygon : final_shape)
    // {
    //     for (auto vtx : polygon)
    //     {
    //         out << "v " << vtx.X / mult << " " << vtx.Y / mult << " 0" << std::endl;
    //     }
    // }
    // out << "l 1 2" << std::endl;
    // out << "l 2 3" << std::endl;
    // out << "l 3 4" << std::endl;
    // out << "l 4 1" << std::endl;
    // int cnt = 5;
    // // int cnt = 1;
    // for (auto polygon : final_shape)
    // {
    //     for (int i = 0; i < polygon.size(); i++)
    //     {
    //         int j = (i + 1) % polygon.size();
    //         out << "l " << cnt + i << " " << cnt + j << std::endl;
    //     }
    //     cnt += polygon.size();
    // }
    // out.close();
    
    eigen_polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-eigen_polygons[i].front()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-eigen_polygons[i].back()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else
                eigen_polygons[i].push_back(cur_point);	

            //eigen_polygons[i].push_back(Vector2a(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);

    eigen_base.push_back(TV2(periodic[0], periodic[1]));
    eigen_base.push_back(TV2(periodic[2], periodic[3]));
    eigen_base.push_back(TV2(periodic[4], periodic[5]));
    eigen_base.push_back(TV2(periodic[6], periodic[7]));
}


void Tiling2D::generateOneStructure()
{
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/";
    int IH = 0;
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    double new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    // cubic_weights << 0.25, 0.25, 0.75, 0.75;
    T x0_rand = zeta(), y0_rand = zeta();
    // cubic_weights << x0_rand, y0_rand, x0_rand + (1.0 - x0_rand) * zeta(), y0_rand + (1.0 - y0_rand) * zeta();
    // cubic_weights << 0.46529, 0.134614, 0.787798, 0.638062;
    cubic_weights << 0.25, 0, 0.75, 0;
    sampleSandwichFromOneFamilyFromDiffParamsDilation(IH, polygons, pbc_corners, params,
        cubic_weights, data_folder + "a_structure.txt");
    generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
        data_folder + "a_structure.vtk");   
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", true);
}

void Tiling2D::generateOneNonperiodicStructure()
{
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/";
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    int tiling_idx = 0;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ tiling_idx ] );
    int num_params = a_tiling.numParameters();
    double new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    std::vector<T> diff_params = params;
    for (int k = 0; k < num_params; k++)
    {
        T rand_params = 0.1 * (zeta() * 2.0 - 1.0);
        diff_params[k] = std::max(std::min(params[k] + rand_params, 0.92), 0.08);
    }
    // cubic_weights << 0.25, 0.25, 0.75, 0.75;
    T x0_rand = zeta(), y0_rand = zeta();
    // cubic_weights << x0_rand, y0_rand, x0_rand + (1.0 - x0_rand) * zeta(), y0_rand + (1.0 - y0_rand) * zeta();
    // cubic_weights << 0.46529, 0.134614, 0.787798, 0.638062;
    cubic_weights << 0.25, 0, 0.75, 0;
    // sampleSandwichFromOneFamilyFromDiffParamsDilation(tiling_idx, polygons, pbc_corners, params, cubic_weights, 
    //     data_folder + "a_structure.txt");
    sampleRegion(tiling_idx, polygons, pbc_corners, diff_params, cubic_weights, 
        data_folder + "a_structure.txt");
    
    generateSandwichMeshNonPeridoic(polygons, pbc_corners, true, 
        data_folder + "a_structure.vtk");   
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", false);
}

void Tiling2D::generateSandwichBatchChangingTilingParams()
{
    std::srand( (unsigned)time( NULL ) );
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKDiffParams/";
    int cnt = 0;
    // tbb::parallel_for(0, 81, [&](int i)
    for (int i = 47; i < 81; i++)
    {
        csk::IsohedralTiling a_tiling( csk::tiling_types[ i ] );
        int num_params = a_tiling.numParameters();
        double new_params[ num_params ];
        a_tiling.getParameters( new_params );
        std::vector<T> params(num_params);
        for (int j = 0; j < num_params;j ++)
            params[j] = new_params[j];
        int offset = 1 + 10 + 5;// + num_params;
        // default parameters
        std::vector<std::vector<TV2>> polygons;
        std::vector<TV2> pbc_corners; 
        Vector<T, 4> cubic_weights;
        cubic_weights << 0.25, 0.0, 0.75, 0.0;
        sampleSandwichFromOneFamilyFromDiffParamsDilation(i, polygons, pbc_corners, params, cubic_weights, 
             data_folder + std::to_string(i * offset) + ".txt");
        generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
            data_folder + std::to_string(i * offset) + ".vtk");   

        // sample tiling parameters
        for (int j = 0; j < 10; j++)
        {
            polygons.clear(); pbc_corners.clear();
            std::vector<T> diff_params = params;
            for (int k = 0; k < num_params; k++)
            {
                T rand_params = 0.1 * (zeta() * 2.0 - 1.0);
                diff_params[k] = std::max(std::min(params[k] + rand_params, 0.92), 0.08);
            }
            sampleSandwichFromOneFamilyFromDiffParamsDilation(i, polygons, pbc_corners, diff_params, cubic_weights, 
                data_folder + std::to_string(i * offset + 1 + j) + ".txt");
            generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
                data_folder + std::to_string(i * offset + 1 + j) + ".vtk");   
        }   
        
        std::vector<Vector<T, 4>> cubic_weights_hardcoded(5);
        cubic_weights_hardcoded[0] << 0.25, 0.25, 0.5, 0.25;
        cubic_weights_hardcoded[1] << 0.25, 0.25, 0.25, 0.75;
        cubic_weights_hardcoded[2] << 0.25, 0.5, 0.5, 0.75;
        cubic_weights_hardcoded[3] << 0.5, 0.25, 0.75, 0.5;
        cubic_weights_hardcoded[4] << 0.5, 0.5, 0.75, 0.75;

        for (int j = 0; j < 5; j++)
        {
            polygons.clear(); pbc_corners.clear();
            cubic_weights = cubic_weights_hardcoded[j];
            // cubic_weights = Vector<T, 4>();
            // T x0_rand = 0.3 + 0.15 * (zeta() * 2.0 - 1.0), y0_rand = 0.3 + 0.15 * (zeta() * 2.0 - 1.0);
            // T x1_rand = 0.7 + 0.15 * (zeta() * 2.0 - 1.0), y1_rand = 0.7 + 0.15 * (zeta() * 2.0 - 1.0);
            // cubic_weights << x0_rand, y0_rand, x1_rand, y1_rand;
            sampleSandwichFromOneFamilyFromDiffParamsDilation(i, polygons, pbc_corners, params, cubic_weights, 
                data_folder + std::to_string(i * offset + 11 + j) + ".txt");
            generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
                data_folder + std::to_string(i * offset + 11 + j) + ".vtk");   
            
        }

        // for (int j = 0; j < num_params; j++)
        // {
        //     polygons.clear(); pbc_corners.clear();
        //     cubic_weights = Vector<T, 4>();
        //     std::vector<T> diff_params = params;
        //     for (int k = 0; k < num_params; k++)
        //     {
        //         T rand_params = 0.1 * (zeta() * 2.0 - 1.0);
        //         diff_params[k] = std::max(std::min(params[k] + rand_params, 0.92), 0.08);
        //     }

        //     T x0_rand = 0.3 + 0.15 * (zeta() * 2.0 - 1.0), y0_rand = 0.3 + 0.15 * (zeta() * 2.0 - 1.0);
        //     T x1_rand = 0.7 + 0.15 * (zeta() * 2.0 - 1.0), y1_rand = 0.7 + 0.15 * (zeta() * 2.0 - 1.0);
        //     cubic_weights << x0_rand, y0_rand, x1_rand, y1_rand;
            
        //     sampleSandwichFromOneFamilyFromDiffParamsDilation(i, polygons, pbc_corners, diff_params, cubic_weights, 
        //         data_folder + std::to_string(i * offset + num_params * 2 + 5 + j) + ".txt");
        //     generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
        //         data_folder + std::to_string(i * offset + num_params * 2 + 5 + j) + ".vtk");   
            
        // }
    }   
}

void Tiling2D::generateSandwichStructureBatch()
{
    // 81 tiling families
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/";
    int cnt = 0;
    // tbb::parallel_for(0, 81, [&](int i)
    for (int i = 70; i < 81; i++)
    {
        int IH = csk::tiling_types[i];
        std::vector<T> ws = {0.1, 0.2, 0.3, 0.4};
        int n_tiling_this_family = 10;
        for (int j = 0; j < ws.size(); j++)
        {
            T w = ws[j];
            std::vector<std::vector<TV2>> polygons;
            std::vector<TV2> pbc_corners; 
            Vector<T, 4> cubic_weights;
            cubic_weights << w, w, 1.0 - w, 1.0 - w;
            sampleSandwichFromOneFamilyFromParamsDilation(IH, polygons, pbc_corners, cubic_weights, 
                true, data_folder + std::to_string(i * (n_tiling_this_family + ws.size()) + j) + ".txt");
            generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
                data_folder + std::to_string(i * (n_tiling_this_family + ws.size()) + j) + ".vtk");   
        }

        for (int j = 0; j < n_tiling_this_family; j++)
        {
            std::vector<std::vector<TV2>> polygons;
            std::vector<TV2> pbc_corners; 
            Vector<T, 4> cubic_weights;
            T x0_rand = zeta(), y0_rand = zeta();
            cubic_weights << x0_rand, y0_rand, x0_rand + (1.0 - x0_rand) * zeta(), y0_rand + (1.0 - y0_rand) * zeta();
            sampleSandwichFromOneFamilyFromParamsDilation(IH, polygons, pbc_corners, cubic_weights, 
                true, data_folder + std::to_string(i * (n_tiling_this_family + ws.size()) + ws.size() + j) + ".txt");
            generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
                data_folder + std::to_string(i * (n_tiling_this_family + ws.size()) + ws.size() + j) + ".vtk");   
            cnt++;
        }
    }
    // );
    // for (int i = 0; i < 81; i++)
    // {
    //     int IH = csk::tiling_types[i];
    //     for (T w : {0.1, 0.2, 0.3, 0.4})
    //     {
    //         std::vector<std::vector<TV2>> polygons;
    //         std::vector<TV2> pbc_corners; 
    //         Vector<T, 4> cubic_weights;
    //         cubic_weights << w, w, 1.0 - w, 1.0 - w;
    //         sampleSandwichFromOneFamilyFromParamsDilation(IH, polygons, pbc_corners, cubic_weights, 
    //             true, data_folder + std::to_string(cnt) + ".txt");
    //         generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
    //             data_folder + std::to_string(cnt) + ".vtk");   
    //         cnt++;
    //     }

    //     int n_tiling_this_family = 10;
    //     for (int j = 0; j < n_tiling_this_family; j++)
    //     {
    //         std::vector<std::vector<TV2>> polygons;
    //         std::vector<TV2> pbc_corners; 
    //         Vector<T, 4> cubic_weights;
    //         T x0_rand = zeta(), y0_rand = zeta();
    //         cubic_weights << x0_rand, y0_rand, x0_rand + (1.0 - x0_rand) * zeta(), y0_rand + (1.0 - y0_rand) * zeta();
    //         sampleSandwichFromOneFamilyFromParamsDilation(IH, polygons, pbc_corners, cubic_weights, 
    //             true, data_folder + std::to_string(cnt) + ".txt");
    //         generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
    //             data_folder + std::to_string(cnt) + ".vtk");   
    //         cnt++;
    //     }
        
    //     break;
    // }
}


void Tiling2D::fetchUnitCellFromOneFamily(int IH, std::vector<std::vector<TV2>>& eigen_polygons,
    std::vector<TV2>& eigen_base, int n_unit, bool random)
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
            //     params[idx] += zeta()*0.2 - 0.1;
        }
        // Send the parameters back to the tiling
        a_tiling.setParameters( params );
    }

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        // Start by making a random Bezier segment.
        if (random)
        {
            // ej.push_back( dvec2( 0, 0 ) );
            // ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
            // ej.push_back( 
            //     dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
            // ej.push_back( dvec2( 1, 0 ) );
        }
        else
        {
            ej.push_back( dvec2( 0, 0 ) );
            ej.push_back( dvec2( 0.25, 0 ) );
            ej.push_back( dvec2( 0.75, 0 ) );
            ej.push_back( dvec2( 1, 0 ) );
        }

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
    std::vector<dvec2> shape;

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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<dvec2> _data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        _data_points.push_back(dvec2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            _data_points.push_back(dvec2(p1[0], p1[1]));
            _data_points.push_back(dvec2(p2[0], p2[1]));
            _data_points.push_back(dvec2(p3[0], p3[1]));
        }

        return _data_points;
    };


    // dvec2 p = dvec3( shape.back(), 1.0 );
    // data_points.push_back(TV2(p[0], p[1]));
    // // std::cout << p[0] << " " << p[1] << std::endl;
    // for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
    //     dvec2 p1 = dvec3( shape[idx], 1.0 );
    //     dvec2 p2 = dvec3( shape[idx+1], 1.0 );
    //     dvec2 p3 = dvec3( shape[idx+2], 1.0 );

    //     data_points.push_back(TV2(p1[0], p1[1]));
    //     data_points.push_back(TV2(p2[0], p2[1]));
    //     data_points.push_back(TV2(p3[0], p3[1]));        
    // }

    std::vector<std::vector<dvec2>> polygons_v;

    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    T width = 5.0;
	T depth = 5.0;
    int ii=0;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : a_tiling.fillRegion( -width, -depth, width, depth ) ) 
    {
    //for(auto i : t.shape()) {
    //	
        //if(ii++==10)
        //	break;

        //std::cout << "Ts " << i->getT1() << " " << i->getT2() << " " << i->getAspect() << std::endl;

        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);

        polygons_v.push_back(outv);
    }

    int chosen_x = (max_x+min_x)/2;
    int chosen_y = (max_y+min_y)/2;

    TV2 xy, x1y, xy1, x1y1;

    for( auto i : a_tiling.fillRegion( -width, -depth, width, depth ) ) {

        dmat3 T = M * i->getTransform();
        vector<dvec2> outv = outShapeVec( shape, T );
    
        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        if(i->getT1() == chosen_x && i->getT2() == chosen_y && i->getAspect()==0)
            xy << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y && i->getAspect()==0)
            x1y << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x && i->getT2() == chosen_y+1 && i->getAspect()==0)
            xy1 << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y+1 && i->getAspect()==0)
            x1y1 << outv[0].x, outv[0].y;		

    }

    Vector<T, 4> transf;
    transf.head<2>() = x1y - xy;
    transf.tail<2>() = xy1 - xy;
    
    // if(transf[0]==0 && transf[3]==0)
    {
        T temp1 = transf[0];
        T temp2 = transf[3];

        transf[0] = transf[2];
        transf[3] = transf[1];

        transf[1] = temp2;
        transf[2] = temp1;
    }

    if(transf[0]<0)
        transf.head<2>() *= -1;
    if(transf[3]<0)
        transf.tail<2>() *= -1;

    T temp = transf[2];
    transf[2] = transf[1];
    transf[1] = temp;

    Vector<T, 8> periodic;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            polygons[i] << ClipperLib::IntPoint((polygons_v[i][j][0]-xy[0])*mult, 
                (polygons_v[i][j][1]-xy[1])*mult);
            // std::cout << " " << polygons[i][j];
        }
        // std::cout << std::endl;
    }
    
    T distance = -2.0;
    ClipperLib::Paths clip(1);
	clip[0] << ClipperLib::IntPoint(periodic[0]*mult,periodic[1]*mult) 
        << ClipperLib::IntPoint(periodic[2]*mult, periodic[3]*mult) 
        << ClipperLib::IntPoint(periodic[4]*mult, periodic[5]*mult) 
        << ClipperLib::IntPoint(periodic[6]*mult, periodic[7]*mult);
    ClipperLib::Paths final_shape;

    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    ClipperLib::Paths offset_polygons;

    c.Execute(offset_polygons, distance*mult);
    
    ClipperLib::Clipper cl;
    // cl.AddPaths(offset_polygons, ClipperLib::ptSubject, true);
    // cl.AddPaths(clip, ClipperLib::ptClip, true);
    cl.AddPaths(clip, ClipperLib::ptSubject, true);
    cl.AddPaths(offset_polygons, ClipperLib::ptClip, true);
    
    cl.Execute(ClipperLib::ctDifference, final_shape, ClipperLib::pftNonZero, ClipperLib::pftNonZero);
    
    // std::ofstream out("tiling_unit_clip.obj");
    // for (int i = 0; i < 4; i++)
    //     out << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // for (auto polygon : final_shape)
    // {
    //     for (auto vtx : polygon)
    //     {
    //         out << "v " << vtx.X << " " << vtx.Y << " 0" << std::endl;
    //     }
    // }
    // out << "l 1 2" << std::endl;
    // out << "l 2 3" << std::endl;
    // out << "l 3 4" << std::endl;
    // out << "l 4 1" << std::endl;
    // int cnt = 5;
    // for (auto polygon : final_shape)
    // {
    //     for (int i = 0; i < polygon.size(); i++)
    //     {
    //         int j = (i + 1) % polygon.size();
    //         out << "l " << cnt + i << " " << cnt + j << std::endl;
    //     }
    //     cnt += polygon.size();
    // }
    // out.close();

    eigen_polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-eigen_polygons[i].front()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-eigen_polygons[i].back()).norm()>1e-4)
                    eigen_polygons[i].push_back(cur_point);
            }
            else
                eigen_polygons[i].push_back(cur_point);	

            //eigen_polygons[i].push_back(Vector2a(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);

    eigen_base.push_back(TV2(periodic[0], periodic[1]));
    eigen_base.push_back(TV2(periodic[2], periodic[3]));
    eigen_base.push_back(TV2(periodic[4], periodic[5]));
    eigen_base.push_back(TV2(periodic[6], periodic[7]));

    // std::ofstream out("tiling_unit_clip.obj");
    // // for (int i = 0; i < 4; i++)
    // //     out << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // int cnt = 1;
    // for (auto polygon : eigen_polygons)
    // {
    //     for (auto vtx : polygon)
    //     {
    //         out << "v " << vtx.transpose() << " 0" << std::endl;
    //     }
    // }
    // for (auto polygon : eigen_polygons)
    // {
    //     for (int i = 0; i < polygon.size(); i++)
    //     {
    //         int j = (i + 1) % polygon.size();
    //         out << "l " << cnt + i << " " << cnt + j << std::endl;
    //     }
    //     cnt += polygon.size();
    // }
    // // out << "l 1 2" << std::endl;
    // // out << "l 2 3" << std::endl;
    // // out << "l 3 4" << std::endl;
    // // out << "l 4 1" << std::endl;
    // out.close();
    
    
}

void Tiling2D::extrudeToMesh(const std::string& tiling_param, const std::string& mesh3d)
{
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    loadTilingStructureFromTxt(tiling_param, polygons, pbc_corners);
    generate3DSandwichMesh(polygons, pbc_corners, true, mesh3d);
}

void Tiling2D::generateSandwichMeshNonPeridoic(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

 
    //Points
    int acc = 1;
    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2, acc++);
    
    // sandwich boxes bottom 5 6 the other two points already exist
    T dx = 0.05 * (pbc_corners[1][0] - pbc_corners[0][0]);
    gmsh::model::occ::addPoint(pbc_corners[0][0],  pbc_corners[0][1] - dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0], pbc_corners[1][1] - dx, 0, 2, acc++);
    
    // sandwich boxes top 7 8 
    gmsh::model::occ::addPoint(pbc_corners[2][0], pbc_corners[2][1] + dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0], pbc_corners[3][1] + dx, 0, 2, acc++);

    // inner lattice
    for (int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    
    //Lines
    acc = 1;

    int acc_line = 1;
    
    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);
    
    // bottom box
    gmsh::model::occ::addLine(5, 6, acc++); 
    gmsh::model::occ::addLine(6, 2, acc++); 
    gmsh::model::occ::addLine(2, 1, acc++); 
    gmsh::model::occ::addLine(1, 5, acc++);

    // top box
    gmsh::model::occ::addLine(4, 3, acc++); 
    gmsh::model::occ::addLine(3, 7, acc++); 
    gmsh::model::occ::addLine(7, 8, acc++); 
    gmsh::model::occ::addLine(8, 4, acc++);

    acc_line = 9;

    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }
    
    acc = 1;
    int acc_loop = 1;
    // clipping box
    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
    gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
    acc_loop = 13;
    // std::cout << "#polygons " << polygons.size() << std::endl;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }
    
    for (int i = 0; i < polygons.size()+3; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }
    

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+4));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    std::cout << "add geometry done" << std::endl;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    std::cout << "add cut box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_bottom_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);
    std::cout << "add bottom box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_top_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy2;
    gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);
    std::cout << "add top box done" << std::endl;
    

    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.5);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();

    gmsh::model::occ::synchronize();
    gmsh::model::mesh::generate(2);

    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
}

void Tiling2D::generateSandwichMeshPerodicInX(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

 
    //Points
    int acc = 1;
    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2, acc++);
    
    // sandwich boxes bottom 5 6 the other two points already exist
    T dx = 0.05 * (pbc_corners[1][0] - pbc_corners[0][0]);
    gmsh::model::occ::addPoint(pbc_corners[0][0],  pbc_corners[0][1] - dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0], pbc_corners[1][1] - dx, 0, 2, acc++);
    
    // sandwich boxes top 7 8 
    gmsh::model::occ::addPoint(pbc_corners[2][0], pbc_corners[2][1] + dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0], pbc_corners[3][1] + dx, 0, 2, acc++);

    // inner lattice
    for (int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    
    //Lines
    acc = 1;

    int acc_line = 1;
    
    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);
    
    // bottom box
    gmsh::model::occ::addLine(5, 6, acc++); 
    gmsh::model::occ::addLine(6, 2, acc++); 
    gmsh::model::occ::addLine(2, 1, acc++); 
    gmsh::model::occ::addLine(1, 5, acc++);

    // top box
    gmsh::model::occ::addLine(4, 3, acc++); 
    gmsh::model::occ::addLine(3, 7, acc++); 
    gmsh::model::occ::addLine(7, 8, acc++); 
    gmsh::model::occ::addLine(8, 4, acc++);

    acc_line = 9;

    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }
    
    acc = 1;
    int acc_loop = 1;
    // clipping box
    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
    gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
    acc_loop = 13;
    // std::cout << "#polygons " << polygons.size() << std::endl;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }
    
    for (int i = 0; i < polygons.size()+3; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }
    

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+4));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    std::cout << "add geometry done" << std::endl;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    std::cout << "add cut box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_bottom_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);
    std::cout << "add bottom box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_top_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy2;
    gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);
    std::cout << "add top box done" << std::endl;
    

    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.5);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();

    int zero_idx;
    for(int i=0; i < pbc_corners.size(); i++)
    {
        if(pbc_corners[i].norm()<1e-6)
        {
            zero_idx = i;
            break;
        }
    }

    TV2 t1 = pbc_corners[(zero_idx+1)%pbc_corners.size()];
    TV2 t2 = pbc_corners[(zero_idx+3)%pbc_corners.size()];

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, 0, 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, 0, 0, 0, 0, 1});
    int x_pair_cnt = 0;
    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, -eps, std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, eps, sleft, 1);
    // std::ofstream pbc_output(data_folder + "pbc_data.txt");
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 1);

        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_hor);
                x_pair_cnt++;
            }
        }
    }
    gmsh::model::occ::synchronize();
    gmsh::model::mesh::generate(2);

    // std::cout << "# of periodic pairs in x " << x_pair_cnt << std::endl;
    // gmsh::write(data_folder + "thickshellPatchPeriodicInX.msh");
    // gmsh::write(data_folder + "thickshellPatchPeriodicInX.vtk");
    // std::ofstream translation(data_folder + "thickshellPatchPeriodicInXTranslation.txt");
    // translation << t1.transpose() << std::endl;
    // translation << t2.transpose() << std::endl;
    // translation.close();
    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
    
}

void Tiling2D::generatePeriodicMesh(std::vector<std::vector<TV2>>& polygons, std::vector<TV2>& pbc_corners)
{
      // Before using any functions in the C++ API, Gmsh must be initialized:
    T eps = 1e-5;
    gmsh::initialize();

    T mult = 10000000.0;

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    //Points
    int acc = 1;

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    gmsh::model::occ::synchronize();
    //Lines
    acc = 1;
    int acc_line = 1;

    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }
    gmsh::model::occ::synchronize();
    acc = 1;
    int acc_loop = 1;

    std::cout << "#polygons " << polygons.size() << std::endl;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    gmsh::model::occ::synchronize();
    //Surface
    for (int i = 0; i < polygons.size(); i++)
    {
        if (i == 2)
            continue;
        if (i == 1)
            gmsh::model::occ::addPlaneSurface({i+1, i+2}, i + 1);
        else
            gmsh::model::occ::addPlaneSurface({i+1}, i + 1);
        // gmsh::model::occ::addPlaneSurface({i+1}, i + 1);
    }

    std::cout << "add surface" << std::endl;

    int zero_idx;
    for(int i=0; i < pbc_corners.size(); i++)
    {
        if(pbc_corners[i].norm()<1e-6)
        {
            zero_idx = i;
            break;
        }
    }

    TV2 t1 = pbc_corners[(zero_idx+1)%pbc_corners.size()];
    TV2 t2 = pbc_corners[(zero_idx+3)%pbc_corners.size()];

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, 0, 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, 0, 0, 0, 0, 1});

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, -eps, std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, eps, sleft, 1);
    // std::ofstream pbc_output(data_folder + "pbc_data.txt");
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 1);

        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_hor);
                // pbc_output << "X " << j.second << " " << i.second << std::endl;
            }
        }
    }

    std::vector<std::pair<int, int>> sbottom;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t1[0])-eps, std::min(0.0,t1[1])-eps, -eps, std::max(0.0,t1[0])+eps, std::max(0.0,t1[1])+eps, eps, sbottom, 1);

    for(auto i : sbottom) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > stop;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t2[0], ymin-eps+t2[1], zmin - eps, xmax+eps+t2[0], ymax+eps+t2[1], zmax + eps, stop, 1);

        for(auto j : stop) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t2[0];
            ymin2 -= t2[1];
            xmax2 -= t2[0];
            ymax2 -= t2[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_ver);
                // pbc_output << "Y " << j.second << " " << i.second << std::endl;
            }
        }
    }

    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    // std::vector<std::size_t> pbc_nodes_a, pbc_nodes_b;
    // int tagMaster;
    // std::vector<T> affineTransform;
    // gmsh::model::mesh::getPeriodicNodes(1, 1, tagMaster, pbc_nodes_a, pbc_nodes_b, affineTransform);
    // for (int i = 0; i < pbc_nodes_a.size(); i++)
    // {
    //     pbc_output << pbc_nodes_a[i] << " " << pbc_nodes_b[i] << std::endl;
    // }
    // pbc_output.close();
    gmsh::write(data_folder + "thickshell.msh");
    gmsh::write(data_folder + "thickshell.vtk");
    std::ofstream translation(data_folder + "translation.txt");
    translation << t1.transpose() << std::endl;
    translation << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
}

void Tiling2D::getPBCUnit(VectorXT& vertices, EdgeList& edge_list)
{
    std::vector<TV2> data_points;
    TV2 T1, T2;
    //10 is good
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners;
    fetchUnitCellFromOneFamily(0, polygons, pbc_corners, 2, false);
    generatePeriodicMesh(polygons, pbc_corners);
}