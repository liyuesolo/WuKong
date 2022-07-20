#include "../include/Tiling2D.h"

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
            polygons[i] << ClipperLib::IntPoint((polygons_v[i][j][0]-xy[0])*mult, (polygons_v[i][j][1]-xy[1])*mult);
            // std::cout << " " << polygons[i][j];
        }
        // std::cout << std::endl;
    }
    
    // std::ofstream out("tiling_unit.obj");
    // for (int i = 0; i < 4; i++)
    //     out << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    // for (auto polygon : polygons)
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
    // out.close();
    
    T distance = -2.0;
    ClipperLib::Paths clip(1);
	clip[0] << ClipperLib::IntPoint(periodic[0]*mult,periodic[1]*mult) << ClipperLib::IntPoint(periodic[2]*mult, periodic[3]*mult) << ClipperLib::IntPoint(periodic[4]*mult, periodic[5]*mult) << ClipperLib::IntPoint(periodic[6]*mult, periodic[7]*mult);
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
        // if (i == 2)
        //     continue;
        // if (i == 1)
        //     gmsh::model::occ::addPlaneSurface({i+1, i+2}, i + 1);
        // else
        //     gmsh::model::occ::addPlaneSurface({i+1}, i + 1);
        gmsh::model::occ::addPlaneSurface({i+1}, i + 1);
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
            }
        }
    }

    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);
    
    gmsh::write("thickshell.msh");
    gmsh::write("thickshell.vtk");

    gmsh::finalize();
}

void Tiling2D::getPBCUnit(VectorXT& vertices, EdgeList& edge_list)
{
    std::vector<TV2> data_points;
    TV2 T1, T2;
    //10 is good
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners;
    fetchUnitCellFromOneFamily(0, polygons, pbc_corners, 1, false);
    generatePeriodicMesh(polygons, pbc_corners);
}