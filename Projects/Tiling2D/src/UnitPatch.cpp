#include "../include/Tiling2D.h"
#include <time.h>
std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );

static T zeta()
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
    T depth = (pbc_corners[1] - pbc_corners[0]).norm() * 1.0;
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
    std::vector<TV2>& eigen_base, int n_unit)
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
    size_t num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    T pi;
    for (int i = 0; i < num_params; i++)
        in >> new_params[i];
    a_tiling.setParameters( new_params );
    in >> token;
    int num_edge_shape;
    in >> num_edge_shape;
    vector<vector<dvec2>> edges(a_tiling.numEdgeShapes());
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
    T angle;
    in >> angle;
    in.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 4.0, 4.0, xy);
    
    // Vector<T, 8> periodic;
    
    // periodic.head<2>() = TV2(0,0);
    // periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    // periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    // periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    
    // std::cout << periodic.transpose() << std::endl;
    // TV t1 = periodic.segment<2>(2);
    // TV t2 = periodic.segment<2>(6);
    // TV t1_unit = t1.normalized();
    // TV x_axis(1, 0);

    // T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    // if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
    //     theta_to_x *= -1.0;
    // TM2 R;
    // R << std::cos(theta_to_x), -std::sin(theta_to_x), std::sin(theta_to_x), std::cos(theta_to_x);
    TM R = rotMat(angle);
    // std::cout << R << std::endl;

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            // TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            // polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            TV curr = TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    // periodic.segment<2>(2) = R * periodic.segment<2>(2);
    // periodic.segment<2>(6) = TV(0, 1) * periodic.segment<2>(2).norm();
    // periodic[4] = periodic[2]; periodic[5] = periodic[7];

    // T dx = std::abs(periodic[2]);
    // TV shift = TV(0.1 * dx, 0.0);
    // for (int i = 0; i < 4; i++)
    //     periodic.segment<2>(i * 2) += shift;

    T distance = -1.5;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);
    shapeToPolygon(final_shape, eigen_polygons, mult);

    T min_x = 1e10, min_y = 1e10, max_x = -1e10, max_y = -1e10;
    for (auto polygon : eigen_polygons)
        for (auto pt : polygon)
        {
            min_x = std::min(min_x, pt[0]); min_y = std::min(min_y, pt[1]);
            max_x = std::max(max_x, pt[0]); max_y = std::max(max_y, pt[1]);
        }
    T dx = max_x - min_x;
    T dy = max_y - min_y;
    T scale = 0.35;
    Vector<T, 8> periodic;
    periodic << min_x + scale * dx, min_y + scale * dy, max_x - scale * dx, 
                min_y + scale * dy, max_x - scale * dx, max_y - scale * dy, 
                min_x + scale * dx, max_y - scale * dy;
    periodic.segment<2>(0) = R * periodic.segment<2>(0);
    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    periodic.segment<2>(4) = R * periodic.segment<2>(4);
    periodic.segment<2>(6) = R * periodic.segment<2>(6);
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", false);
    // std::cout << "CLIPPER done" << std::endl;
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    // std::exit(0);
    periodicToBase(periodic, eigen_base);
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
    T new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    vector<vector<dvec2>> edges(a_tiling.numEdgeShapes());
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
    getTilingShape(shape, a_tiling, edges);


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

            //eigen_polygons[i].push_back(TV(final[i][j].X/mult, final[i][j].Y/mult));
        }
    }

    eigen_base.resize(0);
    T box_length = 0.1;
    eigen_base.push_back(TV2(-box_length * dx, -box_length * dy));
    eigen_base.push_back(TV2(box_length * dx, -box_length * dy));
    eigen_base.push_back(TV2(box_length * dx, box_length * dy));
    eigen_base.push_back(TV2(-box_length * dx, box_length * dy));

}



void Tiling2D::sampleOneFamilyWithOrientation(int IH, T angle, int n_unit, 
        T height, std::vector<std::vector<TV2>>& eigen_polygons,
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
    T new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";

    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    getTilingEdges(a_tiling, eij, edges);
    for (auto ej : edges)
    {
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);

    Vector<T, 8> periodic; periodic.setZero();
    periodic.segment<2>(2) = TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = TV2(transf[0],transf[2]) + TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = TV2(transf[1],transf[3]);

    
    TV t1 = periodic.segment<2>(2);
    TV t2 = periodic.segment<2>(6);

    TV t1_unit = t1.normalized();
    TV x_axis(1, 0);

    T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
        theta_to_x *= -1.0;
    TM R = rotMat(theta_to_x);

    
    TM R2 = rotMat(angle);

    periodic.segment<2>(2) = R2 * R * periodic.segment<2>(2);
    periodic.segment<2>(4) = R2 * R * periodic.segment<2>(4);
    periodic.segment<2>(6) = R2 * R * periodic.segment<2>(6);
    
    t1 = periodic.segment<2>(2);
    t2 = periodic.segment<2>(6);

    int n = 1;
    for (; n < 100; n++)
    {
        TV a = t1 * T(n);
        TV b = a.dot(TV(1, 0)) * TV(1, 0);
        TV c = a - b;
        T beta = std::acos(t1.normalized().dot(t2.normalized()));
        T alpha = angle;
        T target = c.norm() / std::cos(beta - (M_PI/2.0 - alpha));
        T div_check = target / t2.norm();
        T delta = std::abs(std::round(div_check) - div_check);
        std::cout << "n " << n << " delta " << delta << " round " << std::round(div_check) << " true " <<  div_check << std::endl;
        if (delta < 1e-6)
        {
            break;
        }
    }
    std::cout << n << std::endl;
    // std::getchar();
    periodic.segment<2>(2) = t1.dot(TV(1, 0)) * TV(1, 0) * n * n_unit;
    periodic[6] = 0; periodic[7] = height;
    periodic[4] = periodic[2]; periodic[5] = periodic[7]; 

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i < polygons_v.size(); ++i)
    {
        for(int j=0; j < polygons_v[i].size(); ++j)
        {
            TV curr = R2 * R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            // TV curr = TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
      

    T distance = -2.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
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
    T new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    // vector<dvec2> edges[ a_tiling.numEdgeShapes() ];
    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    getTilingEdges(a_tiling, eij, edges);
    for (auto ej : edges)
    {
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);

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
            // TV curr = TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    
    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    periodic.segment<2>(6) = TV(0, 1) * periodic.segment<2>(2).norm();
    periodic[4] = periodic[2]; periodic[5] = periodic[7];   


    // T distance = -2.0;//default
    T distance = -1.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
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
    
    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    
    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    getTilingEdges(a_tiling, eij, edges);
    for (auto ej : edges)
    {
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;

    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);

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

    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
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
        T new_params[ num_params ];
        
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
    vector<vector<dvec2>> edges(a_tiling.numEdgeShapes());

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
    getTilingShape(shape, a_tiling, edges);


    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);

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


    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", false);
    
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
}

void Tiling2D::generateOneStructureSquarePatch(int IH, const std::vector<T>& params)
{
    data_folder = "./";
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
    }
    a_tiling.setParameters( new_params );

    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    Vector<T, 4> eij;
    eij << 0.25, 0., 0.75, 0.;
    getTilingEdges(a_tiling, eij, edges);
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    
    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 16.0, 16.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 6.0, 6.0, xy);
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 8.0, 8.0, xy);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            polygons[i] << ClipperLib::IntPoint((polygons_v[i][j][0]-xy[0])*mult, 
                (polygons_v[i][j][1]-xy[1])*mult);
        }
    }
    
    T distance = -1.5;
    ClipperLib::Paths final_shape;

    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    std::vector<std::vector<TV2>> eigen_polygons;
    std::vector<TV2> eigen_base;
    shapeToPolygon(final_shape, eigen_polygons, mult);
    T min_x = 1e10, min_y = 1e10, max_x = -1e10, max_y = -1e10;
    for (auto polygon : eigen_polygons)
        for (auto pt : polygon)
        {
            min_x = std::min(min_x, pt[0]); min_y = std::min(min_y, pt[1]);
            max_x = std::max(max_x, pt[0]); max_y = std::max(max_y, pt[1]);
        }
    T dx = max_x - min_x;
    T dy = max_y - min_y;
    T scale_x = 0.08, scale_y = 0.4;
    // T scale_x = 0.38, scale_y = 0.38;
    Vector<T, 8> periodic;
    periodic << min_x + scale_x * dx, min_y + scale_y * dy, max_x - scale_x * dx, 
                min_y + scale_y * dy, max_x - scale_x * dx, max_y - scale_y * dy, 
                min_x + scale_x * dx, max_y - scale_y * dy;
    
    // periodic << min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y;
    periodicToBase(periodic, eigen_base);
    generateNonPeriodicMesh(eigen_polygons, eigen_base, true, data_folder + "a_structure");
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_None);
}

void Tiling2D::generateToyExampleStructure(const std::vector<T>& params,
    const std::string& result_folder)
{
    std::vector<std::vector<TV2>> eigen_polygons;
    std::vector<TV2> eigen_base;
    std::string filename = result_folder + "structure.txt";
    // std::ofstream out(filename);
    TV center = TV(1, 1);
    
    T scale = 1.0;
    std::vector<TV> corners = {center + TV(-1, -1) * scale, 
        center +TV(1, -1) * scale, 
        center + TV(1, 1) * scale, 
        center + TV(-1, 1) * scale};

    T d_domain = (corners[2] - corners[0]).norm();

    std::vector<TV> inner_square(4);
    for (int i = 0; i < 4; i++)
        inner_square[i] = center + std::abs(params[0]) * (corners[i] - center);
    
    
    std::vector<std::vector<TV>> polygons_v(5);
    polygons_v[0] = inner_square;
    

    polygons_v[1] = {inner_square[1] - TV(2, 0) * scale,
                    corners[0],
                    inner_square[0], inner_square[3], 
                    corners[3],
                    inner_square[2] - TV(2, 0) * scale};

    polygons_v[2] = {corners[3],
                    inner_square[3], inner_square[2], 
                    corners[2],
                    inner_square[1] + TV(0, 2) * scale,
                    inner_square[0] + TV(0, 2) * scale
                    };

    polygons_v[3] = {
        corners[2], inner_square[2], inner_square[1], corners[1], 
        inner_square[0] + TV(2, 0) * scale,
        inner_square[3] + TV(2, 0) * scale
    };

    polygons_v[4] = {
        corners[0], 
        inner_square[3] - TV(0, 2) * scale,
        inner_square[2] - TV(0, 2) * scale,
        corners[1], inner_square[1], inner_square[0]
    };

    Vector<T, 8> periodic;
    for (int i = 0; i < 4; i++)
        periodic.segment<2>(i * 2) = corners[i];


    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = TV(polygons_v[i][j][0], polygons_v[i][j][1]);
    
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            // std::cout << polygons[i] << std::endl;
        }
        TV curr = TV(polygons_v[i][0][0], polygons_v[i][0][1]);
    
        polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
    }
    
    
    T distance = -0.015 * d_domain;
    ClipperLib::Paths final_shape;
    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
    // saveClip(final_shape, periodic, mult, "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/Tiling2D/tiling_unit_clip_in_x.obj", true);
    generatePeriodicMeshHardCodeResolution(eigen_polygons, eigen_base, true, result_folder + "structure");
    
}

void Tiling2D::generateToyExample(T param)
{
    std::vector<std::vector<TV2>> eigen_polygons;
    std::vector<TV2> eigen_base;
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/";
    std::string filename = data_folder + "a_structure.txt";
    // std::ofstream out(filename);
    TV center = TV(1, 1);
    
    T scale = 1.0;
    std::vector<TV> corners = {center + TV(-1, -1) * scale, 
        center +TV(1, -1) * scale, 
        center + TV(1, 1) * scale, 
        center + TV(-1, 1) * scale};

    T d_domain = (corners[2] - corners[0]).norm();

    std::vector<TV> inner_square(4);
    for (int i = 0; i < 4; i++)
        inner_square[i] = center + std::abs(param) * (corners[i] - center);
    
    
    std::vector<std::vector<TV>> polygons_v(5);
    polygons_v[0] = inner_square;
    

    polygons_v[1] = {inner_square[1] - TV(2, 0) * scale,
                    corners[0],
                    inner_square[0], inner_square[3], 
                    corners[3],
                    inner_square[2] - TV(2, 0) * scale};

    polygons_v[2] = {corners[3],
                    inner_square[3], inner_square[2], 
                    corners[2],
                    inner_square[1] + TV(0, 2) * scale,
                    inner_square[0] + TV(0, 2) * scale
                    };

    polygons_v[3] = {
        corners[2], inner_square[2], inner_square[1], corners[1], 
        inner_square[0] + TV(2, 0) * scale,
        inner_square[3] + TV(2, 0) * scale
    };

    polygons_v[4] = {
        corners[0], 
        inner_square[3] - TV(0, 2) * scale,
        inner_square[2] - TV(0, 2) * scale,
        corners[1], inner_square[1], inner_square[0]
    };

    Vector<T, 8> periodic;
    for (int i = 0; i < 4; i++)
        periodic.segment<2>(i * 2) = corners[i];


    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = TV(polygons_v[i][j][0], polygons_v[i][j][1]);
    
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            // std::cout << polygons[i] << std::endl;
        }
        TV curr = TV(polygons_v[i][0][0], polygons_v[i][0][1]);
    
        polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
    }
    
    
    T distance = -0.015 * d_domain;
    ClipperLib::Paths final_shape;
    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);

    generatePeriodicMeshHardCodeResolution(eigen_polygons, eigen_base, true, data_folder + "a_structure");
    solver.pbc_translation_file = data_folder + "a_structure_translation.txt";
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_XY);
}

void Tiling2D::generateOnePerodicUnit()
{
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/";
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    // int tiling_idx = 19;
    // int tiling_idx = 46;
    // int tiling_idx = 0;
    int tiling_idx = 26;
    // int tiling_idx = 20;
    // int tiling_idx = 47;
    // int tiling_idx = 27;

    T unit = 5.0;
    if (tiling_idx == 19 || tiling_idx == 46 || tiling_idx == 60 || tiling_idx == 20)
        unit = 5.0;
    else if (tiling_idx == 26 || tiling_idx == 0 || tiling_idx == 27)
        unit = 10.0;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ tiling_idx ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    
    // params = {0.1841};
    for (int k = 0; k < num_params - 1; k++)
        std::cout << params[k] << ", ";
    std::cout << params[num_params - 1] << std::endl;
    // std::cout << params[0] << " " << params[1] << std::endl;
    params = {0.02878188105, 0.5263784471};
    // params[0] += 1e-4;
    
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0., 0.75, 0.;
    fetchUnitCellFromOneFamily(tiling_idx, 2, polygons, pbc_corners, params, 
        cubic_weights, data_folder + "a_structure.txt", unit);
    
    generatePeriodicMesh(polygons, pbc_corners, true, data_folder + "a_structure");
    // generateHomogenousMesh(polygons, pbc_corners, true, data_folder + "a_structure");
    // generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
    //     data_folder + "a_structure.vtk");   
    solver.pbc_translation_file = data_folder + "a_structure_translation.txt";
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_XY);
    if (tiling_idx == 0)
        solver.pbc_strain_w = 1e7;
}

void Tiling2D::generateOneStructureWithRotation()
{
    int IH = 0;
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    T angle = M_PI / 8.0;
    cubic_weights << 0.25, 0, 0.75, 0;
    sampleOneFamilyWithOrientation(IH, angle, 2, 50, polygons, pbc_corners, params,
        cubic_weights, data_folder + "a_structure.txt");
    generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
        data_folder + "a_structure.vtk");   
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_X);
}

void Tiling2D::generateOneStructure()
{
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/";
    int IH = 46;
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    // cubic_weights << 0.25, 0.25, 0.75, 0.75;
    T x0_rand = zeta(), y0_rand = zeta();
    cubic_weights << x0_rand, y0_rand, x0_rand + (1.0 - x0_rand) * zeta(), y0_rand + (1.0 - y0_rand) * zeta();
    // cubic_weights << 0.46529, 0.134614, 0.787798, 0.638062;
    cubic_weights << 0.25, 0, 0.75, 0;
    sampleSandwichFromOneFamilyFromDiffParamsDilation(IH, polygons, pbc_corners, params,
        cubic_weights, data_folder + "a_structure.txt");
    generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
        data_folder + "a_structure.vtk", 0, 1);   
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_X);
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
    T new_params[ num_params ];
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
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_None);
}

void Tiling2D::generateSandwichBatchChangingTilingParams()
{
    std::srand( (unsigned)time( NULL ) );
    data_folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKTri6/";
    int cnt = 0;
    // tbb::parallel_for(0, 81, [&](int i)
    for (int i = 0; i < 10; i++)
    {
        csk::IsohedralTiling a_tiling( csk::tiling_types[ i ] );
        int num_params = a_tiling.numParameters();
        T new_params[ num_params ];
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
            data_folder + std::to_string(i * offset) + ".vtk", 4, 2);   
        return;
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


void Tiling2D::fetchUnitCellFromOneFamily(int IH, int n_unit,
    std::vector<std::vector<TV2>>& eigen_polygons,
    std::vector<TV2>& eigen_base, 
    const std::vector<T>& params,
    const Vector<T, 4>& eij, const std::string& filename,
    T unit, T angle)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    std::ofstream out(filename);
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    out << "IH " << IH << std::endl;
    size_t num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    getTilingEdges(a_tiling, eij, edges);
    for (auto ej : edges)
    {
        for (dvec2 e : ej)
            out << e[0] << " " << e[1] << " ";
        out << std::endl;
    }
    out << angle << std::endl;
    out.close();

    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;

    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, unit, unit, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 8.0, 8.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 4.0, 4.0, xy);

    

    Vector<T, 8> periodic;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    // TM R = rotMat(angle);
    TM R = TM::Identity();

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            // polygons[i] << ClipperLib::IntPoint((polygons_v[i][j][0]-xy[0])*mult, 
            //     (polygons_v[i][j][1]-xy[1])*mult);
            // std::cout << " " << polygons[i][j];
        }
        // break;
        // std::cout << std::endl;
    }
    // std::ofstream polygon_obj("polygon_obj.obj");
    // // for (auto polygon : polygons)
    // for (int i = 0; i < polygons.size(); i++)
    // {
    //     auto polygon = polygons[i];
        
    //     for (auto vtx : polygon)
    //         polygon_obj << "v " << vtx.X << " " << vtx.Y << " 0" << std::endl;
        
    // }
    // polygon_obj.close();

    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    periodic.segment<2>(4) = R * periodic.segment<2>(4);
    periodic.segment<2>(6) = R * periodic.segment<2>(6);
    
    T distance = -1.5;
    ClipperLib::Paths final_shape;

    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
}

void Tiling2D::extrudeToMesh(const std::string& tiling_param, const std::string& mesh3d, int n_unit)
{
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    loadTilingStructureFromTxt(tiling_param, polygons, pbc_corners, n_unit);
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
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename,
    int resolution, int element_order)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", element_order);

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
    if (resolution == 0)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);    
    }
    else if (resolution == 1)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    }
    else if (resolution == 2)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.5);
    }
    else if (resolution == 3)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.2);
    }
    else if (resolution == 4)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    }

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

    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
    
}

void Tiling2D::generateHomogenousMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
    T eps = 1e-5;
    gmsh::initialize();

    T mult = 10000000.0;

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 2);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
    
    //Points
    int acc = 1;
    pbc_corners = {TV(0, 0), TV(10,0), TV(10,10), TV(0,10)};

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;

    
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
   

    gmsh::model::occ::addPlaneSurface({1});

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
    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.05);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.2);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.3);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.8);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    
    gmsh::write(prefix + ".vtk");
    std::ofstream translation(prefix + "_translation.txt");
    translation << t1.transpose() << std::endl;
    translation << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
}
void Tiling2D::generateNonPeriodicMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    // gmsh::logger::start();
    // gmsh::logger::stop();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 2);

    // gmsh::option::setNumber("General.Verbosity", 0);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T th = eps;
    //Points
    int acc = 1;

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 1, acc++);
        }
    }
    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;
    std::vector<T> poly_lines;
    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(starting_vtx++, starting_vtx, acc++);
            poly_lines.push_back(acc);
        }
        gmsh::model::occ::addLine(starting_vtx, starting_vtx-polygons[i].size()+1, acc++);
        poly_lines.push_back(acc);
        ++starting_vtx;
    }

    gmsh::model::mesh::field::add("Distance", 1);
    // gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 2.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    // gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    acc_loop = 5;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    for (int i = 0; i < polygons.size()+1; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+2));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    
    gmsh::write(prefix + ".vtk");
    gmsh::finalize();
}

void Tiling2D::generatePeriodicMesh(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
      // Before using any functions in the C++ API, Gmsh must be initialized:

    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    // gmsh::logger::start();
    // gmsh::logger::stop();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 2);

    gmsh::option::setNumber("General.Verbosity", 0);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T th = eps;
    //Points
    int acc = 1;

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 1, acc++);
        }
    }
    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;
    std::vector<T> poly_lines;
    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(starting_vtx++, starting_vtx, acc++);
            poly_lines.push_back(acc);
        }
        gmsh::model::occ::addLine(starting_vtx, starting_vtx-polygons[i].size()+1, acc++);
        poly_lines.push_back(acc);
        ++starting_vtx;
    }

    gmsh::model::mesh::field::add("Distance", 1);
    // gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.0);
    gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    // gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    acc_loop = 5;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    for (int i = 0; i < polygons.size()+1; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+2));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
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

    
    gmsh::write(prefix + ".vtk");
    std::ofstream translation(prefix + "_translation.txt");
    translation << std::setprecision(20) << t1.transpose() << std::endl;
    translation << std::setprecision(20) << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
    
}

void Tiling2D::generatePeriodicMeshHardCodeResolution(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
      // Before using any functions in the C++ API, Gmsh must be initialized:

    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    // gmsh::logger::start();
    // gmsh::logger::stop();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 2);

    gmsh::option::setNumber("General.Verbosity", 0);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T th = eps;
    //Points
    int acc = 1;

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 1, acc++);
        }
    }
    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;
    std::vector<T> poly_lines;
    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(starting_vtx++, starting_vtx, acc++);
            poly_lines.push_back(acc);
        }
        gmsh::model::occ::addLine(starting_vtx, starting_vtx-polygons[i].size()+1, acc++);
        poly_lines.push_back(acc);
        ++starting_vtx;
    }

    gmsh::model::mesh::field::add("Distance", 1);
    // gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.02);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.02);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.008);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.008);
    // gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    acc_loop = 5;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    for (int i = 0; i < polygons.size()+1; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+2));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
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

    
    gmsh::write(prefix + ".vtk");
    std::ofstream translation(prefix + "_translation.txt");
    translation << std::setprecision(20) << t1.transpose() << std::endl;
    translation << std::setprecision(20) << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
    
}