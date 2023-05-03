#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <igl/readOBJ.h>
#include "../include/Tiling3D.h"


#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "../include/Util.h"

void Tiling3D::generateGreenStrainSecondPKPairsServerToyExample(const std::vector<T>& params,
        const std::string& result_folder, int loading_type, bool generate_mesh)
{
    std::ofstream out;
    if (loading_type == 0)
        out.open(result_folder + "data_uniaxial.txt");
    else if (loading_type == 1)
        out.open(result_folder + "data_biaxial.txt");
    else if (loading_type == 2)
        out.open(result_folder + "data_triaxial.txt");
    out << std::setprecision(12);
    TV2 range_strain(0.7, 1.5);
    TV2 range_strain_biaixial(0.9, 1.2);
    TV2 range_strain_triaixial(0.9, 1.2);
	TV2 range_theta(0.0, 2.0 * M_PI);
    TV2 range_phi(-M_PI_2, M_PI_2);
        
    int n_sp_strain = 50;
    int n_sp_strain_bi = 10;
    int n_sp_strain_tri = 5;
    int n_sp_theta = 15;
    int n_sp_phi = 15;

    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
    T delta_strain_bi = (range_strain_biaixial[1] - range_strain_biaixial[0]) / T(n_sp_strain_bi);
    T delta_strain_tri = (range_strain_triaixial[1] - range_strain_triaixial[0]) / T(n_sp_strain_tri);

    auto runSim = [&](int& sim_cnt, T theta, T phi, TV strain_magnitude)
    {
        sim_cnt++;
            
        // bool solve_succeed = solver.staticSolve();
        bool solve_succeed = true;

        VectorXT residual(solver.num_nodes * 3); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(Green_strain, secondPK_stress, psi);
        for (int m = 0; m < params.size(); m++)
        {
            out << params[m] << " ";
        }
        for (int i = 0; i < 3; i++)
            out << Green_strain(i, i) << " ";
        out << Green_strain(0, 1) << " " << Green_strain(0, 2) << " " << Green_strain(1, 2) << " ";
        for (int i = 0; i < 3; i++)
            out << secondPK_stress(i, i) << " ";
        out << secondPK_stress(0, 1) << " " << secondPK_stress(0, 2) << " " << secondPK_stress(1, 2) << " ";
        out << psi << " " << theta << " " << phi << " " << strain_magnitude[0] 
            << " " << strain_magnitude[1] << " " << strain_magnitude[2] << " "
            << residual.norm() << std::endl;
        if (!solve_succeed)
        {
            solver.reset();
        }
    };

    if (generate_mesh)
        solver.generate3DUnitCell(result_folder + "/structure", params[0], params[1]);
    return;

    bool valid_structure = solver.initializeSimulationDataFromFiles(result_folder + "structure.vtk");
    if (!valid_structure)
        return;
    solver.verbose = false;
    int cnt = 0;
    
    if (loading_type == 0)
    {
        solver.loading_type = UNI_AXIAL;
        
        for(int i = 0; i < n_sp_theta; i++)
        {
            T theta = range_theta[0] + ((double)i/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
            for(int j = 0; j < n_sp_phi; j++)
            {
                T phi = range_phi[0] + ((double)j/(double)n_sp_phi)*(range_phi[1] - range_phi[0]);
                solver.theta = theta;
                solver.phi = phi;
                for (T strain = 1.001; strain < range_strain[1]; strain += delta_strain)
                {
                    TV strain_mag(strain, 0, 0);
                    solver.strain_magnitudes = strain_mag;
                    runSim(cnt, theta, phi, strain_mag);
                }
                solver.reset();
                for (T strain = 0.999; strain > range_strain[0]; strain -= delta_strain)
                {    
                    TV strain_mag(strain, 0, 0);
                    solver.strain_magnitudes = strain_mag;
                    runSim(cnt, theta, phi, strain_mag);
                }
            }
        }
    }
    else if (loading_type == 1)
    {
        solver.loading_type = BI_AXIAL;
        
        for(int i = 0; i < n_sp_theta; i++)
        {
            T theta = range_theta[0] + ((double)i/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
            for(int j = 0; j < n_sp_phi; j++)
            {
                T phi = range_phi[0] + ((double)j/(double)n_sp_phi)*(range_phi[1] - range_phi[0]);
                solver.theta = theta;
                solver.phi = phi;
                for (T strain = 1.001; strain < range_strain_biaixial[1]; strain += delta_strain_bi)
                {
                    solver.reset();
                    for (T strain_ortho = 1.001; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
                    {
                        TV strain_mag(strain, strain_ortho, 0);
                        solver.strain_magnitudes = strain_mag;
                        runSim(cnt, theta, phi, strain_mag);
                    }
                    solver.reset();
                    for (T strain_ortho = 0.999; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
                    {
                        TV strain_mag(strain, strain_ortho, 0);
                        solver.strain_magnitudes = strain_mag;
                        runSim(cnt, theta, phi, strain_mag);
                    }
                }
                
                for (T strain = 0.999; strain > range_strain_biaixial[0]; strain -= delta_strain_bi)
                {    
                    solver.reset(); 
                    for (T strain_ortho = 1.001; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
                    {
                        TV strain_mag(strain, strain_ortho, 0);
                        solver.strain_magnitudes = strain_mag;
                        runSim(cnt, theta, phi, strain_mag);
                    }
                    solver.reset();
                    for (T strain_ortho = 0.999; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
                    {
                        TV strain_mag(strain, strain_ortho, 0);
                        solver.strain_magnitudes = strain_mag;
                        runSim(cnt, theta, phi, strain_mag);
                    }
                }
            }
        }
    }
    else if (loading_type == 2)
    {
        solver.loading_type = TRI_AXIAL;
        for(int i = 0; i < n_sp_theta; i++)
        {
            T theta = range_theta[0] + ((double)i/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
            for(int j = 0; j < n_sp_phi; j++)
            {
                T phi = range_phi[0] + ((double)j/(double)n_sp_phi)*(range_phi[1] - range_phi[0]);
                solver.theta = theta;
                solver.phi = phi;
                for (T strain = 1.001; strain < range_strain_triaixial[1]; strain += delta_strain_tri)
                {
                    for (T strain_ortho = 1.001; strain_ortho < range_strain_triaixial[1]; strain_ortho += delta_strain_tri)
                    {
                        solver.reset();
                        for (T strain_third = 1.001; strain_third < range_strain_triaixial[1]; strain_third += delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                        solver.reset();
                        for (T strain_third = 0.999; strain_third > range_strain_triaixial[0]; strain_third -= delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                    }
                    for (T strain_ortho = 0.999; strain_ortho > range_strain_triaixial[0]; strain_ortho -= delta_strain_tri)
                    {
                        solver.reset();
                        for (T strain_third = 1.001; strain_third < range_strain_triaixial[1]; strain_third += delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                        solver.reset();
                        for (T strain_third = 0.999; strain_third > range_strain_triaixial[0]; strain_third -= delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                    }
                }
                
                for (T strain = 0.999; strain > range_strain_triaixial[0]; strain -= delta_strain_tri)
                {    
                    for (T strain_ortho = 1.001; strain_ortho < range_strain_triaixial[1]; strain_ortho += delta_strain_tri)
                    {
                        solver.reset();
                        for (T strain_third = 1.001; strain_third < range_strain_triaixial[1]; strain_third += delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                        solver.reset();
                        for (T strain_third = 0.999; strain_third > range_strain_triaixial[0]; strain_third -= delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                    }
                    for (T strain_ortho = 0.999; strain_ortho > range_strain_triaixial[0]; strain_ortho -= delta_strain_tri)
                    {
                        solver.reset();
                        for (T strain_third = 1.001; strain_third < range_strain_triaixial[1]; strain_third += delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                        solver.reset();
                        for (T strain_third = 0.999; strain_third > range_strain_triaixial[0]; strain_third -= delta_strain_tri)
                        {
                            TV strain_mag(strain, strain_ortho, strain_third);
                            solver.strain_magnitudes = strain_mag;
                            runSim(cnt, theta, phi, strain_mag);
                        }
                    }
                }
            }
        }
    }
    
    out.close();
}


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

    auto centrePSRect = [&]( double xmin, double ymin, double xmax, double ymax )
    {
        double sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
        return dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
            * dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
            * dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
    };

    auto outShapeVec = [&]( const std::vector<dvec2>& vec, const dmat3& M )
    {
        std::vector<TV2> data_points;

        dvec2 p = M * dvec3( vec.back(), 1.0 );
        data_points.push_back(TV2(p[0], p[1]));

        for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
            dvec2 p1 = M * dvec3( vec[idx], 1.0 );
            dvec2 p2 = M * dvec3( vec[idx+1], 1.0 );
            dvec2 p3 = M * dvec3( vec[idx+2], 1.0 );

            data_points.push_back(TV2(p1[0], p1[1]));
            data_points.push_back(TV2(p2[0], p2[1]));
            data_points.push_back(TV2(p3[0], p3[1]));
        }

        return data_points;
    };
    
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

    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;

    dmat3 M = centrePSRect( -width, -height, width, height );

    // for( auto i : a_tiling.fillRegion(-5, -4, 2, 4) ) {
    // for( auto i : a_tiling.fillRegion(0, -1.5, 7, 5) ) {
    // for( auto i : a_tiling.fillRegion(0, -1.5, 5, 3) ) {
    for( auto i : a_tiling.fillRegion(0, 0, width, height) ) {

        // std::vector<TV2> data_points;
        // dmat3 TT = i->getTransform();
        
        // dvec2 p = TT * dvec3( shape.back(), 1.0 );

        // data_points.push_back(TV2(p[0], p[1]));
        // // std::cout << p[0] << " " << p[1] << std::endl;

        // for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
        //     dvec2 p1 = TT * dvec3( shape[idx], 1.0 );
        //     dvec2 p2 = TT * dvec3( shape[idx+1], 1.0 );
        //     dvec2 p3 = TT * dvec3( shape[idx+2], 1.0 );

        //     data_points.push_back(TV2(p1[0], p1[1]));
        //     data_points.push_back(TV2(p2[0], p2[1]));
        //     data_points.push_back(TV2(p3[0], p3[1]));
        // }

        // if (TT[0][0] != TT[1][1])
        //     std::reverse(data_points.begin(), data_points.end());

        // raw_points.push_back(data_points);   


        dmat3 T = M * i->getTransform();

        std::vector<TV2> data_points = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(data_points.begin(), data_points.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);
        raw_points.push_back(data_points);

    }
}



void Tiling3D::fetchOneFamily(int IH, T* params, TV2& T1, TV2& T2, 
    PointLoops& raw_points, T width, T height)
{
    bool random = true;
    using namespace csk;
    using namespace std;
    using namespace glm;

    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    
    // size_t num_params = a_tiling.numParameters();
    // if( num_params > 1 ) {
    //         double params[ num_params ];
    //     // Get the parameters out of the tiling
    //     a_tiling.getParameters( params );
    //     // Change a parameter
    //     for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) {
    //         if (random)
    //             params[idx] += zeta()*0.2 - 0.1;
    //     }
    //     // Send the parameters back to the tiling
    //     a_tiling.setParameters( params );
    // }

    a_tiling.setParameters( params );

    T1 = TV2(a_tiling.getT1().x, a_tiling.getT1().y);
    T2 = TV2(a_tiling.getT2().x, a_tiling.getT2().y);

    vector<dvec2> edges[ a_tiling.numEdgeShapes() ];

    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        // Start by making a random Bezier segment.
        ej.push_back( dvec2( 0, 0 ) );
        // if (random)
        // {
        //     ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
        //     ej.push_back( 
        //         dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
        // }
        // else
        // {
        //     ej.push_back( dvec2( 0.25, 0 ) );
        //     ej.push_back( dvec2( 0.75, 0) );   
        // }
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



void Tiling3D::test()
{
    std::vector<PointLoops> raw_points;
    // this is in cm
    int IH = 5;
    T params[] = {0.1224, 0.4979, 0.0252, 0.4131, 0.4979}; //Isohedral 5
    // fetchOneFamily(IH, params, raw_points, 10, 10);
    // extrudeToMesh(raw_points, 0.1, 1.0, "IH" + std::to_string(IH) + "_10cmx10cmx10mm.obj");
}

void Tiling3D::fetchTilingVtxLoop(std::vector<PointLoops>& raw_points)
{
    int IH = 0;
    T params[] = {0.1161, 0.5464, 0.4313, 0.5464}; //Isohedral 0
    
    // int IH = 13;
    // T params[] = {0.1, 0.2}; //Isohedral 7

    // int IH = 29;
    // T params[] = {0}; //Isohedral 29

    // int IH = 6;
    // T params[] = {0.5, 0.5, 0.5, 0.5, 0.5}; //Isohedral 06

    fetchOneFamilyFillRegion(IH, params, raw_points, 70, 70);

    // PointLoops point_loop_unit;
    // T square_width =  0.4;
    // std::vector<Edge> edges;
    // std::vector<TV2> valid_points;
    // clapBottomLayerWithSquare(IH, params, point_loop_unit, valid_points, edges, square_width);
    // std::ofstream out("cropped_tiling.obj");
    // for (auto pt : valid_points)
    //     out << "v " << pt.transpose() << " 0" << std::endl;
    // for (const Edge& edge : edges)
    //     out << "l " << (edge + Edge::Ones()).transpose() << std::endl;
    // out.close();
    // std::exit(0);
}



void Tiling3D::buildSimulationMeshFromTilingInfo(int IH, T* params,
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    PointLoops point_loop_unit;
    // T square_width = 0.52;
    T square_width = 0.4;
    std::vector<Edge> edges;
    std::vector<TV2> unique_points;
    // clapBottomLayerWithSquare(IH, params, point_loop_unit, unique_points, edges, square_width);

    fetchTilingCropped(IH, params, unique_points, edges, square_width);
    struct Vertex
    {
        TV x;
        std::vector<Edge> edges;
        Vertex(const TV& _x) : x(_x) {}
    };
    
    std::vector<Vertex> vertices;
    for (TV2& pt : unique_points)
    {
        vertices.push_back(Vertex(TV(pt[0], pt[1], 0.0)));
    }

    T width = 0.04; // nozzle diameter
    
    for (const Edge& edge : edges)
    {
        vertices[edge[0]].edges.push_back(edge);
        vertices[edge[1]].edges.push_back(Edge(edge[1], edge[0]));
    }

    std::ofstream out("tiling_2d_edges.obj");
    
    for (const TV2& vtx : unique_points)
    {
        out << "v " << vtx.transpose() << " 0" << std::endl;
    }
    for (const Edge& edge : edges)
        out << "l " << edge.transpose() + Edge::Ones().transpose() << std::endl;

    out.close();

    T height = 0.1;
    T thickness = 0.01;
    int sub_divide_width = 5;
    int sub_divide_height = std::floor(height / thickness);
    std::vector<TV> mesh_vertices;
    std::vector<Face> mesh_faces;

    std::unordered_map<Edge, IdList, VectorHash<2>> edge_vtx_tracker;
    
    auto appendVtxToEdge = [&](const Vertex& vtx, const Edge& edge, int vtx_idx)
    {
        if (edge_vtx_tracker.find(edge) == edge_vtx_tracker.end())
            edge_vtx_tracker[edge] = {vtx_idx};
        else
            edge_vtx_tracker[edge].push_back(vtx_idx);
    };

    // sort edge by angle
    for (Vertex& vtx : vertices)
    {
        TV pos = vtx.x;
        TV ref = (vertices[edges[0][1]].x - pos).normalized();
        TV avg_normal(-0.1, -0.1, -1);

        std::sort(vtx.edges.begin(), vtx.edges.end(), [&](const Edge& edge_a, const Edge& edge_b)
        {
            TV E0 = (vertices[edge_a[1]].x - pos).normalized();
            TV E1 = (vertices[edge_b[1]].x - pos).normalized();

            T dot_sign0 = E0.dot(ref);
            T dot_sign1 = E1.dot(ref);
            TV cross_sin0 = E0.cross(ref);
            TV cross_sin1 = E1.cross(ref);

            T angle_a = cross_sin0.dot(avg_normal) < 0 ? std::acos(dot_sign0) : 2.0 * M_PI - std::acos(dot_sign0);
            T angle_b = cross_sin1.dot(avg_normal) < 0 ? std::acos(dot_sign1) : 2.0 * M_PI - std::acos(dot_sign1);
            
            return angle_a < angle_b;
        });
    }
    // insert bisecting vector vtx
    int cnt = 0;
    TM2 R90 = TM2::Zero();

    R90.row(0) = TV2(0, -1);
    R90.row(1) = TV2(1, 0);
    std::vector<Edge> dangling_edges;
    std::vector<TV> dangling_edge_face_normal;
    for (const Vertex& vtx : vertices)
    {
        IdList ixn_vtx_ids;
        
        if (vtx.edges.size() == 1)
        {
            
            TV vi = vtx.x, vj = vertices[vtx.edges[0][1]].x;
            TV left = TV::Zero(), right = TV::Zero();
            TV dir = (vj - vi).normalized();
            left.head<2>() = vi.head<2>() - R90 * dir.head<2>() * 0.5 * thickness;
            right.head<2>() = vi.head<2>() + R90 * dir.head<2>() * 0.5 * thickness;

            mesh_vertices.push_back(left);
            int vtx_idx0 = mesh_vertices.size() - 1;            
            appendVtxToEdge(vtx, vtx.edges[0], vtx_idx0);
            mesh_vertices.push_back(right);
            int vtx_idx1 = mesh_vertices.size() - 1;          
            appendVtxToEdge(vtx, vtx.edges[0], vtx_idx1);
            dangling_edges.push_back(Edge(vtx_idx0, vtx_idx1));
            dangling_edge_face_normal.push_back(-dir);
        }
        else
        {
            for (int i = 0; i < vtx.edges.size(); i++)
            {
                int j = (i + 1) % vtx.edges.size();
                
                TV vi = vtx.x, vj = vertices[vtx.edges[i][1]].x, vk = vertices[vtx.edges[j][1]].x;
                
                TV bisec = TV::Zero();

                T cross_dot = (vj - vi).cross(vk - vi).dot(TV(-1e4, -1e4, -1));
                if (cross_dot > 0)
                    bisec = vi + ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;
                else
                    bisec = vi - ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;

                TV dir = (vj - vi).normalized();

                if (dir.cross((vk-vi).normalized()).norm() < 1e-6)
                    bisec.head<2>() = vi.head<2>() - R90 * dir.head<2>() * 0.5 * thickness;


                mesh_vertices.push_back(bisec);
                int vtx_idx = mesh_vertices.size() - 1;            
                ixn_vtx_ids.push_back(vtx_idx);
                appendVtxToEdge(vtx, vtx.edges[i], vtx_idx);
                appendVtxToEdge(vtx, vtx.edges[j], vtx_idx);
            }
        }
        

        if (ixn_vtx_ids.size() == 3)
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
        else if (ixn_vtx_ids.size() == 4)
        {
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[2], ixn_vtx_ids[3]));
        }
        else if (ixn_vtx_ids.size() == 5)
        {
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[2], ixn_vtx_ids[3]));
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[3], ixn_vtx_ids[4]));
        }
        else if (ixn_vtx_ids.size() > 5)
        {
            Eigen::MatrixXd boundary_vertices;
            Eigen::MatrixXi boundary_edges;
            Eigen::MatrixXd points_inside_a_hole;

            // Triangulated interior
            Eigen::MatrixXd V2;
            Eigen::MatrixXi F2;

            boundary_vertices.resize(ixn_vtx_ids.size(), 2);
            boundary_edges.resize(ixn_vtx_ids.size(), 2);

            for (int k = 0; k < ixn_vtx_ids.size(); k++)
            {
                boundary_vertices.row(k) = mesh_vertices[ixn_vtx_ids[k]].head<2>();
                int l = (k + 1) % ixn_vtx_ids.size();
                boundary_edges.row(k) = Edge(k, l);
            }
            
            std::string cmd = "pY";
            igl::triangle::triangulate(boundary_vertices,
                boundary_edges,
                points_inside_a_hole, 
                cmd, 
                // "a0.005q",
                V2, F2);
            for (int k = 0; k < F2.rows(); k++)
            {
                mesh_faces.push_back(Face(ixn_vtx_ids[F2(k, 0)], ixn_vtx_ids[F2(k, 1)], ixn_vtx_ids[F2(k, 2)]));
            }
            
        }
            
        cnt ++;
    }

    out.open("tiling_bisec_vtx.obj");
    
    for (const TV& vtx : mesh_vertices)
    {
        out << "v " << vtx.transpose() << std::endl;
    }

    out.close();

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
            int n_sub_div = std::floor(std::min((v3 - v0).norm(), (v2-v1).norm()) / thickness);
            // int n_sub_div = sub_divide_width;
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
            int n_sub_div = std::floor(std::min((v2 - v0).norm(), (v3-v1).norm()) / thickness);
            // int n_sub_div = sub_divide_width;
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
        mesh_vertices.push_back(vtx + TV(0.0, 0.0, height));
    
    TV min_corner, max_corner;
    min_corner.setConstant(1e6); max_corner.setConstant(-1e6);
    for (int i = nv; i < mesh_vertices.size(); i++)
    {
        for (int d = 0; d < 3; d++)
        {
            min_corner[d] = std::min(mesh_vertices[i][d], min_corner[d]);
            max_corner[d] = std::max(mesh_vertices[i][d], max_corner[d]);
        }
    }

    TV center = 0.5 * (min_corner + max_corner);
    Matrix<T, 2, 2> rotation;
    T r_angle = 5.0 / 180.0 * M_PI;
    rotation << std::cos(r_angle), -std::sin(r_angle), std::sin(r_angle), std::cos(r_angle);
    for (int i = nv; i < mesh_vertices.size(); i++)
    {
        mesh_vertices[i].head<2>() = center.head<2>() + rotation * (mesh_vertices[i] - center).head<2>();
    }

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

    
    auto addFace = [&](const TV& edge_vec, const Face& f, bool invert)
    {
        TV n = normal(f);
        T sign = invert ? -1.0 : 1.0;
        Face face = f;
        if (sign * edge_vec.dot(n) < 0)
        {
            face = inverseNormal(f);
        }
        mesh_faces.push_back(face);
    };

    auto addVertex = [&](const TV& vtx)->int
    {
        auto find_iter = std::find_if(mesh_vertices.begin(), mesh_vertices.end(), 
                            [&vtx](const TV x)->bool
                               { return (x - vtx).norm() < 1e-6; }
                             );
        if (find_iter == mesh_vertices.end())
        {
            mesh_vertices.push_back(vtx);
            return mesh_vertices.size() - 1;
        }
        else
        {
            return std::distance(mesh_vertices.begin(), find_iter);
        }
        
    };

    for (int i = 0; i < boundary_edges.size() / 2; i++)
    {
        Edge ei = boundary_edges[i * 2 + 0];
        Edge ei_oppo = boundary_edges[i * 2 + 1];
        TV v0 = mesh_vertices[ei[0]], v1 = mesh_vertices[ei[1]];
        TV v2 = mesh_vertices[ei_oppo[0]], v3 = mesh_vertices[ei_oppo[1]];

        TV edge_vec = (v2 - v0 + TV(1e-4, 1e-4, 1e-4)).normalized();

        TV delta_vec0 = (mesh_vertices[ei[0] + nv] - mesh_vertices[ei[0]]) / T(sub_divide_height);
        TV delta_vec1 = (mesh_vertices[ei[1] + nv] - mesh_vertices[ei[1]]) / T(sub_divide_height);
        TV delta_vec2 = (mesh_vertices[ei_oppo[0] + nv] - mesh_vertices[ei_oppo[0]]) / T(sub_divide_height);
        TV delta_vec3 = (mesh_vertices[ei_oppo[1] + nv] - mesh_vertices[ei_oppo[1]]) / T(sub_divide_height);

        int loop0 = ei[0], loop1 = ei[1], loop2 = ei_oppo[0], loop3 = ei_oppo[1];
        for (int j = 1; j < sub_divide_height; j++)
        {
            int idx0 = addVertex(v0 + j * delta_vec0);
            int idx1 = addVertex(v1 + j * delta_vec1);
            int idx2 = addVertex(v2 + j * delta_vec2);
            int idx3 = addVertex(v3 + j * delta_vec3);

            addFace(edge_vec, Face(loop1, loop0, idx0), false);
            addFace(edge_vec, Face(loop1, idx0, idx1), false);

            addFace(edge_vec, Face(loop2, loop3, idx2), true);
            addFace(edge_vec, Face(idx2, loop3, idx3), true);

            loop0 = idx0; loop1 = idx1; loop2 = idx2; loop3 = idx3;
        }
        addFace(edge_vec, Face(loop1, loop0, ei[0] + nv), false);
        addFace(edge_vec, Face(loop1, ei[0] + nv, ei[1] + nv), false);

        addFace(edge_vec, Face(loop2, loop3, ei_oppo[0] + nv), true);
        addFace(edge_vec, Face(ei_oppo[0] + nv, loop3, ei_oppo[1] + nv), true);
    }

    for (int i = 0; i < dangling_edges.size(); i++)
    {
        Edge ei = dangling_edges[i];
        TV v0 = mesh_vertices[ei[0]], v1 = mesh_vertices[ei[1]];
        TV delta_vec0 = (mesh_vertices[ei[0] + nv] - mesh_vertices[ei[0]]) / T(sub_divide_height);
        TV delta_vec1 = (mesh_vertices[ei[1] + nv] - mesh_vertices[ei[1]]) / T(sub_divide_height);
        int loop0 = ei[0], loop1 = ei[1];
        for (int j = 1; j < sub_divide_height; j++)
        {
            int idx0 = addVertex(v0 + j * delta_vec0);
            int idx1 = addVertex(v1 + j * delta_vec1);

            addFace(dangling_edge_face_normal[i], Face(loop1, loop0, idx0), true);
            addFace(dangling_edge_face_normal[i], Face(loop1, idx0, idx1), true);

            loop0 = idx0; loop1 = idx1; 
        }
        addFace(dangling_edge_face_normal[i], Face(loop1, loop0, ei[0] + nv), true);
        addFace(dangling_edge_face_normal[i], Face(loop1, ei[0] + nv, ei[1] + nv), true);
    }
    
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

void Tiling3D::buildSimulationMesh(const std::vector<PointLoops>& raw_points,
    Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    T height = 1.0;
    T thickness = 0.1;
    int sub_divide_width = 5;
    int sub_divide_height = std::floor(height / thickness);
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

    std::ofstream out("tiling_2d_edges.obj");
    
    for (const TV2& vtx : unique_points)
    {
        out << "v " << vtx.transpose() << " 0" << std::endl;
    }
    for (const Edge& edge : edges)
        out << "l " << edge.transpose() + Edge::Ones().transpose() << std::endl;

    out.close();

    std::unordered_map<Edge, IdList, VectorHash<2>> edge_vtx_tracker;
    
    auto appendVtxToEdge = [&](const Vertex& vtx, const Edge& edge, int vtx_idx)
    {
        if (edge_vtx_tracker.find(edge) == edge_vtx_tracker.end())
            edge_vtx_tracker[edge] = {vtx_idx};
        else
            edge_vtx_tracker[edge].push_back(vtx_idx);
    };

    // sort edge by angle
    for (Vertex& vtx : vertices)
    {
        TV pos = vtx.x;
        TV ref = (vertices[edges[0][1]].x - pos).normalized();
        TV avg_normal(-0.1, -0.1, -1);

        std::sort(vtx.edges.begin(), vtx.edges.end(), [&](const Edge& edge_a, const Edge& edge_b)
        {
            TV E0 = (vertices[edge_a[1]].x - pos).normalized();
            TV E1 = (vertices[edge_b[1]].x - pos).normalized();

            T dot_sign0 = E0.dot(ref);
            T dot_sign1 = E1.dot(ref);
            TV cross_sin0 = E0.cross(ref);
            TV cross_sin1 = E1.cross(ref);

            T angle_a = cross_sin0.dot(avg_normal) < 0 ? std::acos(dot_sign0) : 2.0 * M_PI - std::acos(dot_sign0);
            T angle_b = cross_sin1.dot(avg_normal) < 0 ? std::acos(dot_sign1) : 2.0 * M_PI - std::acos(dot_sign1);
            
            return angle_a < angle_b;
        });
    }
    // insert bisecting vector vtx
    int cnt = 0;
    TM2 R90 = TM2::Zero();

    R90.row(0) = TV2(0, -1);
    R90.row(1) = TV2(1, 0);
    for (const Vertex& vtx : vertices)
    {
        IdList ixn_vtx_ids;
        
        for (int i = 0; i < vtx.edges.size(); i++)
        {
            int j = (i + 1) % vtx.edges.size();
            
            TV vi = vtx.x, vj = vertices[vtx.edges[i][1]].x, vk = vertices[vtx.edges[j][1]].x;
            
            TV bisec = TV::Zero();
            // only two edges here and this is the larger angle
            // if (j == 0 && i == 1)
            //     bisec = vi - ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;
            // else
            //     bisec = vi + ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;

            T cross_dot = (vj - vi).cross(vk - vi).dot(TV(-1e4, -1e4, -1));
            if (cross_dot > 0)
                bisec = vi + ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;
            else
                bisec = vi - ((vj - vi).normalized() + (vk - vi).normalized()).normalized() * 0.5 * thickness;

            
            

            if ((vj-vi).cross(vk-vi).norm() < 1e-6)
                bisec.head<2>() = vi.head<2>() - R90 * (vj - vi).head<2>() * 0.5 * thickness;


            mesh_vertices.push_back(bisec);
            int vtx_idx = mesh_vertices.size() - 1;            
            ixn_vtx_ids.push_back(vtx_idx);
            appendVtxToEdge(vtx, vtx.edges[i], vtx_idx);
            appendVtxToEdge(vtx, vtx.edges[j], vtx_idx);
        }
        if (ixn_vtx_ids.size() == 3)
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
        else if (ixn_vtx_ids.size() == 4)
        {
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[2], ixn_vtx_ids[3]));
        }
        else if (ixn_vtx_ids.size() == 5)
        {
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[1], ixn_vtx_ids[2]));
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[2], ixn_vtx_ids[3]));
            mesh_faces.push_back(Face(ixn_vtx_ids[0], ixn_vtx_ids[3], ixn_vtx_ids[4]));
        }
        else if (ixn_vtx_ids.size() > 5)
        {
            Eigen::MatrixXd boundary_vertices;
            Eigen::MatrixXi boundary_edges;
            Eigen::MatrixXd points_inside_a_hole;

            // Triangulated interior
            Eigen::MatrixXd V2;
            Eigen::MatrixXi F2;

            boundary_vertices.resize(ixn_vtx_ids.size(), 2);
            boundary_edges.resize(ixn_vtx_ids.size(), 2);

            for (int k = 0; k < ixn_vtx_ids.size(); k++)
            {
                boundary_vertices.row(k) = mesh_vertices[ixn_vtx_ids[k]].head<2>();
                int l = (k + 1) % ixn_vtx_ids.size();
                boundary_edges.row(k) = Edge(k, l);
            }
            
            std::string cmd = "pY";
            igl::triangle::triangulate(boundary_vertices,
                boundary_edges,
                points_inside_a_hole, 
                cmd, 
                // "a0.005q",
                V2, F2);
            for (int k = 0; k < F2.rows(); k++)
            {
                mesh_faces.push_back(Face(ixn_vtx_ids[F2(k, 0)], ixn_vtx_ids[F2(k, 1)], ixn_vtx_ids[F2(k, 2)]));
            }
            
        }
            
        cnt ++;
    }

    out.open("tiling_bisec_vtx.obj");
    
    for (const TV& vtx : mesh_vertices)
    {
        out << "v " << vtx.transpose() << std::endl;
    }

    out.close();

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
            int n_sub_div = std::floor(std::min((v3 - v0).norm(), (v2-v1).norm()) / thickness);
            // int n_sub_div = sub_divide_width;
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
            int n_sub_div = std::floor(std::min((v2 - v0).norm(), (v3-v1).norm()) / thickness);
            // int n_sub_div = sub_divide_width;
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

    out.open("tiling_bottom_edges_bd.obj");
    
    for (const TV& vtx : mesh_vertices)
    {
        out << "v " << vtx.transpose() << std::endl;
    }
    for (const Edge& edge : boundary_edges)
        out << "l " << edge.transpose() + Edge::Ones().transpose() << std::endl;

    out.close();

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

    
    auto addFace = [&](const TV& edge_vec, const Face& f, bool invert)
    {
        TV n = normal(f);
        T sign = invert ? -1.0 : 1.0;
        Face face = f;
        if (sign * edge_vec.dot(n) < 0)
        {
            face = inverseNormal(f);
        }
        mesh_faces.push_back(face);
    };

    auto addVertex = [&](const TV& vtx)->int
    {
        auto find_iter = std::find_if(mesh_vertices.begin(), mesh_vertices.end(), 
                            [&vtx](const TV x)->bool
                               { return (x - vtx).norm() < 1e-6; }
                             );
        if (find_iter == mesh_vertices.end())
        {
            mesh_vertices.push_back(vtx);
            return mesh_vertices.size() - 1;
        }
        else
        {
            return std::distance(mesh_vertices.begin(), find_iter);
        }
        
    };

    for (int i = 0; i < boundary_edges.size() / 2; i++)
    {
        Edge ei = boundary_edges[i * 2 + 0];
        Edge ei_oppo = boundary_edges[i * 2 + 1];
        TV v0 = mesh_vertices[ei[0]], v1 = mesh_vertices[ei[1]];
        TV v2 = mesh_vertices[ei_oppo[0]], v3 = mesh_vertices[ei_oppo[1]];

        TV edge_vec = (v2 - v0 + TV(1e-4, 1e-4, 1e-4)).normalized();

        TV delta_vec = (mesh_vertices[ei[0] + nv] - mesh_vertices[ei[0]]) / T(sub_divide_height);

        int loop0 = ei[0], loop1 = ei[1], loop2 = ei_oppo[0], loop3 = ei_oppo[1];
        for (int j = 1; j < sub_divide_height; j++)
        {
            int idx0 = addVertex(v0 + j * delta_vec);
            int idx1 = addVertex(v1 + j * delta_vec);
            int idx2 = addVertex(v2 + j * delta_vec);
            int idx3 = addVertex(v3 + j * delta_vec);

            addFace(edge_vec, Face(loop1, loop0, idx0), false);
            addFace(edge_vec, Face(loop1, idx0, idx1), false);

            addFace(edge_vec, Face(loop2, loop3, idx2), true);
            addFace(edge_vec, Face(idx2, loop3, idx3), true);

            loop0 = idx0; loop1 = idx1; loop2 = idx2; loop3 = idx3;
        }
        addFace(edge_vec, Face(loop1, loop0, ei[0] + nv), false);
        addFace(edge_vec, Face(loop1, ei[0] + nv, ei[1] + nv), false);

        addFace(edge_vec, Face(loop2, loop3, ei_oppo[0] + nv), true);
        addFace(edge_vec, Face(ei_oppo[0] + nv, loop3, ei_oppo[1] + nv), true);
    }
    

    out.open("test_tiling.obj");
    
    for (const TV& vtx : mesh_vertices)
    {
        out << "v " << vtx.transpose() << std::endl;
    }
    for (const Face& face : mesh_faces)
        out << "f " << face.transpose() + IV::Ones().transpose() << std::endl;

    out.close();
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

void Tiling3D::initializeSimulationData(bool tetgen)
{
    bool load_mesh = true;
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    
    std::vector<int> dirichlet_vertices;

    if (load_mesh)
    {
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling1.obj", V, F);
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling6.obj", V, F);
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling1.obj", V, F);

        //used
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling8.obj", V, F);
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling0.obj", V, F);
        
        igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling10.obj", V, F);
        // dirichlet_vertices = {8, 9, 1, 0, 11, 10, 19, 18, 27, 26, 35, 34, 43, 42,
        //     166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 1176, 177
        // };
        

        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling11.obj", V, F);

        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling13.obj", V, F);
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling13_rotate.obj", V, F);
        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/Tiling3D/data/tiling13_shift.obj", V, F);
    }
    else
    {
        std::vector<PointLoops> raw_points;

        // fetchTilingVtxLoop(raw_points);

        // buildSimulationMesh(raw_points, V, F, C);

        int IH = 0;
        T params[] = {0.1161, 0.5464, 0.4313, 0.5464}; //Isohedral 0
        
        // int IH = 6;
        // T params[] = {0.5, 0.5, 0.5, 0.5, 0.5}; //Isohedral 06
        
        // int IH = 29;
        // T params[] = {0}; //Isohedral 7

        // int IH = 13;
        // T params[] = {0.1, 0.2}; 

        // int IH = 2;
        // T params[] = {0.12239750492, 0.5, 0.225335752741, 0.625}; 

        // int IH = 5;
        // T params[] = {0.12239750492, 0.5, 0.225335752741, 0.625, 0.5};
        // int IH = 24;
        // T params[] = {0.5, 0.230769230769, 0.5, 0.5};
        
        // int IH = 9;
        // T params[] = {0.12239750492, 0.225335752741};

        buildSimulationMeshFromTilingInfo(IH, params, V, F, C);    
    }
    
    if (tetgen || load_mesh)
    {
        Eigen::MatrixXd TV;
        Eigen::MatrixXi TT;
        Eigen::MatrixXi TF;
        
        Eigen::VectorXd tmp = V.col(2);
        V.col(2) = V.col(1);
        V.col(1) = tmp;
        igl::copyleft::tetgen::tetrahedralize(V,F, "pq1.414Y", TV,TT,TF);
        // igl::copyleft::tetgen::tetrahedralize(V,F, "Y", TV,TT,TF);
        
        // solver.project_block_PD = true;
        // solver.compute_bending_stiffness = true;
        solver.initializeElementData(TV, TF, TT);

        Vector<bool, 4> flag;
        flag.setConstant(false);
        flag[0] = true; flag[2] = true;
        flag[1] = true; flag[3] = true;
        // solver.addCornerVtxToDirichletVertices(flag);
        // if (flag[1])
        //     solver.bending_direction = 45.0 / 180.0 * M_PI;
        // else
        //     solver.bending_direction = 135.0 / 180.0 * M_PI;

        // solver.bending_direction = 45.0 / 180.0 * M_PI;
        // solver.curvature = 1;
        // solver.max_newton_iter = 1000;
        // solver.computeCylindricalBendingBCPenaltyPairs();

        // solver.imposeCylindricalBending();

        // solver.fixEndPointsX();
        // solver.dragMiddle();

        // solver.applyForceLeftRight();
        // solver.applyForceTopBottom();

        // solver.ThreePointBendingTestWithCylinder();
        // solver.ThreePointBendingTest();
        // solver.addForceMiddleTop();
        // solver.fixNodes(dirichlet_vertices);
        // solver.penaltyInPlaneCompression(0, 0.1);

    }
    else
    {
        solver.initializeSurfaceData(V, F);
    }
    


}

