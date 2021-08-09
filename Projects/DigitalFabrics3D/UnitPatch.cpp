#include <iostream>
#include <utility>
#include <fstream>
#include <unordered_map>

#include "UnitPatch.h"
#include "HybridC2Curve.h"

#include "tactile/tiling.hpp"
#include "IO.h"



// static double ROD_A = 0.05;
// static double ROD_B = 0.05;

static double ROD_A = 5e-4;
static double ROD_B = 5e-4;

#include <random>
#include <cmath>
std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );

static double zeta()
{
	return dis(gen);
}

template<class T, int dim>
void UnitPatch<T, dim>::addPoint(const TV& point, int& full_dof_cnt, int& node_cnt)
{
    deformed_states.conservativeResize(full_dof_cnt + dim);
    deformed_states.template segment<dim>(full_dof_cnt) = point;
    full_dof_cnt += dim;
    node_cnt++;
}

template<class T, int dim>
void UnitPatch<T, dim>::buildScene(int patch_type)
{
    if (patch_type == 0)
        build3DtestScene(4);
    else if (patch_type == 1)
        buildOneCrossScene(16);
    else if (patch_type == 2)
        buildGridScene(64);
    else if (patch_type == 3)
        buildOmegaScene(2);
    else if (patch_type == 4)
        buildStraightRodScene(16);
    else if (patch_type == 5)
        buildTactiles(2);
    else if (patch_type == 6)
        loadFromTiles(2);
    else if (patch_type == 7)
        buildFingerScene(8);
}

// template<class T, int dim>
// void UnitPatch<T, dim>::buildFingerScene(int sub_div)
// {
//     if constexpr (dim == 3)
//     {
//         auto unit_yarn_map = sim.yarn_map;
//         sim.yarn_map.clear();
//         sim.add_rotation_penalty = false;
//         sim.add_pbc_bending = false;
//         sim.add_pbc_twisting = false;
//         sim.add_pbc = false;

//         sim.add_contact_penalty=true;
//         sim.new_frame_work = true;
//         sim.add_eularian_reg = true;
        
//         sim.ke = 1e-2;

//         clearSimData();
        
//         int full_dof_cnt = 0;
//         int node_cnt = 0;
//         int rod_cnt = 0;

//         std::vector<Entry> w_entry;

//         std::vector<std::vector<TV>> passing_points(3, std::vector<TV>());
//         std::vector<std::vector<int>> passing_points_id(3, std::vector<int>());

//         for (int i = 0; i < 4; i++)
//         {
//             for (int j = 0; j < 3; j++)
//             {
//                 TV v0 = TV(0.25 * (j+1), 1.0 - 0.25 * i, 0.0) * sim.unit;
//                 addPoint(v0, full_dof_cnt, node_cnt);
//                 passing_points_id[j].push_back(i * 3 + j);
//                 passing_points[j].push_back(v0);
//             }
//         }

//         // for (int i = 0; i < 4; i++)
//         // {
//         //     for (int j = 0; j < 3; j++)
//         //     {
//         //         TV v0 = TV(0.25 * (j+1), 1.0 - 0.25 * i, -0.1) * sim.unit;
//         //         addPoint(v0, full_dof_cnt, node_cnt);
//         //         passing_points_id[3 + j].push_back(12 + i * 3 + j);
//         //         passing_points[3 + j].push_back(v0);
//         //     }
//         // }
        
//         for (int i = 0; i < 4; i++)
//         {
//             TV from = deformed_states.template segment<dim>(i * 3 * dim);
//             TV middle = deformed_states.template segment<dim>((i * 3 + 1) * dim);
//             TV to = deformed_states.template segment<dim>((i * 3 + 2) * dim);
//             std::vector<TV> points = { from, middle, to };
//             std::vector<int> ids = { i * 3, i * 3 + 1, i * 3 + 2 };
//             addAStraightRod(from, to, points, ids, 
//                 sub_div, full_dof_cnt, node_cnt, rod_cnt, false);            
//         }

//         // for (int i = 0; i < 4; i++)
//         // {
//         //     TV from = deformed_states.template segment<dim>((12 + i * 3) * dim);
//         //     TV middle = deformed_states.template segment<dim>((12 + i * 3 + 1) * dim);
//         //     TV to = deformed_states.template segment<dim>((12 + i * 3 + 2) * dim);
//         //     std::vector<TV> points = { from, middle, to };
//         //     std::vector<int> ids = { 12 + i * 3, 12 + i * 3 + 1, 12 + i * 3 + 2 };
//         //     addAStraightRod(from, to, points, ids, 
//         //         sub_div, full_dof_cnt, node_cnt, rod_cnt, false);            
//         // }

//         addAStraightRod(passing_points[0].front(), passing_points[0].back(), passing_points[0], passing_points_id[0], 
//             sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);

//         TV from = passing_points[1].front();
//         TV to = TV(0.5, 0, 0) * sim.unit;
//         addAStraightRod(from, to, passing_points[1], passing_points_id[1], 
//             sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);
        
        
//         addAStraightRod(passing_points[2].front(), passing_points[2].back(), passing_points[2], passing_points_id[2], 
//             sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);

//         // addAStraightRod(passing_points[3].front(), passing_points[3].back(), passing_points[3], passing_points_id[3], 
//         //     sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);

//         // addAStraightRod(passing_points[4].front(), passing_points[4].back(), passing_points[4], passing_points_id[4], 
//         //     sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);
        
//         // addAStraightRod(passing_points[5].front(), passing_points[5].back(), passing_points[5], passing_points_id[5], 
//         //     sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);


//         // for (int i = 0; i < 4; i++)
//         // {
//         //     for (int j = 0; j < 3; j++)
//         //     {
//         //         if (j == 1)
//         //             continue;
//         //         TV from = deformed_states.template segment<dim>((12 + i * 3 + j) * dim);
//         //         TV to = deformed_states.template segment<dim>((i * 3 + j) * dim);
//         //         std::vector<TV> points = { from, to };
//         //         std::vector<int> ids = { 12 + i * 3 + j , i * 3 + j};
//         //         addAStraightRod(from, to, points, ids, 
//         //             sub_div, full_dof_cnt, node_cnt, rod_cnt, false);            
//         //     }
//         // }

//         for (int i = 0; i < 4; i++)
//         {
//             for (int j = 0; j < 3; j++)
//             {
                
//                 RodCrossing<T, dim>* crossing = 
//                         new RodCrossing<T, dim>(i * 3 + j, {i, 4 + j});
                            
//                 crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
//                 crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
                
//                 crossing->on_rod_idx[i] = sim.Rods[i]->dof_node_location[j];
//                 crossing->on_rod_idx[4 + j] = sim.Rods[4+j]->dof_node_location[i];
                
//                 if (i == 3 && j==1 )
//                 {
                    
//                     crossing->is_fixed = false;
//                     crossing->sliding_ranges.push_back(Range(0, 0));
//                     crossing->sliding_ranges.push_back(Range(0.05, 0.05));
//                 }
//                 else if (i == 2 && j == 1)
//                 {
//                     crossing->is_fixed = false;
//                     crossing->sliding_ranges.push_back(Range(0, 0));
//                     crossing->sliding_ranges.push_back(Range(0.0, 0.0));
//                 }
//                 else
//                 {
//                     crossing->is_fixed = true;
//                     crossing->sliding_ranges.push_back(Range(0, 0));
//                     crossing->sliding_ranges.push_back(Range(0, 0));
//                 }
                
//                 sim.rod_crossings.push_back(crossing);
//             }
            
//         }

//         // for (int i = 0; i < 4; i++)
//         // {
//         //     for (int j = 0; j < 3; j++)
//         //     {
//         //         for (int k = 0; k < 2; k++)
//         //         {
//         //             if (j == 1)
//         //             {
//         //                 RodCrossing<T, dim>* crossing = 
//         //                     new RodCrossing<T, dim>(i * 3 + j + k * 12, {i, 4 + j});
                                
//         //                 crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
//         //                 crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
                        
//         //                 crossing->on_rod_idx[i] = sim.Rods[i]->dof_node_location[j];
//         //                 crossing->on_rod_idx[4 + j] = sim.Rods[4+j]->dof_node_location[i];
                        
//         //                 if (i == 3 && j==1 )
//         //                 {
                            
//         //                     crossing->is_fixed = false;
//         //                     crossing->sliding_ranges.push_back(Range(0, 0));
//         //                     crossing->sliding_ranges.push_back(Range(0.05, 0.05));
//         //                 }
//         //                 else if (i == 2 && j == 1)
//         //                 {
//         //                     crossing->is_fixed = false;
//         //                     crossing->sliding_ranges.push_back(Range(0, 0));
//         //                     crossing->sliding_ranges.push_back(Range(0.0, 0.0));
//         //                 }
//         //                 else
//         //                 {
//         //                     crossing->is_fixed = true;
//         //                     crossing->sliding_ranges.push_back(Range(0, 0));
//         //                     crossing->sliding_ranges.push_back(Range(0, 0));
//         //                 }
                        
//         //                 sim.rod_crossings.push_back(crossing);
//         //             }
//         //             else
//         //             {

//         //             }
                    
//         //         }
//         //     }
            
//         // }
        
//         int dof_cnt = 0;
//         markCrossingDoF(w_entry, dof_cnt);
        
//         for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
        
//         appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
        
//         sim.rest_states = deformed_states;

//         sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
//         sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
//         for (auto& rod : sim.Rods)
//         {
//             rod->fixEndPointEulerian(sim.dirichlet_dof);
//             rod->setupBishopFrame();
//         }

//         Offset offset;
//         sim.Rods[5]->getEntry(10, offset);
//         for (int d = 0; d < dim; d++)
//         {        
//             sim.dirichlet_dof[sim.Rods[5]->reduced_map[offset[d]]] = 0;
//         }

        
    
//         sim.Rods[5]->backOffsetReduced(offset);
//         sim.dirichlet_dof[offset[0]] = 0;
//         sim.dirichlet_dof[offset[1]] = -sim.unit * 0.5;
//         sim.dirichlet_dof[offset[2]] = 0;

//         sim.Rods[4]->backOffsetReduced(offset);
//         sim.dirichlet_dof[offset[0]] = 0;
//         sim.dirichlet_dof[offset[1]] = 0;
//         sim.dirichlet_dof[offset[2]] = 0;

//         sim.Rods[6]->backOffsetReduced(offset);
//         sim.dirichlet_dof[offset[0]] = 0;
//         sim.dirichlet_dof[offset[1]] = 0;
//         sim.dirichlet_dof[offset[2]] = 0;
        
//         sim.fixCrossing();
        
//     }
// }

template<class T, int dim>
void UnitPatch<T, dim>::buildFingerScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.add_pbc_twisting = false;
        sim.add_pbc = false;

        sim.add_contact_penalty=true;
        sim.new_frame_work = true;
        sim.add_eularian_reg = true;
        
        sim.ke = 1e-4;

        clearSimData();
        
        int full_dof_cnt = 0;
        int node_cnt = 0;
        int rod_cnt = 0;

        std::vector<Entry> w_entry;

        std::vector<std::vector<TV>> passing_points(6, std::vector<TV>());
        std::vector<std::vector<int>> passing_points_id(6, std::vector<int>());

        for (int k = 0; k < 2; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    TV v0 = TV(0.25 * (j+1), 1.0 - 0.25 * i, -0.1 * T(k)) * sim.unit;
                    addPoint(v0, full_dof_cnt, node_cnt);
                    passing_points_id[3*k + j].push_back( 12 * k + i * 3 + j);
                    passing_points[3*k + j].push_back(v0);
                }
            }
        }
        
        
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                TV from = deformed_states.template segment<dim>((12 * j + i * 3) * dim);
                TV middle = deformed_states.template segment<dim>((12 * j + i * 3 + 1) * dim);
                TV to = deformed_states.template segment<dim>((12 * j + i * 3 + 2) * dim);
                std::vector<TV> points = { from, middle, to };
                std::vector<int> ids = { 12 * j + i * 3, 12 * j + i * 3 + 1, 12 * j + i * 3 + 2 };
                addAStraightRod(from, to, points, ids, 
                    sub_div, full_dof_cnt, node_cnt, rod_cnt, false);            
            }
        }

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                TV from = deformed_states.template segment<dim>((12 + i * 3 + j) * dim);
                TV to = deformed_states.template segment<dim>((i * 3 + j) * dim);
                std::vector<TV> points = { from, to };
                std::vector<int> ids = { 12 + i * 3 + j , i * 3 + j};
                if (j == 1)
                    continue;
                addAStraightRod(from, to, points, ids, 
                    sub_div, full_dof_cnt, node_cnt, rod_cnt, false);            
            }
        }

        // addAStraightRod(passing_points[0].front(), passing_points[0].back(), passing_points[0], passing_points_id[0], 
        //     sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);

        TV from = passing_points[1].front();
        TV to = TV(0.5, 0, 0) * sim.unit;
        addAStraightRod(from, to, passing_points[1], passing_points_id[1], 
            sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);
        
        for (int i = 3; i < 6; i++)
        {
            addAStraightRod(passing_points[i].front(), passing_points[i].back(), passing_points[i], passing_points_id[i], 
                sub_div * 2, full_dof_cnt, node_cnt, rod_cnt, false);
        }

        for (int k = 0; k < 2; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int rod0 = 4 * k + i; // x direction
                    int rod1 = j == 0 ? 8 + i * 2 : 8 + i * 2 + 1; // z direction
                    int rod2 = k == 0 ? 16 : 17 + j; // y direction
                    
                    std::vector<int> rods_involved;
                    if (k == 0 && j != 1)
                        rods_involved = {rod0, rod1};
                    else if (j == 1)
                        rods_involved = {rod0, rod2};
                    else
                        rods_involved = {rod0, rod1, rod2};

                    RodCrossing<T, dim>* crossing = 
                            new RodCrossing<T, dim>(k * 12 + i * 3 + j, rods_involved);
                                
                    crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
                    if (j != 1)
                        crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
                    if (k==1 || (k ==0 && j == 1))
                        crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
                    
                    crossing->on_rod_idx[rod0] = sim.Rods[rod0]->dof_node_location[j];
                    
                    if (j != 1)
                        crossing->on_rod_idx[rod1] = sim.Rods[rod1]->dof_node_location[1] - sim.Rods[rod1]->dof_node_location[k];
                    
                    if (k==1 || (k ==0 && j == 1))
                        crossing->on_rod_idx[rod2] = sim.Rods[rod2]->dof_node_location[i];
                    
                    
                    if (i > 0 && j==1 && k == 0)
                    // if (i == 3 && j==1 && k == 0)
                    // if(false)
                    {
                        crossing->is_fixed = false;
                        crossing->sliding_ranges.push_back(Range(0, 0));
                        crossing->sliding_ranges.push_back(Range(0.2, 0.2));
                        // std::cout << "free" << std::endl;
                    }
                    else
                    {
                        crossing->is_fixed = true;
                        crossing->sliding_ranges.push_back(Range(0, 0));
                        if (j != 1)
                            crossing->sliding_ranges.push_back(Range(0, 0));
                        if (k==1 || (k ==0 && j == 1))
                            crossing->sliding_ranges.push_back(Range(0, 0));
                    }
                    
                    sim.rod_crossings.push_back(crossing);
                }
                
            }
        }

    
        
        int dof_cnt = 0;
        markCrossingDoF(w_entry, dof_cnt);
        
        for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
        
        appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
        
        sim.rest_states = deformed_states;

        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        for (auto& rod : sim.Rods)
        {
            rod->fixEndPointEulerian(sim.dirichlet_dof);
            rod->setupBishopFrame();
        }

        Offset offset;
        sim.Rods[16]->getEntry(10, offset);
        for (int d = 0; d < dim; d++)
        {        
            sim.dirichlet_dof[sim.Rods[16]->reduced_map[offset[d]]] = 0;
        }

    
        sim.Rods[16]->backOffsetReduced(offset);
        sim.dirichlet_dof[offset[0]] = 0;
        sim.dirichlet_dof[offset[1]] = -sim.unit * 0.1;
        sim.dirichlet_dof[offset[2]] = 0;

        sim.Rods[3]->fixEndPointLagrangian(sim.dirichlet_dof);
        
        // for (int i = 17; i < 20; i++)
        // {
        //     sim.Rods[i]->backOffsetReduced(offset);
            
        //     sim.dirichlet_dof[offset[0]] = 0;
        //     sim.dirichlet_dof[offset[1]] = 0;
        //     sim.dirichlet_dof[offset[2]] = 0;
        // }
        // for (int i = 0; i < sim.Rods[16]->numSeg(); i++)
        // {
        //     sim.dirichlet_dof[sim.Rods[16]->theta_reduced_dof_start_offset + i] = 0;
        // }

        
        sim.fixCrossing();
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::fetchOneFamily(std::vector<TV2>& data_points, int IH, 
    TV2& T1, TV2& T2, bool random)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    T1 = TV2(a_tiling.getT1().x, a_tiling.getT1().y);
    T2 = TV2(a_tiling.getT2().x, a_tiling.getT2().y);
    size_t num_params = a_tiling.numParameters();
    if( num_params > 1 ) {
        double params[ num_params ];
        // Get the parameters out of the tiling
        a_tiling.getParameters( params );
        // Change a parameter
        for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) {
            if (random)
                params[idx] += zeta()*0.2 - 0.1;
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
            ej.push_back( dvec2( 0, 0 ) );
            ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
            ej.push_back( 
                dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
            ej.push_back( dvec2( 1, 0 ) );
        }
        else
        {
            ej.push_back( dvec2( 0, 0 ) );
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
    data_points.push_back(TV2(p[0], p[1]) * sim.unit);
    // std::cout << p[0] << " " << p[1] << std::endl;
    for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
        dvec2 p1 = dvec3( shape[idx], 1.0 );
        dvec2 p2 = dvec3( shape[idx+1], 1.0 );
        dvec2 p3 = dvec3( shape[idx+2], 1.0 );

        data_points.push_back(TV2(p1[0], p1[1]) * sim.unit);
        data_points.push_back(TV2(p2[0], p2[1]) * sim.unit);
        data_points.push_back(TV2(p3[0], p3[1]) * sim.unit);        
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildTactiles(int sub_div)
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
        ej.push_back( dvec2( zeta() * 0.75, zeta() * 0.6 - 0.3 ) );
        ej.push_back( 
            dvec2( zeta() * 0.75 + 0.25, zeta() * 0.6 - 0.3 ) );
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

    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = false;
    sim.add_pbc_bending = false;
    sim.new_frame_work = true;

    clearSimData();
    // std::vector<TV2> data_points;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    std::vector<Entry> w_entry;


    for( auto i : a_tiling.fillRegion(-2.0, -2.0, 2.0, 2.0 ) ) {

        std::vector<TV2> data_points;
        dmat3 TT = i->getTransform();
        
        dvec2 p = TT * dvec3( shape.back(), 1.0 );
        data_points.push_back(TV2(p[0], p[1]) * sim.unit);
        // std::cout << p[0] << " " << p[1] << std::endl;
        for( size_t idx = 0; idx < shape.size(); idx += 3 ) {
            dvec2 p1 = TT * dvec3( shape[idx], 1.0 );
            dvec2 p2 = TT * dvec3( shape[idx+1], 1.0 );
            dvec2 p3 = TT * dvec3( shape[idx+2], 1.0 );

            data_points.push_back(TV2(p1[0], p1[1]) * sim.unit);
            data_points.push_back(TV2(p2[0], p2[1]) * sim.unit);
            data_points.push_back(TV2(p3[0], p3[1]) * sim.unit);
        }
        
        addCurvedRod(data_points, sub_div, full_dof_cnt, node_cnt, rod_cnt, true);
        
    }
    
    // std::cout << "add curved rod" << std::endl;
    int dof_cnt = 0;
    // markCrossingDoF(w_entry, dof_cnt);

    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;

    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        rod->setupBishopFrame();
    }
    sim.rest_states = sim.deformed_states;

    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'



template<class T, int dim>
void UnitPatch<T, dim>::cropTranslationalUnitByparallelogram(const std::vector<std::vector<TV2>>& input_points,
    std::vector<TV2>& output_points, const TV2& top_left, const TV2& top_right,
    const TV2& bottom_right, const TV2& bottom_left, std::vector<Vector<int, 2>>& edge_pairs,
    std::unordered_map<int, std::vector<int>>& crossing_tracker,
    std::vector<std::vector<Vector<int, 2>>>& boundary_pairs,
    std::vector<std::vector<int>>& boundary_pair_rod_idx)
{
    if constexpr (dim == 3)
    {
        crossing_tracker.clear();

        std::vector<TV2> parallogram = {top_left, top_right, bottom_right, bottom_left};

        using Edge = std::pair<TV2, TV2>;

        std::vector<Edge> edges;

        boundary_pairs.resize(4, std::vector<Vector<int, 2>>());
        boundary_pair_rod_idx.resize(4, std::vector<int>());
        
        for (auto one_tile : input_points)
        {
            // -2 because the tile vertices loop back to the first one
            for (int i = 0; i < one_tile.size() - 2; i++)
            {
                const TV2 xi = one_tile[i];
                const TV2 xj = one_tile[i+ 1];
                
                bool xi_inside = insidePolygon(parallogram, xi);
                bool xj_inside = insidePolygon(parallogram, xj);

                // both points are inside the parallelogram
                if (xi_inside && xj_inside)
                {
                    
                    Edge xij = std::make_pair(xi, xj);

                    auto find_edge_iter = std::find_if(edges.begin(), edges.end(), [&xij](Edge e)
                        {   
                            return (((e.first - xij.first).norm() < 1e-6) && ((e.second - xij.second).norm() < 1e-6)) || 
                                (((e.first - xij.second).norm() < 1e-6) && ((e.second - xij.first).norm() < 1e-6));
                        }
                    );

                    bool new_edge = find_edge_iter == edges.end();
                    
                    if(new_edge)
                    {
                        edges.push_back(std::make_pair(xi, xj));
                        // std::cout << xi.transpose() << " " << xj.transpose() << std::endl;
                        auto find_xi_iter = std::find_if(output_points.begin(), output_points.end(), 
                            [&xi](const TV2 x)->bool
                               { return (x - xi).norm() < 1e-6; }
                             );
                        int xi_idx = -1, xj_idx = -1;
                    
                        if (find_xi_iter == output_points.end())
                        {
                            // xi is a new vtx
                            output_points.push_back(xi);
                            xi_idx = int(output_points.size()) - 1;
                            //pre push this edge
                            crossing_tracker[xi_idx] = {int(edge_pairs.size())};
                        }
                        else
                        {
                            int index = std::distance(output_points.begin(), find_xi_iter);
                            if (crossing_tracker.find(index) == crossing_tracker.end())
                            {
                                crossing_tracker[index] = {int(edge_pairs.size())};
                            }
                            else
                            {
                                crossing_tracker[index].push_back(int(edge_pairs.size()));
                            }
                            xi_idx = index;
                        }

                        auto find_xj_iter = std::find_if(output_points.begin(), output_points.end(), 
                                [&xj](const TV2 x)->bool
                               { return (x - xj).norm() < 1e-6; }
                        );
                        if (find_xj_iter == output_points.end())
                        {
                            output_points.push_back(xj);
                            xj_idx = int(output_points.size()) - 1;
                            crossing_tracker[xj_idx] = {int(edge_pairs.size())};
                        }
                        else
                        {
                            int index = std::distance(output_points.begin(), find_xj_iter);
                            if (crossing_tracker.find(index) == crossing_tracker.end())
                            {
                                crossing_tracker[index] = {int(edge_pairs.size())};
                            }
                            else
                            {
                                crossing_tracker[index].push_back(int(edge_pairs.size()));
                            }
                            xj_idx = index;
                        }
                        
                        edge_pairs.push_back(Vector<int,2>(xi_idx, xj_idx));
                    }
                }
                else if(!xi_inside && xj_inside)
                {
                    
                    // std::cout << "One is inside" << std::endl;
                    Edge xij = std::make_pair(xi, xj);
                    auto find_edge_iter = std::find_if(edges.begin(), edges.end(), [&xij](Edge e)
                        {   
                            return (((e.first - xij.first).norm() < 1e-6) && ((e.second - xij.second).norm() < 1e-6)) || 
                                (((e.first - xij.second).norm() < 1e-6) && ((e.second - xij.first).norm() < 1e-6));
                        }
                    );

                    bool new_edge = find_edge_iter == edges.end();

                    if(new_edge)
                    {
                        edges.push_back(std::make_pair(xi, xj));
                        TV2 intersection;
                        int xj_idx = -1;
                        bool intersected = false;
                        int intersecting_edge = -1;
                        if (lineSegementsIntersect2D(xi, xj, parallogram[0], parallogram[1], intersection))
                        {
                            intersected = true;
                            intersecting_edge = 0;
                        }
                        else if(lineSegementsIntersect2D(xi, xj, parallogram[1], parallogram[2], intersection))
                        {
                            intersected = true;
                            intersecting_edge = 1;
                        }
                        else if(lineSegementsIntersect2D(xi, xj, parallogram[2], parallogram[3], intersection))
                        {
                            intersected = true;
                            intersecting_edge = 2;
                        }
                        else if (lineSegementsIntersect2D(xi, xj, parallogram[3], parallogram[0], intersection))
                        {
                            intersected = true;
                            intersecting_edge = 3;
                        }
                        if (intersected)
                        {
                            
                            output_points.push_back(intersection);
                            int xi_idx = output_points.size() - 1;
                            crossing_tracker[xi_idx] = {xi_idx};

                            auto find_xj_iter = std::find_if(output_points.begin(), output_points.end(), [&xj](const TV2 x)->bool
                               { return (x - xj).norm() < 1e-6; }
                             );
                            if (find_xj_iter == output_points.end())
                            {
                                output_points.push_back(xj);
                                xj_idx = int(output_points.size()) - 1;
                                crossing_tracker[xj_idx] = {int(edge_pairs.size())};
                            }
                            else
                            {
                                int index = std::distance(output_points.begin(), find_xj_iter);
                                if (crossing_tracker.find(index) == crossing_tracker.end())
                                {
                                    crossing_tracker[index] = {int(edge_pairs.size())};
                                }
                                else
                                {
                                    crossing_tracker[index].push_back(int(edge_pairs.size()));
                                }
                                xj_idx = index;
                            }

                            boundary_pairs[intersecting_edge].push_back(Vector<int,2>(xj_idx, xi_idx));
                            boundary_pair_rod_idx[intersecting_edge].push_back(edge_pairs.size());
                            edge_pairs.push_back(Vector<int,2>(xj_idx, xi_idx));
                        }
                    }
                    
                }
                else if(!xj_inside && xi_inside)
                {
                    // ignored due to duplicated edges
                }
                else
                {
                    // continue;
                }
            }
        }
    
        
    }
    
}

template<class T, int dim>
void UnitPatch<T, dim>::loadFromTiles(int sub_div)
{
    auto unit_yarn_map = sim.yarn_map;
    sim.yarn_map.clear();
    sim.add_rotation_penalty = true;
    sim.add_pbc_bending = false;
    sim.add_pbc_twisting = false;
    sim.add_pbc = true;
    sim.add_contact_penalty=false;
    sim.new_frame_work = true;

    clearSimData();
    // std::vector<TV2> data_points;
    int full_dof_cnt = 0;
    int node_cnt = 0;
    int rod_cnt = 0;

    std::vector<Entry> w_entry;

    std::vector<std::vector<TV2>> all_points;

    std::vector<TV2> data_points;
    TV2 T1, T2;
    //10 is good
    fetchOneFamily(data_points, 0, T1, T2, false);    

    // std::cout << std::acos(T1.dot(TV2(1, 0)))/M_PI * 180 << " " << std::acos(T2.dot(TV2(1, 0)))/M_PI * 180 << std::endl;

    TV2 center = TV2::Zero();
    for (TV2& pt : data_points)
        center += pt;
    center /= T(data_points.size() - 1);

    int n_tile_T1 = 2, n_tile_T2 = 2;

    std::vector<TV2> parallogram;
    parallogram.push_back(center + TV2(0, 0) * sim.unit);
    parallogram.push_back(center + T(n_tile_T1 - 1) * T1 * sim.unit);
    parallogram.push_back(center + T(n_tile_T1 - 1) * T1 * sim.unit + T(n_tile_T2 - 1) * T2 * sim.unit);
    parallogram.push_back(center + T(n_tile_T2 - 1) * T2 * sim.unit);
    parallogram.push_back(center + TV2(0, 0) * sim.unit);
    // addCurvedRod(parallogram, sub_div, full_dof_cnt, node_cnt, rod_cnt, true);  
    all_points.push_back(data_points);
    // addCurvedRod(data_points, sub_div, full_dof_cnt, node_cnt, rod_cnt, true);

    for (int tile_T1 = 0; tile_T1 < n_tile_T1; tile_T1++)
    {
        for (int tile_T2 = 0; tile_T2 < n_tile_T2; tile_T2++)
        {
            std::vector<TV2> shifted_points = data_points;
            for (TV2& pt : shifted_points)
            {
                pt += T(tile_T1) * T1 * sim.unit + T(tile_T2) * T2 * sim.unit;
            }
            all_points.push_back(shifted_points);
            // addCurvedRod(shifted_points, sub_div, full_dof_cnt, node_cnt, rod_cnt, true);  
        }
    }
    
    std::vector<TV2> valid_points;
    std::vector<Vector<int, 2>> edge_pairs;
    std::unordered_map<int, std::vector<int>> crossing_tracker;
    std::vector<std::vector<Vector<int, 2>>> boundary_pairs;
    std::vector<std::vector<int>> boundary_pair_rod_idx;

    cropTranslationalUnitByparallelogram(all_points, valid_points, 
        parallogram[1], parallogram[2], parallogram[3], parallogram[4], edge_pairs, crossing_tracker,
        boundary_pairs, boundary_pair_rod_idx);

    // hard coded i for four edges
    std::vector<std::vector<std::pair<int, TV2>>> sort_pairs;
    for (int i = 0; i < 4; i++)
    {
        int cnt = 0;
        std::vector<std::pair<int, TV2>> sort_pair;
        for (const Vector<int, 2>& edge : boundary_pairs[i])
        {
            TV2 vtx = valid_points[edge[1]];
            sort_pair.push_back(std::make_pair(cnt++, vtx));
        }
        if (i == 0 || i == 1)
            std::sort(sort_pair.begin(), sort_pair.end(), [parallogram,i](std::pair<int, TV2> a, std::pair<int, TV2> b){
                return  (a.second - parallogram[i+1]).norm() < (b.second - parallogram[i+1]).norm();
            });
        else
            std::sort(sort_pair.begin(), sort_pair.end(), [parallogram,i](std::pair<int, TV2> a, std::pair<int, TV2> b){
                return  (a.second - parallogram[(i+2)%5]).norm() < (b.second - parallogram[(i+2)%5]).norm();
            });
        sort_pairs.push_back(sort_pair);
    }

    // extrudeMeshToObj(valid_points, edge_pairs, "hex.obj", 0.1 * sim.unit);
    
    std::unordered_map<int, int> node_idx_map;

    if constexpr (dim == 3)
    {
        std::vector<bool> is_crossing(valid_points.size(), false);
        
        for (auto& element : crossing_tracker)
        {
            TV2 vtx = valid_points[element.first];
            std::vector<int> edges_from_vtx = element.second;
            if (edges_from_vtx.size() >= 2)
            {
                node_idx_map[element.first] = node_cnt;
                deformed_states.conservativeResize(full_dof_cnt + dim);
                deformed_states.template segment<2>(full_dof_cnt) = vtx;
                deformed_states[full_dof_cnt+dim-1] = 0.0;
                full_dof_cnt += dim;
                node_cnt++;
                is_crossing[element.first] = true;
            }
        }
        // std::cout << node_cnt << std::endl;
        for (auto & pair : edge_pairs)
        {
            int v1 = pair[0], v2 = pair[1];
            TV from = TV(valid_points[v1][0], valid_points[v1][1], 0);
            TV to = TV(valid_points[v2][0], valid_points[v2][1], 0);

            TV2 dir = (from - to).normalized().template segment<2>(0);
            
            T theta1 = dir.dot(T1);
            T theta2 = dir.dot(T2);

            // std::cout << theta1 << " " << theta2 << std::endl;

            if (std::acos(theta1) > std::acos(theta2))
            {
                if (theta1 < 0)
                {
                    std::swap(v1, v2);
                    std::swap(from, to);
                    std::swap(pair[0], pair[1]);
                }
            }
            else
            {
                if (theta2 < 0)
                {
                    std::swap(v1, v2);
                    std::swap(from, to);
                    std::swap(pair[0], pair[1]);
                }
            }

            std::vector<TV> passing_points; 
            std::vector<int> passing_points_id; 
            if (is_crossing[v1] && is_crossing[v2])
            {
                passing_points = { from, to };
                passing_points_id = { node_idx_map[v1], node_idx_map[v2] }; 
            }
            else if (is_crossing[v1] && !is_crossing[v2])
            {
                passing_points = { from };
                passing_points_id = { node_idx_map[v1] }; 
            }
            else if (!is_crossing[v1] && is_crossing[v2])
            {
                passing_points = { to };
                passing_points_id = { node_idx_map[v2] }; 
                node_idx_map[v1] = node_cnt;
                
            }
            else
            {
                node_idx_map[v1] = node_cnt;   
            }

            addAStraightRod(from, to, passing_points, passing_points_id, 
                sub_div, full_dof_cnt, node_cnt, rod_cnt, false);

            if (is_crossing[v1] && !is_crossing[v2])
            {
                node_idx_map[v2] = node_cnt - 1;
            }
            else if(!is_crossing[v1] && !is_crossing[v2])
            {
                node_idx_map[v2] = node_cnt - 1;
                
            }
        }
        
        // std::cout << crossing_tracker.size () << " " << valid_points.size() << std::endl;
        for (auto& element : crossing_tracker)
        {
            TV2 vtx = valid_points[element.first];
            std::vector<int> edges_from_vtx = element.second;
            if (edges_from_vtx.size() < 2)
                continue;
            RodCrossing<T, dim>* crossing = 
                    new RodCrossing<T, dim>(node_idx_map[element.first], edges_from_vtx);
            
            for (int rod_idx : edges_from_vtx)
            {
                // std::cout << rod_idx << " ";
                crossing->sliding_ranges.push_back(Range(0, 0));
                crossing->undeformed_twist.push_back(Vector<T, 2>(0, 0)); 
                TV2 vi = valid_points[edge_pairs[rod_idx][0]];

                if ((vi - vtx).norm() < 1e-6 )
                    crossing->on_rod_idx[rod_idx] = 0;
                else
                    crossing->on_rod_idx[rod_idx] = sim.Rods[rod_idx]->numSeg();
            }
            // std::cout << std::endl;
            
            
            crossing->is_fixed = true;
            
            sim.rod_crossings.push_back(crossing);
            
        }

    }
    

    int dof_cnt = 0;
    markCrossingDoF(w_entry, dof_cnt);
    
    for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
    
    appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
    
    sim.rest_states = deformed_states;

    sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
    sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
    for (auto& rod : sim.Rods)
    {
        rod->fixEndPointEulerian(sim.dirichlet_dof);
        // for (int i = 0; i < rod->numSeg(); i++)
        // {
        //     sim.dirichlet_dof[rod->theta_reduced_dof_start_offset + i] = 0;
        // }
        rod->setupBishopFrame();
    }

    if constexpr (dim == 3)
    {
        // for (auto edges : boundary_pairs)
        //     for (auto edge : edges)
        //         std::cout << edge.transpose() << std::endl;

        for (int i = 0; i < sort_pairs[0].size(); i++)
        {
            
            Offset a, b, c, d, e, f, g, h;

            // location at original array
            int location0 = sort_pairs[0][i].first;
            int location1 = sort_pairs[1][i].first;
            int location2 = sort_pairs[2][i].first;
            int location3 = sort_pairs[3][i].first;

            auto rod0 = sim.Rods[boundary_pair_rod_idx[0][location0]];
            auto rod1 = sim.Rods[boundary_pair_rod_idx[1][location1]];
            auto rod2 = sim.Rods[boundary_pair_rod_idx[2][location2]];
            auto rod3 = sim.Rods[boundary_pair_rod_idx[3][location3]];

            Vector<int, 2> edge0 = boundary_pairs[0][location0];
            Vector<int, 2> edge1 = boundary_pairs[1][location1];
            Vector<int, 2> edge2 = boundary_pairs[2][location2];
            Vector<int, 2> edge3 = boundary_pairs[3][location3];

            std::cout << node_idx_map[edge0[1]] << " " <<  node_idx_map[edge0[1]] - 1
                << " " << node_idx_map[edge2[1]] - 1 << " " << node_idx_map[edge2[1]]<< " "
                << " " << node_idx_map[edge3[1]]  << " " << node_idx_map[edge3[1]] - 1<< " "
                << node_idx_map[edge1[1]] - 1 << " " <<  node_idx_map[edge1[1]]
                 << std::endl;

            rod0->getEntry(node_idx_map[edge0[1]] - 1, b);
            rod0->getEntry(node_idx_map[edge0[1]], a);
            rod2->getEntry(node_idx_map[edge2[1]] - 1, c);
            rod2->getEntry(node_idx_map[edge2[1]], d);

            sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(a, d)));

            sim.pbc_bending_pairs.push_back({a, b, c, d});
            sim.pbc_bending_pairs_rod_id.push_back({rod0->rod_id, rod0->rod_id, rod2->rod_id, rod2->rod_id});


            rod3->getEntry(node_idx_map[edge3[1]] - 1, f);
            rod3->getEntry(node_idx_map[edge3[1]], e);
            rod1->getEntry(node_idx_map[edge1[1]] - 1, g);
            rod1->getEntry(node_idx_map[edge1[1]], h);

            sim.pbc_pairs.push_back(std::make_pair(1, std::make_pair(e, h)));
            
            
            sim.pbc_bending_pairs.push_back({e, f, g, h});
            sim.pbc_bending_pairs_rod_id.push_back({rod3->rod_id, rod3->rod_id, rod1->rod_id, rod1->rod_id});

            
            if (i == 0)
            {
                sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(a, d), std::make_pair(rod0->rod_id, rod2->rod_id));
                sim.pbc_pairs_reference[1] = std::make_pair(std::make_pair(e, h), std::make_pair(rod3->rod_id, rod1->rod_id));

                sim.dirichlet_dof[rod0->reduced_map[a[0]]] = 0;
                sim.dirichlet_dof[rod0->reduced_map[a[1]]] = 0;
                sim.dirichlet_dof[rod0->reduced_map[a[2]]] = 0;
            }
            
        }
    }

    // if constexpr (dim == 3)
    // {
    //     TV b(1, 0, 0);
    //     TV n(0, 0, 1);

    //     for (auto& crossing : sim.rod_crossings)
    //     {
    //         Matrix<T, 3, 3> rb_frames = Matrix<T, 3, 3>::Identity();
            
    //         rb_frames = crossing->rotation_accumulated * rb_frames;
    //         for(int rod_idx : crossing->rods_involved)
    //         {
    //             int node_loc = crossing->on_rod_idx[rod_idx];
    
    //             auto rod = sim.Rods[rod_idx];
    //             T udt0 = 0, udt1 = 0;
    //             if (node_loc > -1 && node_loc < rod->numSeg() - 1)
    //             {
    //                 TV ut = parallelTransportOrthonormalVector(n, b, rod->prev_tangents[node_loc]);
    //                 udt1 = signedAngle(ut, rod->reference_frame_us[node_loc], rod->prev_tangents[node_loc]);
    //             }
    //             if (node_loc < rod->numSeg() && node_loc > 0)
    //             {
    //                 TV ut = parallelTransportOrthonormalVector(rod->reference_frame_us[node_loc - 1], rod->prev_tangents[node_loc - 1], b);
    //                 udt0 = signedAngle(ut, n, b);
    //             }
                       
    //             crossing->undeformed_twist.push_back(Vector<T, 2>(udt0, udt1)); 
    //         }
    //     }
    // }

    if (sim.disable_sliding)
    {
        sim.fixCrossing();
    }
    else
    {
        for (auto& crossing : sim.rod_crossings)
        {
            for (int d = 0; d < dim; d++)
            {   
                sim.dirichlet_dof[crossing->reduced_dof_offset + d] = 0;    
            }
        }
    }

    Offset end0, end1;
    // sim.Rods[0]->frontOffset(end0); 
    
    // sim.dirichlet_dof[sim.Rods[0]->reduced_map[end0[0]]] = 0;
    // sim.dirichlet_dof[sim.Rods[0]->reduced_map[end0[1]]] = 0;
    // sim.dirichlet_dof[sim.Rods[0]->reduced_map[end0[2]]] = 0;

    // sim.Rods.back()->backOffset(end0); 
    // sim.dirichlet_dof[sim.Rods.back()->reduced_map[end0[0]]] = 0.05 * sim.unit;
    // sim.dirichlet_dof[sim.Rods.back()->reduced_map[end0[1]]] = 0.05 * sim.unit;
    // sim.dirichlet_dof[sim.Rods.back()->reduced_map[end0[2]]] = 0.05 * sim.unit;

    // sim.Rods[1]->backOffset(end0); 
    // sim.dirichlet_dof[sim.Rods[1]->reduced_map[end0[0]]] = 0;
    // sim.dirichlet_dof[sim.Rods[1]->reduced_map[end0[1]]] = 0;
    // sim.dirichlet_dof[sim.Rods[1]->reduced_map[end0[2]]] = 0;

    // sim.Rods[2]->backOffset(end0); 
    // sim.dirichlet_dof[sim.Rods[2]->reduced_map[end0[0]]] = 0;
    // sim.dirichlet_dof[sim.Rods[2]->reduced_map[end0[1]]] = 0;
    // sim.dirichlet_dof[sim.Rods[2]->reduced_map[end0[2]]] = 0;

}

template<class T, int dim>
void UnitPatch<T, dim>::buildStraightRodScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();

        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;
        int rod_cnt = 0;
        
        TV from = TV(0, 0.5, 0) * sim.unit;
        TV to = TV(2, 0.5001, 0.001) * sim.unit;

        std::vector<int> passing_points_id;
        std::vector<TV> passing_points;

        addAStraightRod(from, to, passing_points, passing_points_id, 
                sub_div, full_dof_cnt, node_cnt, rod_cnt, false);
        
        int dof_cnt;
        for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
        
        appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
        
        sim.rest_states = deformed_states;
    
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());


        int cnt = 0;
        for (auto& rod : sim.Rods)
        {
            std::cout << rod->kt << std::endl;
            // rod->kt  =0 ;
            rod->fixEndPointEulerian(sim.dirichlet_dof);
            sim.dirichlet_dof[rod->theta_reduced_dof_start_offset] = 0;
            // sim.dirichlet_dof[rod->theta_reduced_dof_start_offset + rod->indices.size()-1] = 0;
            rod->setupBishopFrame();
            Offset end0, end1;
            rod->frontOffset(end0); rod->backOffset(end1);
            sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
        }

        Offset end0, end1;
        sim.Rods[0]->frontOffset(end0); sim.Rods[0]->backOffset(end1);
        sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(end0, end1), std::make_pair(0, 0));

        sim.Rods[0]->fixPointLagrangian(0, TV::Zero(), sim.dirichlet_dof);
        sim.Rods[0]->fixPointLagrangian(1, TV::Zero(), sim.dirichlet_dof);

        // sim.Rods[0]->fixPointLagrangian(sim.Rods[0]->indices.size() - 2, 
        //         TV(-0.3, 0.01, 0.01) * sim.unit, 
        //         sim.dirichlet_dof);
        // sim.Rods[0]->fixPointLagrangian(sim.Rods[0]->indices.size() - 1, 
        //         TV(-0.3, 0.01, 0.01) * sim.unit, 
        //         sim.dirichlet_dof); 


    }
}

template<class T, int dim>
void UnitPatch<T, dim>::addCurvedRod(const std::vector<TV2>& data_points,
        int sub_div, int& full_dof_cnt, int& node_cnt, int& rod_cnt, bool closed)
{
    int sub_div_2 = sub_div / 2;
    HybridC2Curve<T, 2>* curve = new HybridC2Curve<T, 2>(sub_div);
    for (const auto& pt : data_points)
        curve->data_points.push_back(pt);

    std::vector<TV2> points_on_curve;
    curve->sampleCurves(points_on_curve);

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));
    
    Rod<T, dim>* r0 = new Rod<T, dim>(deformed_states, sim.rest_states, rod_cnt, closed, ROD_A, ROD_B);
    std::unordered_map<int, Offset> offset_map;
    std::vector<int> node_index_list;
    std::vector<T> data_points_discrete_arc_length;
    
    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[i] = Offset::Zero();
        node_index_list.push_back(i);
        //push Lagrangian DoF
        TV2 pt = points_on_curve[i];
        if constexpr (dim == 3)
            deformed_states.template segment<dim>(full_dof_cnt) = TV(pt[0], pt[1], pt[2]);
        else if constexpr (dim == 2)
            deformed_states.template segment<dim>(full_dof_cnt) = pt;

        offset_map[i][0] = full_dof_cnt++;
        offset_map[i][1] = full_dof_cnt++;
        offset_map[i][2] = full_dof_cnt++;
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = T(i) * (curve->data_points.size() - 1) / (points_on_curve.size() - 1);
        offset_map[i][3] = full_dof_cnt++;

        node_cnt++;
    }

    r0->offset_map = offset_map;
    r0->indices = node_index_list;
    
    for(int i = 0; i < curve->data_points.size(); i++)
        data_points_discrete_arc_length.push_back(deformed_states[i*sub_div_2*(dim+1)]);
    
    Vector<T, dim + 1> q0, q1;
    r0->frontDoF(q0); r0->backDoF(q1);

    DiscreteHybridCurvature<T, dim>* rest_state_rod0 = new DiscreteHybridCurvature<T, dim>(q0, q1);
    sim.curvature_functions.push_back(rest_state_rod0);
    rest_state_rod0->setData(curve, data_points_discrete_arc_length);

    r0->rest_state = rest_state_rod0;
    r0->dof_node_location = {};
    sim.Rods.push_back(r0);
    rod_cnt++;
}

template<class T, int dim>
void UnitPatch<T, dim>::buildOmegaScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();

        std::vector<Entry> w_entry;
    
        int full_dof_cnt = 0;
        int node_cnt = 0;
        int rod_cnt = 0;
        std::vector<TV2> data_points;

        data_points.push_back(TV2(0.0, 0.0) * sim.unit);
        data_points.push_back(TV2(0.4, 0.0) * sim.unit);
        data_points.push_back(TV2(0.2, 0.3) * sim.unit);
        data_points.push_back(TV2(0.8, 0.3) * sim.unit);
        data_points.push_back(TV2(0.6, 0.0) * sim.unit);
        data_points.push_back(TV2(1.0, 0.0) * sim.unit);

        addCurvedRod(data_points, sub_div, full_dof_cnt, node_cnt, rod_cnt, false);

        int dof_cnt = 0;
        // markCrossingDoF(w_entry, dof_cnt);

        for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
        
        appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
        
        sim.rest_states = deformed_states;
    
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        
        // std::cout << sim.W << std::endl;
        
        int cnt = 0;
        for (auto& rod : sim.Rods)
        {            
            rod->fixEndPointEulerian(sim.dirichlet_dof);
            // sim.dirichlet_dof[rod->theta_reduced_dof_start_offset] = M_PI/8.0;
            // sim.dirichlet_dof[rod->theta_reduced_dof_start_offset] = 0.0;
            rod->setupBishopFrame();
            Offset end0, end1;
            rod->frontOffset(end0); rod->backOffset(end1);
            sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
        }

        sim.rest_states = sim.deformed_states;
        
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        Offset end0, end1;
        sim.Rods[0]->frontOffset(end0); sim.Rods[0]->backOffset(end1);
        sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(end0, end1), std::make_pair(0,0));

        Offset ob, of;
        sim.Rods[0]->backOffsetReduced(ob);
        sim.Rods[0]->frontOffsetReduced(of);


        sim.dirichlet_dof[of[0]] = 0.0 * sim.unit;
        sim.dirichlet_dof[of[1]] = 0.0 * sim.unit;
        sim.dirichlet_dof[of[2]] = 0.0 * sim.unit;

        sim.dirichlet_dof[ob[0]] = 0.2 * sim.unit;
        sim.dirichlet_dof[ob[1]] = 0.0 * sim.unit;
        sim.dirichlet_dof[ob[2]] = 0.1 * sim.unit;

        
    }
}

// this assumes passing points to be pushed before everything
template<class T, int dim>
void UnitPatch<T, dim>::addAStraightRod(const TV& from, const TV& to, 
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div, int& full_dof_cnt, int& node_cnt, int& rod_cnt, bool closed)
{
    
    std::unordered_map<int, Offset> offset_map;

    std::vector<TV> points_on_curve;
    std::vector<int> rod_indices;
    std::vector<int> key_points_location_rod;
    addStraightYarnCrossNPoints(from, to, passing_points, passing_points_id,
                                sub_div, points_on_curve, rod_indices,
                                key_points_location_rod, node_cnt);
                   

    deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));

    Rod<T, dim>* rod = new Rod<T, dim>(deformed_states, sim.rest_states, rod_cnt, closed, ROD_A, ROD_B);

    for (int i = 0; i < points_on_curve.size(); i++)
    {
        offset_map[node_cnt] = Offset::Zero();
        //push Lagrangian DoF    
        deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
        for (int d = 0; d < dim; d++)
        {
            offset_map[node_cnt][d] = full_dof_cnt++;  
        }
        // push Eulerian DoF
        deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
        offset_map[node_cnt][dim] = full_dof_cnt++;
        node_cnt++;
    }
    
    deformed_states.conservativeResize(full_dof_cnt + passing_points.size());

    for (int i = 0; i < passing_points.size(); i++)
    {
        deformed_states[full_dof_cnt] = (passing_points[i] - from).norm() / (to - from).norm();
        offset_map[passing_points_id[i]] = Offset::Zero();
        offset_map[passing_points_id[i]][dim] = full_dof_cnt++; 
        Vector<int, dim> offset_dof_lag;
        for (int d = 0; d < dim; d++)
        {
            offset_dof_lag[d] = passing_points_id[i] * dim + d;
        }
        offset_map[passing_points_id[i]].template segment<dim>(0) = offset_dof_lag;
    }
    
    rod->offset_map = offset_map;
    rod->indices = rod_indices;
    Vector<T, dim + 1> q0, q1;
    rod->frontDoF(q0); rod->backDoF(q1);

    rod->rest_state = new LineCurvature<T, dim>(q0, q1);
    
    rod->dof_node_location = key_points_location_rod;
    
    sim.Rods.push_back(rod);
    rod_cnt++;
}

template<class T, int dim>
void UnitPatch<T, dim>::appendThetaAndJointDoF(std::vector<Entry>& w_entry, 
    int& full_dof_cnt, int& dof_cnt)
{
    // for (auto& rod : sim.Rods)
    // {
    //     rod->theta_dof_start_offset = full_dof_cnt;
    //     rod->theta_reduced_dof_start_offset = dof_cnt;
    //     deformed_states.conservativeResize(full_dof_cnt + rod->indices.size() - 1);
    //     for (int i = 0; i < rod->indices.size() - 1; i++)
    //         w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
    //     deformed_states.template segment(rod->theta_dof_start_offset, 
    //         rod->indices.size() - 1).setZero();
    // }   

    deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * dim);
    deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * dim).setZero();

    for (auto& crossing : sim.rod_crossings)
    {
        crossing->dof_offset = full_dof_cnt;
        crossing->reduced_dof_offset = dof_cnt;
        for (int d = 0; d < dim; d++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }
    }

    for (auto& rod : sim.Rods)
    {
        rod->theta_dof_start_offset = full_dof_cnt;
        rod->theta_reduced_dof_start_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + rod->indices.size() - 1);
        for (int i = 0; i < rod->indices.size() - 1; i++)
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        deformed_states.template segment(rod->theta_dof_start_offset, 
            rod->indices.size() - 1).setZero();
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildGridScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = true;
        sim.add_pbc_bending = true;
        sim.add_pbc = true;
        sim.new_frame_work = true;

        clearSimData();

        sim.unit = 0.03;
        // sim.unit = 4;
        // std::cout << sim.unit << std::endl;
        
        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;

        int n_row = 4, n_col = 4;

        // push crossings first 
        T dy = 1.0 / n_row * sim.unit;
        T dx = 1.0 / n_col * sim.unit;
        
        //num of crossing
        deformed_states.resize(n_col * n_row * dim);
        
        std::unordered_map<int, Offset> crossing_offset_copy;

        auto getXY = [=](int row, int col, T& x, T& y)
        {
            if (row == 0) y = 0.5 * dy;
            else if (row == n_row) y = n_row * dy;
            else y = 0.5 * dy + (row ) * dy;
            if (col == 0) x = 0.5 * dx;
            else if (col == n_col) x = n_col * dx;
            else x = 0.5 * dx + (col ) * dx;
        };

        for (int row = 0; row < n_row; row++)
        {
            for (int col = 0; col < n_col; col++)
            {
                T x, y;
                getXY(row, col, x, y);
                deformed_states.template segment<dim>(node_cnt * dim) = TV(x, y, 0);
                
                full_dof_cnt += dim;
                node_cnt ++;       
            }
        }

        int rod_cnt = 0;
        for (int row = 0; row < n_row; row++)
        {
            T x0 = 0.0, x1 = 1.0 * sim.unit;
            T x, y;
            
            std::vector<int> passing_points_id;
            std::vector<TV> passing_points;
            
            for (int col = 0; col < n_col; col++)
            {
                int node_idx = row * n_col + col;
                passing_points_id.push_back(node_idx);
                passing_points.push_back(deformed_states.template segment<dim>(node_idx * dim));
            }

            getXY(row, 0, x, y);

            TV from = TV(x0, y, 0);
            TV to = TV(x1, y, 0);
        
            addAStraightRod(from, to, passing_points, passing_points_id, 
                sub_div, full_dof_cnt, node_cnt, rod_cnt, false);
            // rod_cnt ++;
        }
        
        for (int col = 0; col < n_col; col++)
        {
            T y0 = 0.0, y1 = 1.0 * sim.unit;
            T x, y;
            std::vector<int> passing_points_id;
            std::vector<TV> passing_points;
            getXY(0, col, x, y);
            for (int row = 0; row < n_row; row++)
            {
                int node_idx = row * n_col + col;
                passing_points_id.push_back(node_idx);
                passing_points.push_back(deformed_states.template segment<dim>(node_idx * dim));
            }
            
            TV from = TV(x, y0, 0);
            TV to = TV(x, y1, 0);

            addAStraightRod(from, to, passing_points, passing_points_id, sub_div, 
                            full_dof_cnt, node_cnt, rod_cnt, false);
            // rod_cnt ++;
        }
        

        T dv = 1.0 / n_row;
        T du = 1.0 / n_col;

        for (int row = 0; row < n_row; row++)
        {
            for (int col = 0; col < n_col; col++)
            {
                int node_idx = row * n_col + col;
                RodCrossing<T, dim>* crossing = 
                    new RodCrossing<T, dim>(node_idx, {row, n_row + col});

                crossing->sliding_ranges = { Range(0.4 * du, 0.4 * du), Range(0.4 * dv, 0.4 * dv)};

                crossing->on_rod_idx[row] = sim.Rods[row]->dof_node_location[col];
                crossing->on_rod_idx[n_row + col] = sim.Rods[n_row + col]->dof_node_location[row];
                
                if (sim.disable_sliding)
                {
                    crossing->is_fixed = true;
                    // int off = sim.Rods[row]->theta_reduced_dof_start_offset;
                    // sim.dirichlet_dof[off + sim.Rods[row]->dof_node_location[col]] = 0;
                    // sim.dirichlet_dof[off + sim.Rods[row]->dof_node_location[col] - 1] = 0;

                    // off = sim.Rods[n_row + col]->theta_reduced_dof_start_offset;
                    // sim.dirichlet_dof[off + sim.Rods[n_row + col]->dof_node_location[row]] = 0;
                    // sim.dirichlet_dof[off + sim.Rods[n_row + col]->dof_node_location[row] - 1] = 0;
                }
                sim.rod_crossings.push_back(crossing);
            }
        }    

        int dof_cnt = 0;
        markCrossingDoF(w_entry, dof_cnt);

        for (auto& rod : sim.Rods) rod->markDoF(w_entry, dof_cnt);
        
        appendThetaAndJointDoF(w_entry, full_dof_cnt, dof_cnt);
        
        sim.rest_states = deformed_states;

        
    
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());
        
        // std::cout << sim.W << std::endl;

        int cnt = 0;
        for (auto& rod : sim.Rods)
        {
            // rod->kt *= 0.1;
            rod->fixEndPointEulerian(sim.dirichlet_dof);
            // for (int i = 0; i < rod->numSeg(); i++)
            // {
            //     sim.dirichlet_dof[rod->theta_reduced_dof_start_offset + i] = 0;
            // }
            
            rod->setupBishopFrame();
            cnt++;
            Offset end0, end1;
            rod->frontOffset(end0); rod->backOffset(end1);
            if (rod->rod_id < n_row)
                sim.pbc_pairs.push_back(std::make_pair(0, std::make_pair(end0, end1)));
            else
                sim.pbc_pairs.push_back(std::make_pair(1, std::make_pair(end0, end1)));
            
            Offset a, b;
            rod->getEntryByLocation(1, a); rod->getEntryByLocation(rod->indices.size() - 2, b);
            sim.pbc_bending_pairs.push_back({end0, a, b, end1});
            sim.pbc_bending_pairs_rod_id.push_back({rod->rod_id, rod->rod_id, rod->rod_id, rod->rod_id});
        }
        
        TV b(1, 0, 0);
        TV n(0, 0, 1);

        for (auto& crossing : sim.rod_crossings)
        {
            Matrix<T, 3, 3> rb_frames = Matrix<T, 3, 3>::Identity();
            
            rb_frames = crossing->rotation_accumulated * rb_frames;
            for(int rod_idx : crossing->rods_involved)
            {
                int node_loc = crossing->on_rod_idx[rod_idx];
                auto rod = sim.Rods[rod_idx];
                TV ut = parallelTransportOrthonormalVector(n, b, rod->prev_tangents[node_loc]);
                T udt1 = signedAngle(ut, rod->reference_frame_us[node_loc], rod->prev_tangents[node_loc]);
                ut = parallelTransportOrthonormalVector(rod->reference_frame_us[node_loc - 1], rod->prev_tangents[node_loc - 1], b);
                T udt0 = signedAngle(ut, n, b);
                crossing->undeformed_twist.push_back(Vector<T, 2>(udt0, udt1)); 
                // std::cout << udt0 << " " << udt1 << std::endl;

                if constexpr (dim == 3)
                {
                    
                    T theta1 = rod->reference_angles[node_loc];
                    TV t1 = rod->prev_tangents[node_loc];
                    TV u1 = rod->reference_frame_us[node_loc];
                    TV b1 = t1.cross(u1);
                    TV m11 = u1 * std::cos(theta1) + b1 * std::sin(theta1);
                    TV m12 = -u1 * std::sin(theta1) + b1 * std::cos(theta1);
                    // std::cout << t1.transpose() << " " << u1.transpose() << std::endl;

                    T theta2 = rod->reference_angles[node_loc - 1];
                    TV t2 = rod->prev_tangents[node_loc - 1];
                    TV u2 = rod->reference_frame_us[node_loc - 1];
                    TV b2 = t2.cross(u2);
                    TV m21 = u2 * std::cos(theta2) + b2 * std::sin(theta2);
                    TV m22 = -u2 * std::sin(theta2) + b2 * std::cos(theta2);

                    // std::cout << "dot " << u1.dot(u2) <<std::endl;
                    // // std::cout << t2.transpose() << std::endl;
                    // std::cout << "tangent cross " << m11.cross(rb_frames.col(0)).norm() << " " << m21.cross(rb_frames.col(0)).norm() <<std::endl;
                    // std::cout << "normal cross " << m12.cross(rb_frames.col(2)).norm() << " " << m22.cross(rb_frames.col(2)).norm() <<std::endl;
                    // std::cout << std::endl;
                }
            }
        }
        // std::cout<< sim.addJointBendingAndTwistingEnergy() << std::endl;
        // sim.dirichlet_dof[sim.Rods[1]->reduced_map[sim.Rods[1]->offset_map[4][0]]] = 0;
        // sim.dirichlet_dof[sim.Rods[1]->reduced_map[sim.Rods[1]->offset_map[4][1]]] = 0;
        // sim.dirichlet_dof[sim.Rods[1]->reduced_map[sim.Rods[1]->offset_map[4][2]]] = 0.2 * sim.unit;

        // VectorXT dq(sim.W.cols());
        // std::ifstream in("/home/yueli/Documents/ETH/WuKong/eigen_vector.txt");
        // double value;
        // cnt = 0;
        // while(in >> value)
        // {
        //     dq[cnt++] = value;
        // }
        // in.close();
        // sim.deformed_states = sim.rest_states + sim.W * dq;
        // sim.rest_states = sim.deformed_states;

        

        Offset end0, end1;
        sim.Rods[0]->frontOffset(end0); sim.Rods[0]->backOffset(end1);
        sim.pbc_pairs_reference[0] = std::make_pair(std::make_pair(end0, end1), std::make_pair(0, 0));
        
        sim.dirichlet_dof[sim.Rods[0]->reduced_map[end0[0]]] = 0;
        sim.dirichlet_dof[sim.Rods[0]->reduced_map[end0[1]]] = 0;
        sim.dirichlet_dof[sim.Rods[0]->reduced_map[end0[2]]] = 0;

        sim.Rods[n_row]->frontOffset(end0); sim.Rods[n_row]->backOffset(end1);
        sim.pbc_pairs_reference[1] = std::make_pair(std::make_pair(end0, end1), std::make_pair(n_row, n_row));


        if (sim.disable_sliding)
        {
            sim.fixCrossing();
        }
        else
        {
            for (auto& crossing : sim.rod_crossings)
            {
                for (int d = 0; d < dim; d++)
                {   
                    sim.dirichlet_dof[crossing->reduced_dof_offset + d] = 0;    
                }
            }
        }
        
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::buildOneCrossScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        int sub_div_2 = sub_div / 2;
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();
        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;
        std::unordered_map<int, Offset> offset_map;
        
        TV from(0.0, 0.5, 0.0);
        TV to(1.0, 0.5, 0.0);
        from *= sim.unit; to *= sim.unit;

        TV center = TV(0.5, 0.5, 0.0) * sim.unit;
        int center_id = 0;
        deformed_states.resize(dim);
        deformed_states.template segment<dim>(full_dof_cnt) = center;
        offset_map[node_cnt] = Offset::Zero();
        for (int d = 0; d < dim; d++) offset_map[node_cnt][d] = full_dof_cnt++;
        node_cnt++;
        auto center_offset = offset_map[center_id];

        std::vector<TV> points_on_curve;
        std::vector<int> rod0;
        std::vector<int> key_points_location_rod0;

        addStraightYarnCrossNPoints(from, to, {center}, {0}, sub_div, points_on_curve, rod0, key_points_location_rod0, node_cnt);

        deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));

        Rod<T, dim>* r0 = new Rod<T, dim>(deformed_states, sim.rest_states, 0, false, ROD_A, ROD_B);

        for (int i = 0; i < points_on_curve.size(); i++)
        {
            offset_map[node_cnt] = Offset::Zero();
            
            //push Lagrangian DoF
            
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            
            for (int d = 0; d < dim; d++)
            {
                offset_map[node_cnt][d] = full_dof_cnt++;    
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
            offset_map[node_cnt][dim] = full_dof_cnt++;
            node_cnt++;
        }
        deformed_states.conservativeResize(full_dof_cnt + 1);
        deformed_states[full_dof_cnt] = (center - from).norm() / (to - from).norm();
        offset_map[center_id][dim] = full_dof_cnt++;

        r0->offset_map = offset_map;
        r0->indices = rod0;

        Vector<T, dim + 1> q0, q1;
        r0->frontDoF(q0); r0->backDoF(q1);
        r0->rest_state = new LineCurvature<T, dim>(q0, q1);
        
        r0->dof_node_location = key_points_location_rod0;
        sim.Rods.push_back(r0);

        offset_map.clear();
        
        TV rod1_from(0.5, 0.0, 0.0);
        TV rod1_to(0.5, 1.0, 0.0);
        rod1_from *= sim.unit; rod1_to *= sim.unit;

        points_on_curve.clear();
        points_on_curve.resize(0);
        std::vector<int> rod1;
        std::vector<int> key_points_location_rod1;

        addStraightYarnCrossNPoints(rod1_from, rod1_to, {center}, {0}, sub_div, points_on_curve, rod1, key_points_location_rod1, node_cnt);

        deformed_states.conservativeResize(full_dof_cnt + (points_on_curve.size()) * (dim + 1));

        Rod<T, dim>* r1 = new Rod<T, dim>(deformed_states, sim.rest_states, 1, false, ROD_A, ROD_B);
        for (int i = 0; i < points_on_curve.size(); i++)
        {
            offset_map[node_cnt] = Offset::Zero();
            //push Lagrangian DoF
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            // std::cout << points_on_curve[i].transpose() << std::endl;
            for (int d = 0; d < dim; d++)
            {
                offset_map[node_cnt][d] = full_dof_cnt++;    
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_on_curve[i] - rod1_from).norm() / (rod1_to - rod1_from).norm();
            offset_map[node_cnt][dim] = full_dof_cnt++;
            node_cnt++;
        }

        deformed_states.conservativeResize(full_dof_cnt + 1);

        deformed_states[full_dof_cnt] = (center - rod1_from).norm() / (rod1_to - rod1_from).norm();
        offset_map[center_id] = Offset::Zero();
        offset_map[center_id].template segment<dim>(0) = center_offset.template segment<dim>(0);
        offset_map[center_id][dim] = full_dof_cnt++;

        r1->offset_map = offset_map;
        r1->indices = rod1;

        r1->frontDoF(q0); r1->backDoF(q1);
        r1->rest_state = new LineCurvature<T, dim>(q0, q1);
        
        r1->dof_node_location = key_points_location_rod1;
        sim.Rods.push_back(r1);

        RodCrossing<T, dim>* rc0 = new RodCrossing<T, dim>(0, {0, 1});
        rc0->sliding_ranges = { Range(0.2, 0.2), Range(0.2, 0.2)};
        rc0->on_rod_idx[0] = key_points_location_rod0[0];
        rc0->on_rod_idx[1] =  key_points_location_rod1[0];
        sim.rod_crossings.push_back(rc0);

        

        int dof_cnt = 0;
        markCrossingDoF(w_entry, dof_cnt);
        r0->markDoF(w_entry, dof_cnt);
        r1->markDoF(w_entry, dof_cnt);

        r0->theta_dof_start_offset = full_dof_cnt;
        r0->theta_reduced_dof_start_offset = dof_cnt;        
        int theta_reduced_dof_offset0 = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + r0->indices.size() - 1);
        for (int i = 0; i < r0->indices.size() - 1; i++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }   
        deformed_states.template segment(r0->theta_dof_start_offset, 
            r0->indices.size() - 1).setZero();

        r1->theta_dof_start_offset = full_dof_cnt;
        
        int theta_reduced_dof_offset1 = dof_cnt;
        r1->theta_reduced_dof_start_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + r1->indices.size() - 1);
        for (int i = 0; i < r1->indices.size() - 1; i++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }   
        deformed_states.template segment(r1->theta_dof_start_offset, 
            r1->indices.size() - 1).setZero();

        deformed_states.conservativeResize(full_dof_cnt + sim.rod_crossings.size() * dim);
        deformed_states.template segment(full_dof_cnt, sim.rod_crossings.size() * dim).setZero();

        for (auto& crossing : sim.rod_crossings)
        {
            crossing->dof_offset = full_dof_cnt;
            crossing->reduced_dof_offset = dof_cnt;
            for (int d = 0; d < dim; d++)
            {
                w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
            }
        }
        
        sim.rest_states = sim.deformed_states;
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        // std::cout << "r0->theta_dof_start_offset " << r0->theta_dof_start_offset << " sim.W.cols() " << sim.W.cols() << std::endl;
        

        Offset ob, of;
        r0->backOffsetReduced(ob);
        r0->frontOffsetReduced(of);

        // std::cout << ob.transpose() << " " << of.transpose() << std::endl;
        r0->fixEndPointEulerian(sim.dirichlet_dof);
        r1->fixEndPointEulerian(sim.dirichlet_dof);

        // r1->fixEndPointLagrangian(sim.dirichlet_dof);

        // sim.fixCrossing();

        sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        sim.dirichlet_dof[ob[1]] = 0.3 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0;
        sim.dirichlet_dof[ob[2]] = 0.3 * sim.unit;


        sim.dirichlet_dof[theta_reduced_dof_offset0] = 0;
        sim.dirichlet_dof[theta_reduced_dof_offset1] = 0;

        Offset ob1, of1;
        r1->backOffsetReduced(ob1);
        r1->frontOffsetReduced(of1);


        sim.dirichlet_dof[ob1[0]] = 0.15 * sim.unit;
        sim.dirichlet_dof[ob1[1]] = -0.2 * sim.unit;
        sim.dirichlet_dof[ob1[2]] = -0.1 * sim.unit;

        for (int d = 0; d < dim; d++)
        {
            sim.dirichlet_dof[of[d]] = 0;
            sim.dirichlet_dof[ob1[d]] = 0;
            sim.dirichlet_dof[of1[d]] = 0;
        }

        sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.0 * sim.unit;
        sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.0 * sim.unit;
        sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.0 * sim.unit;
        
        for (auto& rod : sim.Rods)
        {
            rod->setupBishopFrame();
        }
        
    }
}
template<class T, int dim>
void UnitPatch<T, dim>::build3DtestScene(int sub_div)
{
    if constexpr (dim == 3)
    {
        int sub_div_2 = sub_div / 2;
        auto unit_yarn_map = sim.yarn_map;
        sim.yarn_map.clear();
        sim.add_rotation_penalty = false;
        sim.add_pbc_bending = false;
        sim.new_frame_work = true;

        clearSimData();
        std::vector<Eigen::Triplet<T>> w_entry;
        int full_dof_cnt = 0;
        int node_cnt = 0;
        
        TV from(0.0, 0.5, 0.0);
        TV to(1.0, 0.5, 0.0);
        from *= sim.unit; to *= sim.unit;

        std::vector<TV> points_on_curve;
        std::vector<int> rod0;
        std::vector<int> dummy;

        addStraightYarnCrossNPoints(from, to, {}, {}, sub_div, points_on_curve, rod0, dummy, 0);

        // std::cout << points_on_curve.size() << " " << rod0.size() << std::endl;

        deformed_states.resize((points_on_curve.size()) * (dim + 1));

        Rod<T, dim>* r0 = new Rod<T, dim>(deformed_states, sim.rest_states, 0, false, ROD_A, ROD_B);

        std::unordered_map<int, Offset> offset_map;
        std::vector<int> node_index_list;

        std::vector<T> data_points_discrete_arc_length;
        
        for (int i = 0; i < points_on_curve.size(); i++)
        {
            offset_map[i] = Offset::Zero();
            node_cnt++;
            node_index_list.push_back(i);
            //push Lagrangian DoF
            
            deformed_states.template segment<dim>(full_dof_cnt) = points_on_curve[i];
            // std::cout << points_on_curve[i].transpose() << std::endl;
            for (int d = 0; d < dim; d++)
            {
                offset_map[i][d] = full_dof_cnt++;    
            }
            // push Eulerian DoF
            deformed_states[full_dof_cnt] = (points_on_curve[i] - from).norm() / (to - from).norm();
            
            offset_map[i][dim] = full_dof_cnt++;
            
        }

        r0->offset_map = offset_map;
        r0->indices = node_index_list;
        // for (int idx : node_index_list)
        //     std::cout << idx << " ";
        // std::cout << std::endl;
        Vector<T, dim + 1> q0, q1;
        r0->frontDoF(q0); r0->backDoF(q1);
        

        r0->rest_state = new LineCurvature<T, dim>(q0, q1);
        
        r0->dof_node_location = {};
        sim.Rods.push_back(r0);

        int dof_cnt = 0;
        
        r0->markDoF(w_entry, dof_cnt);
        r0->theta_dof_start_offset = full_dof_cnt;
        
        int theta_reduced_dof_offset = dof_cnt;
        deformed_states.conservativeResize(full_dof_cnt + r0->indices.size() - 1);
        for (int i = 0; i < r0->indices.size() - 1; i++)
        {
            w_entry.push_back(Entry(full_dof_cnt++, dof_cnt++, 1.0));
        }   
        deformed_states.template segment(r0->theta_dof_start_offset, 
            r0->indices.size() - 1).setZero();
        
        sim.rest_states = sim.deformed_states;
        sim.W = StiffnessMatrix(full_dof_cnt, dof_cnt);
        sim.W.setFromTriplets(w_entry.begin(), w_entry.end());

        // std::cout << "r0->theta_dof_start_offset " << r0->theta_dof_start_offset << " sim.W.cols() " << sim.W.cols() << std::endl;
        

        Offset ob, of;
        r0->backOffsetReduced(ob);
        r0->frontOffsetReduced(of);

        // std::cout << ob.transpose() << " " << of.transpose() << std::endl;
        r0->fixEndPointEulerian(sim.dirichlet_dof);
        

        sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        sim.dirichlet_dof[ob[1]] = 0.3 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0;


        for (int i = theta_reduced_dof_offset; i < dof_cnt; i++)
        {
            sim.dirichlet_dof[i] = 0;
            break;
            // sim.dirichlet_dof[i] = T(i) * M_PI / 4;
        }

        // sim.dirichlet_dof[dof_cnt-1] = M_PI / 2.0;
        // sim.dirichlet_dof[dof_cnt-1] = M_PI;
        // sim.dirichlet_dof[dof_cnt-1] = 0;

        
        // sim.dirichlet_dof[ob[0]] = -0.3 * sim.unit;
        // sim.dirichlet_dof[ob[1]] = 0.1 * sim.unit;
        // sim.dirichlet_dof[ob[2]] = 0.0 * sim.unit;

        for (int d = 0; d < dim; d++)
        {
            sim.dirichlet_dof[of[d]] = 0;
        }

        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][0]]] = 0.0 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][1]]] = 0.0 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[1][2]]] = 0.0 * sim.unit;


        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[node_cnt-2][0]]] = -0.19 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[node_cnt-2][1]]] = 0.18 * sim.unit;
        // sim.dirichlet_dof[r0->reduced_map[r0->offset_map[node_cnt-2][2]]] = 0.18 * sim.unit;
        
        for (auto& rod : sim.Rods)
        {
            rod->setupBishopFrame();
        }
        
        // std::ifstream in("./testdq.txt");
        // for(int i =0; i < deformed_states.rows(); i++)
        //     in>> deformed_states[i];
        
        // in.close();

        
        // VectorXT dq = sim.W.transpose() * deformed_states;
        // sim.testGradient(dq);
        // sim.testHessian(dq);

        // for (auto& it : sim.dirichlet_dof)
        //     std::cout << it.first << " " << it.second << std::endl;
        // std::cout << deformed_states << std::endl;
        // std::cout << sim.W << std::endl;
        // std::cout << sim.W.rows() << " " << sim.W.cols() << " " << deformed_states.rows() << " " << r0->theta_dof_start_offset<< std::endl;
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::markCrossingDoF(std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt)
{
    for (auto& crossing : sim.rod_crossings)
    {
        int node_idx = crossing->node_idx;
        // std::cout << "node " << node_idx << std::endl;
        std::vector<int> rods_involved = crossing->rods_involved;

        Offset entry_rod0; 
        sim.Rods[rods_involved.front()]->getEntry(node_idx, entry_rod0);

        // push Lagrangian dof first
        for (int d = 0; d < dim; d++)
        {
            
            for (int rod_idx : rods_involved)
            {
                // std::cout << "rods involved " << rod_idx << std::endl;
                // if (node_idx == 21)
                //     std::cout << "rods involved " << rod_idx << std::endl;
                sim.Rods[rod_idx]->reduced_map[entry_rod0[d]] = dof_cnt;
            }    
            w_entry.push_back(Entry(entry_rod0[d], dof_cnt++, 1.0));
        }
        
        // push Eulerian dof for all rods
        for (int rod_idx : rods_involved)
        {
            // std::cout << "dim on rod " <<  rod_idx << std::endl;
            sim.Rods[rod_idx]->getEntry(node_idx, entry_rod0);
            // std::cout << "dim dof on rod " <<  entry_rod0[dim] << std::endl;
            sim.Rods[rod_idx]->reduced_map[entry_rod0[dim]] = dof_cnt;
            w_entry.push_back(Entry(entry_rod0[dim], dof_cnt++, 1.0));
        }
        // std::getchar();
        
    }
}

template<class T, int dim>
void UnitPatch<T, dim>::clearSimData()
{
    sim.kc = 1e8;
    sim.add_pbc = true;

    if(sim.disable_sliding)
    {
        sim.add_shearing = true;
        sim.add_eularian_reg = false;
        sim.k_pbc = 1e8;
        sim.k_strain = 1e8;
    }
    else
    {
        sim.add_shearing = false;
        sim.add_eularian_reg = true;
        sim.ke = 1e-4;    
        sim.k_yc = 1e8;
    }
    sim.k_pbc = 1e4;
    sim.k_strain = 1e7;
    sim.kr = 1e3;
    
    // sim.pbc_ref_unique.clear();
    // sim.dirichlet_data.clear();
    // sim.pbc_ref.clear();
    // sim.pbc_bending_pairs.clear();
    sim.yarns.clear();
}

// assuming passing points sorted long from to to direction
template<class T, int dim>
void UnitPatch<T, dim>::addStraightYarnCrossNPoints(const TV& from, const TV& to,
    const std::vector<TV>& passing_points, 
    const std::vector<int>& passing_points_id, int sub_div,
    std::vector<TV>& sub_points, std::vector<int>& node_idx, 
    std::vector<int>& key_points_location, 
    int start, bool pbc)
{
    
    int cnt = 1;
    if(passing_points.size())
    {
        if ((from - passing_points[0]).norm() < 1e-6 )
        {
            node_idx.push_back(passing_points_id[0]);
            cnt = 0;
        }
        else
        {
            node_idx.push_back(start);
            sub_points.push_back(from);
        }
    }
    else
    {
        node_idx.push_back(start);
        sub_points.push_back(from);
    }
    
    T length_yarn = (to - from).norm();
    TV length_vec = (to - from).normalized();
    
    TV loop_point = from;
    TV loop_left = from;
    for (int i = 0; i < passing_points.size(); i++)
    {
        if ((from - passing_points[i]).norm() < 1e-6 )
        {
            key_points_location.push_back(0);
            continue;
        }
        T fraction = (passing_points[i] - loop_point).norm() / length_yarn;
        int n_sub_nodes = std::ceil(fraction * sub_div);
        T length_sub = (passing_points[i] - loop_point).norm() / T(n_sub_nodes);
        for (int j = 0; j < n_sub_nodes - 1; j++)
        {
            sub_points.push_back(loop_left + length_sub * length_vec);
            loop_left = sub_points.back();
            node_idx.push_back(start + cnt);
            cnt++;
        }
        node_idx.push_back(passing_points_id[i]);
        key_points_location.push_back(cnt + i);
        loop_point = passing_points[i];
        loop_left = passing_points[i];
    }
    if (passing_points.size())
    {
        if ((passing_points.back() - to).norm() < 1e-6)
        {
            
            return;
        }
    }
    T fraction;
    int n_sub_nodes;
    T length_sub;
    if( passing_points.size() )
    {
        fraction = (to - passing_points.back()).norm() / length_yarn;
        n_sub_nodes = std::ceil(fraction * sub_div);
        length_sub = (to - passing_points.back()).norm() / T(n_sub_nodes);
    }
    else
    {
        n_sub_nodes = sub_div + 1;
        length_sub = (to - from).norm() / T(sub_div);
    }
    for (int j = 0; j < n_sub_nodes - 1; j++)
    {
        if (j == 0)
        {
            if(passing_points.size())
            {
                sub_points.push_back(passing_points.back() + length_sub * length_vec);
                loop_left = sub_points.back();
            }
        }
        else
        {
            sub_points.push_back(loop_left + length_sub * length_vec);
            loop_left = sub_points.back();
        }
        if(passing_points.size() == 0 && j == 0)
            continue;
        node_idx.push_back(start + cnt);
        cnt++;
    }
    node_idx.push_back(start + cnt);
    sub_points.push_back(to);
}


template class UnitPatch<double, 3>;
template class UnitPatch<double, 2>;   