#include <unordered_set>
#include <fstream>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <Eigen/PardisoSupport>

#include <ipc/ipc.hpp>

#include "../include/VertexModel.h"
#include "../include/autodiff/VertexModelEnergy.h"



void VertexModel::approximateMembraneThickness()
{
    T radii_max = -1e6;
    for (int i = 0; i < basal_vtx_start; i++)
    {
        radii_max = std::max(radii_max, (deformed.segment<3>(i * 3) - mesh_centroid).norm());
    }
    if (sphere_bound_penalty)
        Rc = radii_max;
    else
        Rc = 1.01 * radii_max;
        
}

void VertexModel::computeLinearModes()
{
    int nmodes = 15;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;
    buildSystemMatrix(u, K);
    
    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(K);

        //0 cannot cannot be used as a shift
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values << std::endl;
            std::ofstream out("cell_eigen_vectors.txt");
            out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
            for (int i = 0; i < eigen_vectors.cols(); i++)
                out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
            out << std::endl;
            for (int i = 0; i < eigen_vectors.rows(); i++)
            {
                // for (int j = 0; j < eigen_vectors.cols(); j++)
                for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                    out << eigen_vectors(i, j) << " ";
                out << std::endl;
            }       
            out << std::endl;
            out.close();
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
}

bool VertexModel::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

    Eigen::PardisoLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    
    T alpha = 10e-6;
    solver.analyzePattern(K);
    for (int i = 0; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            K = H + alpha * I;        
            alpha *= 10;
            continue;
        }
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        // VectorXT d_vector = solver.vectorD();
        int num_negative_eigen_values = 0;


        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (positive_definte && search_dir_correct_sign && solve_success)
            return true;
        else
        {
            K = H + alpha * I;        
            alpha *= 10;
        }
    }
    return false;
}


void VertexModel::computeCellCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
    {
        centroid += deformed.segment<3>(vtx_idx * 3);
        centroid += deformed.segment<3>((vtx_idx + basal_vtx_start) * 3);
    }
    centroid /= T(face_vtx_list.size() * 2);
}

void VertexModel::computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
        centroid += deformed.segment<3>(vtx_idx * 3);

    centroid /= T(face_vtx_list.size());
}

void VertexModel::computeHexPrismVolumeFromTet(const Vector<T, 36>& prism_vertices, 
    T& volume, int iter)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    std::vector<TV> vtx(12);
    for (int i = 0; i < 6; i++)
    {
        vtx[i] = prism_vertices.segment<3>((11 - i) * 3);
        vtx[i + 6] = prism_vertices.segment<3>((5 - i) * 3);
    }

    Vector<T, 12> tet_vol;
    
    tet_vol[0] = computeTetVolume(vtx[9], vtx[2], vtx[10], vtx[3]);
    tet_vol[1] = computeTetVolume(vtx[2], vtx[10], vtx[3], vtx[4]);
    tet_vol[2] = computeTetVolume(vtx[1], vtx[11], vtx[0], vtx[7]);
    tet_vol[3] = computeTetVolume(vtx[9], vtx[2], vtx[8], vtx[10]);
    tet_vol[4] = computeTetVolume(vtx[10], vtx[1], vtx[7], vtx[11]);
    tet_vol[5] = computeTetVolume(vtx[0], vtx[11], vtx[6], vtx[7]);
    tet_vol[6] = computeTetVolume(vtx[1], vtx[11], vtx[4], vtx[5]);
    tet_vol[7] = computeTetVolume(vtx[2], vtx[10], vtx[1], vtx[8]);
    tet_vol[8] = computeTetVolume(vtx[2], vtx[10], vtx[4], vtx[1]);
    tet_vol[9] = computeTetVolume(vtx[1], vtx[11], vtx[5], vtx[0]);
    tet_vol[10] = computeTetVolume(vtx[10], vtx[1], vtx[8], vtx[7]);
    tet_vol[11] = computeTetVolume(vtx[10], vtx[1], vtx[11], vtx[4]);

    auto saveTetObjs = [&](const std::vector<TV>& tets, int idx, int iter)
    {
        TV tet_center = TV::Zero();
        for (const TV& vtx : tets)
        {
            tet_center += vtx;
        }
        tet_center *= 0.25;

        TV shift = 0.5 * tet_center;
        
        std::ofstream out("output/cells/iter_" +std::to_string(iter)+ "_tet" + std::to_string(idx) + ".obj");
        for (const TV& vtx : tets)
            out << "v " << (vtx + shift).transpose() << std::endl;
        out << "f 3 2 1" << std::endl;
        out << "f 4 3 1" << std::endl;
        out << "f 2 4 1" << std::endl;
        out << "f 3 4 2" << std::endl;
        out.close();
    };

    saveTetObjs({vtx[9], vtx[2], vtx[10], vtx[3]}, 0, iter);
    saveTetObjs({vtx[2], vtx[10], vtx[3], vtx[4]}, 1, iter);
    saveTetObjs({vtx[1], vtx[11], vtx[0], vtx[7]}, 2, iter);
    saveTetObjs({vtx[9], vtx[2], vtx[8], vtx[10]}, 3, iter);
    saveTetObjs({vtx[10], vtx[1], vtx[7], vtx[11]}, 4, iter);
    saveTetObjs({vtx[0], vtx[11], vtx[6], vtx[7]}, 5, iter);
    saveTetObjs({vtx[1], vtx[11], vtx[4], vtx[5]}, 6, iter);
    saveTetObjs({vtx[2], vtx[10], vtx[1], vtx[8]}, 7, iter);
    saveTetObjs({vtx[2], vtx[10], vtx[4], vtx[1]}, 8, iter);
    saveTetObjs({vtx[1], vtx[11], vtx[5], vtx[0]}, 9, iter);
    saveTetObjs({vtx[10], vtx[1], vtx[8], vtx[7]}, 10, iter);
    saveTetObjs({vtx[10], vtx[1], vtx[11], vtx[4]}, 11, iter);
    
    std::cout << "tets volume : " << tet_vol.transpose() << std::endl;
    volume = tet_vol.sum();
}

void VertexModel::computePentaPrismVolumeFromTet(const Vector<T, 30>& prism_vertices,
 T& volume)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    // TV v0 = prism_vertices.segment<3>(5 * 3);
    // TV v1 = prism_vertices.segment<3>(6 * 3);
    // TV v2 = prism_vertices.segment<3>(7 * 3);
    // TV v3 = prism_vertices.segment<3>(8 * 3);
    // TV v4 = prism_vertices.segment<3>(9 * 3);

    // TV v5 = prism_vertices.segment<3>(0 * 3);
    // TV v6 = prism_vertices.segment<3>(1 * 3);
    // TV v7 = prism_vertices.segment<3>(2 * 3);
    // TV v8 = prism_vertices.segment<3>(3 * 3);
    // TV v9 = prism_vertices.segment<3>(4 * 3);

    TV v0 = prism_vertices.segment<3>(9 * 3);
    TV v1 = prism_vertices.segment<3>(8 * 3);
    TV v2 = prism_vertices.segment<3>(7 * 3);
    TV v3 = prism_vertices.segment<3>(6 * 3);
    TV v4 = prism_vertices.segment<3>(5 * 3);

    TV v5 = prism_vertices.segment<3>(4 * 3);
    TV v6 = prism_vertices.segment<3>(3 * 3);
    TV v7 = prism_vertices.segment<3>(2 * 3);
    TV v8 = prism_vertices.segment<3>(1 * 3);
    TV v9 = prism_vertices.segment<3>(0 * 3);

    // TV v0 = prism_vertices.segment<3>(0 * 3);
    // TV v1 = prism_vertices.segment<3>(1 * 3);
    // TV v2 = prism_vertices.segment<3>(2 * 3);
    // TV v3 = prism_vertices.segment<3>(3 * 3);
    // TV v4 = prism_vertices.segment<3>(4 * 3);

    // TV v5 = prism_vertices.segment<3>(5 * 3);
    // TV v6 = prism_vertices.segment<3>(6 * 3);
    // TV v7 = prism_vertices.segment<3>(7 * 3);
    // TV v8 = prism_vertices.segment<3>(8 * 3);
    // TV v9 = prism_vertices.segment<3>(9 * 3);

    std::cout << v0.transpose() << std::endl;
    std::cout << v1.transpose() << std::endl;
    std::cout << v2.transpose() << std::endl;
    std::cout << v3.transpose() << std::endl;
    std::cout << v4.transpose() << std::endl;

    std::getchar();


    Vector<T, 9> tet_vol;

    tet_vol[0] = computeTetVolume(v3, v8, v9, v0);
    tet_vol[1] = computeTetVolume(v3, v9, v4, v0);
    tet_vol[2] = computeTetVolume(v7, v0, v2, v1);
    tet_vol[3] = computeTetVolume(v9, v8, v5, v0);
    tet_vol[4] = computeTetVolume(v6, v0, v7, v1);
    tet_vol[5] = computeTetVolume(v5, v0, v7, v6);
    tet_vol[6] = computeTetVolume(v5, v8, v7, v0);
    tet_vol[7] = computeTetVolume(v8, v0, v2, v7);
    tet_vol[8] = computeTetVolume(v8, v3, v2, v0);

    auto saveTetObjs = [&](const std::vector<TV>& tets, int idx)
    {
        TV tet_center = TV::Zero();
        for (const TV& vtx : tets)
        {
            tet_center += vtx;
        }
        tet_center *= 0.25;

        TV shift = 0.5 * tet_center;
        
        std::ofstream out("tet" + std::to_string(idx) + ".obj");
        for (const TV& vtx : tets)
            out << "v " << (vtx + shift).transpose() << std::endl;
        out << "f 3 2 1" << std::endl;
        out << "f 4 3 1" << std::endl;
        out << "f 2 4 1" << std::endl;
        out << "f 3 4 2" << std::endl;
        out.close();
    };

    // saveTetObjs({v8, v3, v9, v0}, 0);
    // saveTetObjs({v9, v3, v4, v0}, 1);
    // saveTetObjs({v0, v7, v2, v1}, 2);
    // saveTetObjs({v8, v9, v5, v0}, 3);
    // saveTetObjs({v0, v6, v7, v1}, 4);
    // saveTetObjs({v0, v5, v7, v6}, 5);
    // saveTetObjs({v8, v5, v7, v0}, 6);
    // saveTetObjs({v0, v8, v2, v7}, 7);
    // saveTetObjs({v3, v8, v2, v0}, 8);

    std::cout << "tets volume : " << tet_vol.transpose() << std::endl;
    volume = tet_vol.sum();
}

void VertexModel::computeCubeVolumeFromTet(const Vector<T, 24>& prism_vertices, T& volume)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    
    TV v0 = prism_vertices.segment<3>(4 * 3);
    TV v1 = prism_vertices.segment<3>(5 * 3);
    TV v2 = prism_vertices.segment<3>(7 * 3);
    TV v3 = prism_vertices.segment<3>(6 * 3);
    TV v4 = prism_vertices.segment<3>(0 * 3);
    TV v5 = prism_vertices.segment<3>(1 * 3);
    TV v6 = prism_vertices.segment<3>(3 * 3);
    TV v7 = prism_vertices.segment<3>(2 * 3);

	
    Vector<T, 6> tet_vol;
    tet_vol[0] = computeTetVolume(v2, v4, v6, v5);
    tet_vol[1] = computeTetVolume(v7, v2, v6, v5);
    tet_vol[2] = computeTetVolume(v3, v2, v7, v5);
    tet_vol[3] = computeTetVolume(v4, v2, v0, v5);
    tet_vol[4] = computeTetVolume(v1, v0, v2, v5);
    tet_vol[5] = computeTetVolume(v3, v1, v2, v5);

    auto saveTetObjs = [&](const std::vector<TV>& tets, int idx)
    {
        TV tet_center = TV::Zero();
        for (const TV& vtx : tets)
        {
            tet_center += vtx;
        }
        tet_center *= 0.25;

        TV shift = 0.5 * tet_center;
        
        std::ofstream out("tet" + std::to_string(idx) + ".obj");
        for (const TV& vtx : tets)
            out << "v " << (vtx + shift).transpose() << std::endl;
        out << "f 3 2 1" << std::endl;
        out << "f 4 3 1" << std::endl;
        out << "f 2 4 1" << std::endl;
        out << "f 3 4 2" << std::endl;
        out.close();
    };

    // saveTetObjs({v2, v4, v6, v5}, 0);
    // saveTetObjs({v7, v2, v6, v5}, 1);
    // saveTetObjs({v3, v2, v7, v5}, 2);
    // saveTetObjs({v4, v2, v0, v5}, 3);
    // saveTetObjs({v1, v0, v2, v5}, 4);
    // saveTetObjs({v3, v1, v2, v5}, 5);

    std::cout << "tet vol " << tet_vol.transpose() << std::endl;

    volume = tet_vol.sum();
}

void VertexModel::saveHexTetsStep(int iteration)
{
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            
            positionsFromIndices(positions, cell_vtx_list);
            T tet_vol;
            computeHexPrismVolumeFromTet(positions, tet_vol, iteration);
            
        }
    });
}

void VertexModel::computeCubeVolumeCentroid(const Vector<T, 24>& prism_vertices, T& volume)
{
    auto computeTetVolume = [&](const TV& a, const TV& b, const TV& c, const TV& d)
    {
        T tet_vol = 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
        std::cout << tet_vol << std::endl;
        return tet_vol;
    };

    TV apical_centroid = TV::Zero(), basal_centroid = TV::Zero(), cell_centroid = TV::Zero();
    for (int i = 0; i < 4; i++)
    {
        apical_centroid += prism_vertices.segment<3>(i * 3);
        basal_centroid += prism_vertices.segment<3>((i + 4) * 3);
    }
    cell_centroid = (apical_centroid + basal_centroid) / T(8);
    apical_centroid /= T(4);
    basal_centroid /= T(4);

    volume = 0.0;

    for (int i = 0; i < 4; i++)
    {
        
        int j = (i + 1) % 4;
        TV r0 = prism_vertices.segment<3>(i * 3);
        TV r1 = prism_vertices.segment<3>(j * 3);
        volume -= computeTetVolume(apical_centroid, r1, r0, cell_centroid);

        TV r2 = prism_vertices.segment<3>((i + 4) * 3);
        TV r3 = prism_vertices.segment<3>((j + 4) * 3);
        volume += computeTetVolume(basal_centroid, r3, r2, cell_centroid);

        TV lateral_centroid = T(0.25) * (r0 + r1 + r2 + r3);
        volume += computeTetVolume(lateral_centroid, r1, r0, cell_centroid);
        volume += computeTetVolume(lateral_centroid, r3, r1, cell_centroid);
        volume += computeTetVolume(lateral_centroid, r2, r3, cell_centroid);
        volume += computeTetVolume(lateral_centroid, r0, r2, cell_centroid);
    }
    
}

void VertexModel::computeVolumeAllCells(VectorXT& cell_volume_list)
{
    // each apical face corresponds to one cell
    cell_volume_list = VectorXT::Ones(basal_face_start);

    // use apical face to iterate other faces within this cell for now
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);
            
            // for (int idx : cell_vtx_list)
            //     std::cout << idx << " ";
            // std::cout << std::endl;

            positionsFromIndices(positions, cell_vtx_list);
            
            if (face_vtx_list.size() == 4)
            {
                if (use_cell_centroid)
                    computeVolume4Points(positions, cell_volume_list[face_idx]);
                else
                    computeQuadBasePrismVolume(positions, cell_volume_list[face_idx]);
                // T tet_vol;
                
                // computeCubeVolumeFromTet(positions, tet_vol);
                // // // computeCubeVolumeCentroid(positions, tet_vol);
                // std::cout << tet_vol << std::endl;
                // std::getchar();
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_cell_centroid)
                    computeVolume5Points(positions, cell_volume_list[face_idx]);
                else
                    computePentaBasePrismVolume(positions, cell_volume_list[face_idx]);
                
                // T tet_vol;
                // computePentaPrismVolumeFromTet(deformed, tet_vol);
                // std::cout << tet_vol << std::endl;
                // computePentaPrismVolumeFromTet(positions, tet_vol);
                // std::cout << tet_vol << std::endl;
                // std::cout << cell_volume_list[face_idx] << std::endl;
                // std::getchar();
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_cell_centroid)
                    computeVolume6Points(positions, cell_volume_list[face_idx]);
                else
                    computeHexBasePrismVolume(positions, cell_volume_list[face_idx]);
                // std::cout << cell_volume_list[face_idx] << std::endl;
                // T tet_vol, tet_ad;
                // computeVolume6Points(positions, tet_ad);
                // computeHexPrismVolumeFromTet(positions, tet_vol);
                // std::cout << "tet manual " << tet_vol << " tet ad " << tet_ad << std::endl;
                // std::getchar();
            }
        }
    });
}


T VertexModel::computeYolkVolume(bool verbose)
{
    T yolk_volume = 0.0;
    if (verbose)
        std::cout << "yolk tet volume: " << std::endl;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        
        if (face_idx < lateral_face_start && face_idx >= basal_face_start)
        {
            T cone_volume;
            if (face_vtx_list.size() == 4) 
                computeConeVolume4Points(positions, mesh_centroid, cone_volume);
                // computeQuadConeVolume(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 5) 
                computeConeVolume5Points(positions, mesh_centroid, cone_volume);
            else if (face_vtx_list.size() == 6) 
                computeConeVolume6Points(positions, mesh_centroid, cone_volume);
            else
                std::cout << "unknown polygon edge number" << __FILE__ << std::endl;
            yolk_volume += cone_volume;
            if (verbose)
                std::cout << cone_volume << " ";
        }
        
    });
    if (verbose)
        std::cout << std::endl;
    return yolk_volume;
}

T VertexModel::computeAreaEnergy(const VectorXT& _u)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    T energy = 0.0;
    use_face_centroid = true;
    // std::cout << lateral_face_start << std::endl;
    // std::cout << faces.size() << std::endl;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        // cell-wise volume preservation term
        
        // if (face_idx >= basal_face_start + 43)
        // if (face_idx >= lateral_face_start)
        if (face_idx == 196)
        {
            // std::ofstream out("bug_face.obj");
            // for (int i = 0; i < face_vtx_list.size(); i++)
            // {
            //     out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
            // }
            // TV face_centroid = TV::Zero();
            // computeFaceCentroid(face_vtx_list, face_centroid);
            // out << "v " << face_centroid.transpose() << std::endl;
            
            // out << "f 5 1 2 " << std::endl;
            // out << "f 5 2 3 " << std::endl;
            // out << "f 5 3 4 " << std::endl;
            // out << "f 5 4 1 " << std::endl;
            // out.close();
            // std::getchar();
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            if (face_vtx_list.size() == 4)
            {
                // computeArea4PointsSquaredSum(coeff, positions, area_energy);
                if (use_face_centroid)
                {
                    computeArea4PointsSquared(coeff, positions, area_energy);
                    area_energy = std::sqrt(area_energy);
                }
                else
                    computeQuadFaceAreaSquaredSum(coeff, positions, area_energy);
                // computeArea4PointsSquared(coeff, positions, area_energy);
                // computeQuadFaceArea(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_face_centroid)
                    computeArea5PointsSquared(coeff, positions, area_energy);
                else
                    computePentFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_face_centroid)
                    computeArea6PointsSquared(coeff, positions, area_energy);
                else
                    computeHexFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else
                std::cout << "unknown polygon edge case" << std::endl;
        }
        energy += area_energy;
    });
    return energy;
}

T VertexModel::computeTotalEnergy(const VectorXT& _u, bool verbose)
{
    if (verbose)
        std::cout << std::endl;
    T energy = 0.0;
    
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    T edge_length_term = 0.0, area_term = 0.0, 
        volume_term = 0.0, yolk_volume_term = 0.0,
        edge_contraction_term = 0.0, sphere_bound_term = 0.0;
    
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){    
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length = computeEdgeSquaredNorm(vi, vj);
        edge_length_term += sigma * edge_length;

    });

    if (add_contraction_term)
    {
        iterateContractingEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            T edge_length = computeEdgeSquaredNorm(vi, vj);
            edge_contraction_term += Gamma * edge_length;
        });
    }

    if (verbose)
    {
        std::cout << "\tE_edge " << edge_length_term << std::endl;
        if (add_contraction_term)
            std::cout << "\tE_edge_contract " << edge_contraction_term << std::endl;
    }
    energy += edge_length_term;
    energy += edge_contraction_term;

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);
    
    // if (verbose)
        // std::cout << "current cell volume " << current_cell_volume.transpose() << std::endl;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            volume_term += 0.5 * B * std::pow(cell_volume_init[face_idx] - current_cell_volume[face_idx], 2);
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            if (face_vtx_list.size() == 4)
            {
                // computeArea4PointsSquaredSum(coeff, positions, area_energy);
                if (use_face_centroid)
                    computeArea4PointsSquared(coeff, positions, area_energy);
                else
                    computeQuadFaceAreaSquaredSum(coeff, positions, area_energy);
                // computeArea4PointsSquared(coeff, positions, area_energy);
                // computeQuadFaceArea(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_face_centroid)
                    computeArea5PointsSquared(coeff, positions, area_energy);
                else
                    computePentFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_face_centroid)
                    computeArea6PointsSquared(coeff, positions, area_energy);
                else
                    computeHexFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else
                std::cout << "unknown polygon edge case" << std::endl;
        }
        area_term += area_energy;
    });
    if (verbose)
    {
        std::cout << "\tE_area: " << area_term << std::endl;
        std::cout << "\tE_volume: " << volume_term << std::endl;
    }

    energy += volume_term;
    energy += area_term;

    if (add_yolk_volume)
    {
        T yolk_vol_curr = computeYolkVolume();
        if (use_yolk_pressure)
        {
            yolk_volume_term += -pressure_constant * yolk_vol_curr;
        }
        else
        {
            yolk_volume_term +=  0.5 * By * std::pow(yolk_vol_curr - yolk_vol_init, 2);    
        }
    }
    if (verbose)
        std::cout << "\tE_yolk_vol " << yolk_volume_term << std::endl;

    energy += yolk_volume_term;

    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            T e = 0.0;;
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            if (sphere_bound_penalty)
            {
                if (Rk >= Rc)
                {
                    computeRadiusPenalty(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
                    sphere_bound_term += e;
                }
            }
            else
            {
                sphereBoundEnergy(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
                // std::cout << e << " Rk " << Rk << " Rc " << Rc << std::endl;
                // std::getchar();
                sphere_bound_term += e;
            }
        }
        if (verbose)
            std::cout << "\tE_inside_sphere " << sphere_bound_term << std::endl;
    }
    energy += sphere_bound_term;

    return energy;
}

T VertexModel::computeResidual(const VectorXT& _u,  VectorXT& residual, bool verbose)
{
    VectorXT residual_temp = residual;
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -sigma;
        addForceEntry<6>(residual, {e[0], e[1]}, dedx);
    });

    if (add_contraction_term)
    {
        iterateContractingEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -Gamma;
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        }); 
    }

    if (print_force_norm)
        std::cout << "\tedge length force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }
    
    
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);

            // cell-wise volume preservation term
            if (face_idx < basal_face_start)
            {
                T coeff = B * (current_cell_volume[face_idx] - cell_volume_init[face_idx]);

                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 24> dedx;
                    if (use_cell_centroid)
                        computeVolume4PointsGradient(positions, dedx);
                    else
                        computeQuadBasePrismVolumeGradient(positions, dedx);
                    dedx *= -coeff;
                    addForceEntry<24>(residual, cell_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 30> dedx;
                    if (use_cell_centroid)
                        computeVolume5PointsGradient(positions, dedx);
                    else
                        computePentaBasePrismVolumeGradient(positions, dedx);
                    dedx *= -coeff;
                    addForceEntry<30>(residual, cell_vtx_list, dedx);
                    // T tet_vol;
                    // computePentaPrismVolumeFromTet(positions, tet_vol);
                    // std::getchar();
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 36> dedx;
                    if (use_cell_centroid)
                        computeVolume6PointsGradient(positions, dedx);
                    else
                        computeHexBasePrismVolumeGradient(positions, dedx);
                    dedx *= -coeff;
                    addForceEntry<36>(residual, cell_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }
    });

    if (print_force_norm)
        std::cout << "\tcell volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // else // basal and lateral faces area term
        if (face_idx >= basal_face_start)
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Vector<T, 12> dedx;
                // computeArea4PointsGradient(coeff, positions, dedx);
                // computeQuadFaceAreaGradient(coeff, positions, dedx);
                // computeArea4PointsSquaredGradient(coeff, positions, dedx);
                // computeArea4PointsSquaredSumGradient(coeff, positions, dedx);
                if (use_face_centroid)
                    computeArea4PointsSquaredGradient(coeff, positions, dedx);
                else
                    computeQuadFaceAreaSquaredSumGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                dedx *=-1;
                addForceEntry<12>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                // computeArea5PointsGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                if (use_face_centroid)
                    computeArea5PointsSquaredGradient(coeff, positions, dedx);
                else
                    computePentFaceAreaSquaredSumGradient(coeff, positions, dedx);
                dedx *= -1.0;
                addForceEntry<15>(residual, face_vtx_list, dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                // computeArea6PointsGradient(coeff, positions, dedx);
                // dedx *= -coeff;
                // computeHexFaceAreaEnergyGradient(coeff, positions, dedx);
                if (use_face_centroid)
                    computeArea6PointsSquaredGradient(coeff, positions, dedx);
                else
                    computeHexFaceAreaSquaredSumGradient(coeff, positions, dedx);
                dedx *= -1.0;
                addForceEntry<18>(residual, face_vtx_list, dedx);
            }
            else
            {
                std::cout << "error " << __FILE__ << std::endl;
            }
        }
    });

    if (print_force_norm)
        std::cout << "\tarea force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                T coeff;
                if (use_yolk_pressure)
                    coeff = -pressure_constant;
                else
                    coeff = By * (yolk_vol_curr - yolk_vol_init);
                if (face_vtx_list.size() == 4)
                {
                    Vector<T, 12> dedx;
                    computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                    // computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<12>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Vector<T, 15> dedx;
                    computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<15>(residual, face_vtx_list, dedx);
                }
                else if (face_vtx_list.size() == 6)
                {
                    Vector<T, 18> dedx;
                    computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                    dedx *= -coeff;
                    addForceEntry<18>(residual, face_vtx_list, dedx);
                }
                else
                {
                    std::cout << "unknown polygon edge number" << std::endl;
                }
            }
        }
    });
    
    if (print_force_norm)
        std::cout << "\tyolk volume preservation force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            Vector<T, 3> dedx;
            if (sphere_bound_penalty)
            {
                T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
                if (Rk >= Rc)
                {
                    computeRadiusPenaltyGradient(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, dedx);
                    addForceEntry<3>(residual, {i}, -dedx);
                }
            }
            else
            {
                sphereBoundEnergyGradient(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, dedx);
                // std::cout << dedx.transpose() << std::endl;
                addForceEntry<3>(residual, {i}, -dedx);
            }
        }
        if(print_force_norm)
            std::cout << "\tsphere bound norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void VertexModel::positionsFromIndices(VectorXT& positions, const VtxList& indices)
{
    positions = VectorXT::Zero(indices.size() * 3);
    for (int i = 0; i < indices.size(); i++)
    {
        positions.segment<3>(i * 3) = deformed.segment<3>(indices[i] * 3);
    }
}

void VertexModel::buildSystemMatrixShermanMorrison(const VectorXT& _u, StiffnessMatrix& K, VectorXT& v)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    std::vector<Entry> entries;
    
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeSquaredNormHessian(vi, vj, hessian);
        hessian *= sigma;
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });

    if (add_contraction_term)
    {
        iterateContractingEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= Gamma;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    }

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }

    if (!use_yolk_pressure)
    {
        VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (add_yolk_volume)
            {
                if (face_idx < lateral_face_start && face_idx >= basal_face_start)
                {
                    
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    if (face_vtx_list.size() == 4)
                    {
                        Vector<T, 12> dedx;
                        computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                        // computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                        addForceEntry<12>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 5)
                    {
                        Vector<T, 15> dedx;
                        computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<15>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 6)
                    {
                        Vector<T, 18> dedx;
                        computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<18>(dVdx_full, face_vtx_list, dedx);
                    }
                    else
                    {
                        std::cout << "unknown polygon edge number" << std::endl;
                    }
                }
            }
        });

        v = dVdx_full * std::sqrt(By);
        if (!run_diff_test)
        {
            iterateDirichletDoF([&](int offset, T target)
            {
                v[offset] = 0.0;
            });
        }
    }
    else
    {
        v = VectorXT::Zero(deformed.rows());
    }
    

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);
            T V = current_cell_volume[face_idx];

            T coeff = B;
            if (face_vtx_list.size() == 4)
            {
                
                Matrix<T, 24, 24> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume4PointsHessian(positions, d2Vdx2);
                else
                    computeQuadBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 24> dVdx;
                if (use_cell_centroid)
                    computeVolume4PointsGradient(positions, dVdx);
                else
                    computeQuadBasePrismVolumeGradient(positions, dVdx);
                    
                // break it down here to avoid super long autodiff code
                Matrix<T, 24, 24> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);

                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume5PointsHessian(positions, d2Vdx2);
                else 
                    computePentaBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 30> dVdx;
                if (use_cell_centroid)
                    computeVolume5PointsGradient(positions, dVdx);
                else
                    computePentaBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 30, 30> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);
                
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume6PointsHessian(positions, d2Vdx2);
                else
                    computeHexBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 36> dVdx;
                if (use_cell_centroid)
                    computeVolume6PointsGradient(positions, dVdx);
                else
                    computeHexBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 36, 36> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);
                
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown polygon edge case" << std::endl;
            }
            // std::cout << "Cell " << face_idx << std::endl;
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                // computeArea4PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                // computeQuadFaceAreaHessian(coeff, positions, hessian);
                // computeArea4PointsSquaredHessian(coeff, positions, hessian);
                if (use_face_centroid)
                    computeArea4PointsSquaredSumHessian(coeff, positions, hessian);
                else
                    computeQuadFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                // computeArea5PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea5PointsSquaredHessian(coeff, positions, hessian);
                else
                    computePentFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                // computeArea6PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea6PointsSquaredHessian(coeff, positions, hessian);
                else
                    computeHexFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
            }
        }
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    // computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }

        
    });

    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            Matrix<T, 3, 3> hessian;
            if (sphere_bound_penalty)
            {
                T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
                if (Rk >= Rc)
                {
                    computeRadiusPenaltyHessian(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, hessian);
                    addHessianEntry<3>(entries, {i}, hessian);    
                }
            }
            else
            {
                sphereBoundEnergyHessian(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, hessian);
                addHessianEntry<3>(entries, {i}, hessian);
            }
        }
    }

        
    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::ofstream out("hessian.txt");
    // out << K;
    // out.close();
    // std::getchar();
    K.makeCompressed();
}

void VertexModel::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    std::vector<Entry> entries;
    
    // edge length term
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeSquaredNormHessian(vi, vj, hessian);
        hessian *= sigma;
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });

    if (add_contraction_term)
    {
        iterateContractingEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= Gamma;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    }

    VectorXT current_cell_volume;
    computeVolumeAllCells(current_cell_volume);

    T yolk_vol_curr = 0.0;
    if (add_yolk_volume)
    {
        yolk_vol_curr = computeYolkVolume();
    }

    if (!use_yolk_pressure)
    {
        VectorXT dVdx_full = VectorXT::Zero(deformed.rows());

        iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
        {
            if (add_yolk_volume)
            {
                if (face_idx < lateral_face_start && face_idx >= basal_face_start)
                {
                    
                    VectorXT positions;
                    positionsFromIndices(positions, face_vtx_list);
                    if (face_vtx_list.size() == 4)
                    {
                        Vector<T, 12> dedx;
                        computeConeVolume4PointsGradient(positions, mesh_centroid, dedx);
                        // computeQuadConeVolumeGradient(positions, mesh_centroid, dedx);
                        addForceEntry<12>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 5)
                    {
                        Vector<T, 15> dedx;
                        computeConeVolume5PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<15>(dVdx_full, face_vtx_list, dedx);
                    }
                    else if (face_vtx_list.size() == 6)
                    {
                        Vector<T, 18> dedx;
                        computeConeVolume6PointsGradient(positions, mesh_centroid, dedx);
                        addForceEntry<18>(dVdx_full, face_vtx_list, dedx);
                    }
                    else
                    {
                        std::cout << "unknown polygon edge number" << std::endl;
                    }
                }
            }
        });

        for (int dof_i = 0; dof_i < num_nodes; dof_i++)
        {
            for (int dof_j = 0; dof_j < num_nodes; dof_j++)
            {
                Vector<T, 6> dVdx;
                getSubVector<6>(dVdx_full, {dof_i, dof_j}, dVdx);
                TV dVdxi = dVdx.segment<3>(0);
                TV dVdxj = dVdx.segment<3>(3);
                Matrix<T, 3, 3> hessian_partial = By * dVdxi * dVdxj.transpose();
                addHessianBlock<3>(entries, {dof_i, dof_j}, hessian_partial);
            }
        }
    }
    

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            VectorXT positions;
            VtxList cell_vtx_list = face_vtx_list;
            for (int idx : face_vtx_list)
                cell_vtx_list.push_back(idx + basal_vtx_start);

            positionsFromIndices(positions, cell_vtx_list);
            T V = current_cell_volume[face_idx];

            T coeff = B;
            if (face_vtx_list.size() == 4)
            {
                
                Matrix<T, 24, 24> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume4PointsHessian(positions, d2Vdx2);
                else
                    computeQuadBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 24> dVdx;
                if (use_cell_centroid)
                    computeVolume4PointsGradient(positions, dVdx);
                else
                    computeQuadBasePrismVolumeGradient(positions, dVdx);
                    
                // break it down here to avoid super long autodiff code
                Matrix<T, 24, 24> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);

                addHessianEntry<24>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 30, 30> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume5PointsHessian(positions, d2Vdx2);
                else 
                    computePentaBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 30> dVdx;
                if (use_cell_centroid)
                    computeVolume5PointsGradient(positions, dVdx);
                else
                    computePentaBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 30, 30> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);
                
                addHessianEntry<30>(entries, cell_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 36, 36> d2Vdx2;
                if (use_cell_centroid)
                    computeVolume6PointsHessian(positions, d2Vdx2);
                else
                    computeHexBasePrismVolumeHessian(positions, d2Vdx2);

                Vector<T, 36> dVdx;
                if (use_cell_centroid)
                    computeVolume6PointsGradient(positions, dVdx);
                else
                    computeHexBasePrismVolumeGradient(positions, dVdx);
                
                // break it down here to avoid super long autodiff code
                Matrix<T, 36, 36> hessian = B * (dVdx * dVdx.transpose() + 
                    (V - cell_volume_init[face_idx]) * d2Vdx2);
                
                addHessianEntry<36>(entries, cell_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown polygon edge case" << std::endl;
            }
            // std::cout << "Cell " << face_idx << std::endl;
        }
        else // basal and lateral faces area term
        {
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                // computeArea4PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                // computeQuadFaceAreaHessian(coeff, positions, hessian);
                // computeArea4PointsSquaredHessian(coeff, positions, hessian);
                if (use_face_centroid)
                    computeArea4PointsSquaredSumHessian(coeff, positions, hessian);
                else
                    computeQuadFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                // computeArea5PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea5PointsSquaredHessian(coeff, positions, hessian);
                else
                    computePentFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                // computeArea6PointsHessian(coeff, positions, hessian);
                // hessian *= coeff;
                if (use_face_centroid)
                    computeArea6PointsSquaredHessian(coeff, positions, hessian);
                else
                    computeHexFaceAreaSquaredSumHessian(coeff, positions, hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
            }
        }
        if (add_yolk_volume)
        {
            if (face_idx < lateral_face_start && face_idx >= basal_face_start)
            {
                
                VectorXT positions;
                positionsFromIndices(positions, face_vtx_list);
                
                if (face_vtx_list.size() == 4)
                {
                    
                    Matrix<T, 12, 12> d2Vdx2;
                    computeConeVolume4PointsHessian(positions, mesh_centroid, d2Vdx2);
                    // computeQuadConeVolumeHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 12, 12> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<12>(entries, face_vtx_list, hessian);
                }
                else if (face_vtx_list.size() == 5)
                {
                    Matrix<T, 15, 15> d2Vdx2;
                    computeConeVolume5PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 15, 15> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<15>(entries, face_vtx_list, hessian);

                }
                else if (face_vtx_list.size() == 6)
                {
                    Matrix<T, 18, 18> d2Vdx2;
                    computeConeVolume6PointsHessian(positions, mesh_centroid, d2Vdx2);
                    Matrix<T, 18, 18> hessian;
                    if (use_yolk_pressure)
                        hessian = -pressure_constant * d2Vdx2;
                    else
                        hessian = By * (yolk_vol_curr - yolk_vol_init) * d2Vdx2;
                    
                    addHessianEntry<18>(entries, face_vtx_list, hessian);
                }
                else
                {
                    std::cout << "unknown polygon edge case" << std::endl;
                }
            }
        }

        
    });

    if (use_sphere_radius_bound)
    {
        for (int i = 0; i < basal_vtx_start; i++)
        {
            Matrix<T, 3, 3> hessian;
            if (sphere_bound_penalty)
            {
                T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
                if (Rk >= Rc)
                {
                    computeRadiusPenaltyHessian(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, hessian);
                    addHessianEntry<3>(entries, {i}, hessian);    
                }
            }
            else
            {
                sphereBoundEnergyHessian(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, hessian);
                addHessianEntry<3>(entries, {i}, hessian);
            }
        }
    }

        
    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    // std::cout << K << std::endl;
    // std::ofstream out("hessian.txt");
    // out << K;
    // out.close();
    // std::getchar();
    K.makeCompressed();
}

void VertexModel::projectDirichletDoFMatrix(StiffnessMatrix& A, 
    const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}
