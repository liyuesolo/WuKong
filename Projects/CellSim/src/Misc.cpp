#include "../include/VertexModel.h"
#include <fstream>

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

