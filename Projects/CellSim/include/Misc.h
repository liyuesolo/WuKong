#ifndef MISC_H
#define MISC_H

#include "VecMatDef.h"

#include <utility>
#include <iostream>
#include <fstream>

void saveOBJPrism(int edge)
{
    using TV = Vector<double, 3>;
    using IV = Vector<int, 3>;

    if (edge == 4)
    {
        TV v0 = TV(0, 0, 0); TV v1 = TV(1, 0, 0); 
        TV v2 = TV(1, 0, 1); TV v3 = TV(0, 0, 1);
        TV v4 = TV(0, 1, 0); TV v5 = TV(1, 1, 0); 
        TV v6 = TV(1, 1, 1); TV v7 = TV(0, 1, 1);

        std::ofstream out("prsim_quad_base.obj");
        out << "v " << v0.transpose() << std::endl;
        out << "v " << v1.transpose() << std::endl;
        out << "v " << v2.transpose() << std::endl;
        out << "v " << v3.transpose() << std::endl;
        out << "v " << v4.transpose() << std::endl;   
        out << "v " << v5.transpose() << std::endl;
        out << "v " << v6.transpose() << std::endl;
        out << "v " << v7.transpose() << std::endl;
        out << "f 2 4 1" << std::endl;
        out << "f 3 4 2" << std::endl;
        out << "f 5 8 6" << std::endl;
        out << "f 6 8 7" << std::endl;
        for (int i = 0; i < edge; i++)
        {
            int j = (i + 1) % edge;
            out << "f " << j + 1 << " " << i + 1 << " " << edge + i + 1<< std::endl;
            out << "f " << j + 1 << " " << edge + i + 1 << " " << edge + j + 1 << std::endl;
        }
        out.close();     
    }
    else if (edge == 5)
    {
        TV v0 = TV(0, 0, 1);
        TV v1 = TV(std::sin(2.0/5.0 * M_PI), 0, std::cos(2.0/5.0 * M_PI));
        TV v2 = TV(std::sin(4.0/5.0 * M_PI), 0, -std::cos(1.0/5.0 * M_PI));
        TV v3 = TV(-std::sin(4.0/5.0 * M_PI), 0, -std::cos(1.0/5.0 * M_PI));
        TV v4 = TV(-std::sin(2.0/5.0 * M_PI), 0, std::cos(2.0/5.0 * M_PI));

        TV v5 = TV(0, 1, 1);
        TV v6 = TV(std::sin(2.0/5.0 * M_PI), 1, std::cos(2.0/5.0 * M_PI));
        TV v7 = TV(std::sin(4.0/5.0 * M_PI), 1, -std::cos(1.0/5.0 * M_PI));
        TV v8 = TV(-std::sin(4.0/5.0 * M_PI), 1, -std::cos(1.0/5.0 * M_PI));
        TV v9 = TV(-std::sin(2.0/5.0 * M_PI), 1, std::cos(2.0/5.0 * M_PI));

        std::ofstream out("prsim_penta_base.obj");
        out << "v " << v0.transpose() << std::endl;
        out << "v " << v1.transpose() << std::endl;
        out << "v " << v2.transpose() << std::endl;
        out << "v " << v3.transpose() << std::endl;
        out << "v " << v4.transpose() << std::endl;
        out << "v " << v5.transpose() << std::endl;
        out << "v " << v6.transpose() << std::endl;
        out << "v " << v7.transpose() << std::endl;
        out << "v " << v8.transpose() << std::endl;
        out << "v " << v9.transpose() << std::endl;

        out << "f 1 5 2" << std::endl;
        out << "f 5 4 2" << std::endl;
        out << "f 2 4 3" << std::endl;
        out << "f 6 7 10" << std::endl;
        out << "f 10 7 9" << std::endl;
        out << "f 7 8 9" << std::endl;
        out << "f 7 1 2" << std::endl;
        out << "f 6 1 7" << std::endl;
        out << "f 8 2 3" << std::endl;
        out << "f 7 2 8" << std::endl;
        out << "f 9 3 4" << std::endl;
        out << "f 9 8 3" << std::endl;
        out << "f 10 4 5" << std::endl;
        out << "f 10 9 4" << std::endl;
        out << "f 10 5 1" << std::endl;
        out << "f 10 1 6" << std::endl;
        out.close();
    }
    else if (edge == 6)
    {
        TV v0 = TV(1, -0.5, 0);
        TV v1 = TV(0.5, -0.5, -0.5 * std::sqrt(3));
        TV v2 = TV(-0.5, -0.5, -0.5 * std::sqrt(3));
        TV v3 = TV(-1, -0.5, 0);
        TV v4 = TV(-0.5, -0.5, 0.5 * std::sqrt(3));
        TV v5 = TV(0.5, -0.5, 0.5 * std::sqrt(3));

        TV v6 = TV(1, 0.5, 0);
        TV v7 = TV(0.5, 0.5, -0.5 * std::sqrt(3));
        TV v8 = TV(-0.5, 0.5, -0.5 * std::sqrt(3));
        TV v9 = TV(-1, 0.5, 0);
        TV v10 = TV(-0.5, 0.5, 0.5 * std::sqrt(3));
        TV v11 = TV(0.5, 0.5, 0.5 * std::sqrt(3));

        std::ofstream out("prsim_hex_base.obj");
        out << "v " << v0.transpose() << std::endl;
        out << "v " << v1.transpose() << std::endl;
        out << "v " << v2.transpose() << std::endl;
        out << "v " << v3.transpose() << std::endl;
        out << "v " << v4.transpose() << std::endl;
        out << "v " << v5.transpose() << std::endl;
        out << "v " << v6.transpose() << std::endl;
        out << "v " << v7.transpose() << std::endl;
        out << "v " << v8.transpose() << std::endl;
        out << "v " << v9.transpose() << std::endl;
        out << "v " << v10.transpose() << std::endl;
        out << "v " << v11.transpose() << std::endl;

        out << "f 6 5 4" << std::endl;
        out << "f 3 6 4" << std::endl;
        out << "f 3 1 6" << std::endl;
        out << "f 3 2 1" << std::endl;
        out << "f 10 11 12" << std::endl;
        out << "f 10 12 9" << std::endl;
        out << "f 9 12 7" << std::endl;
        out << "f 9 7 8" << std::endl;

        out << "f 6 1 7" << std::endl;
        out << "f 12 6 7" << std::endl;

        out << "f 1 2 8" << std::endl;
        out << "f 7 1 8" << std::endl;

        out << "f 3 9 2" << std::endl;
        out << "f 9 8 2" << std::endl;

        out << "f 4 10 3" << std::endl;
        out << "f 10 9 3" << std::endl;

        out << "f 5 11 4" << std::endl;
        out << "f 11 10 4" << std::endl;

        out << "f 6 12 5" << std::endl;
        out << "f 12 11 5" << std::endl;


        out.close();

    }
    else if (edge == 7)
    {
        std::vector<TV> vertices;
        vertices.push_back(TV(0, 0, 1));
        vertices.push_back(TV(-0.781831, 0, 0.62349));
        vertices.push_back(TV(-0.974928, 0, -0.222521));
        vertices.push_back(TV(-0.433884, 0, -0.900969));
        vertices.push_back(TV(0.433884, 0, -0.900969));
        vertices.push_back(TV(0.974928, 0, -0.222521));
        vertices.push_back(TV(0.781831, 0, 0.62349));

        for (int i = 0; i < edge; i++)
            vertices.push_back(TV(vertices[i][0], vertices[i][1] + 1, vertices[i][2]));

        std::ofstream out("prsim_sep_base.obj");
        for (int i = 0; i < edge * 2; i++)
            out << "v " << vertices[i].transpose() << std::endl;
        
        std::vector<IV> faces;
        faces.push_back(IV(1, 6, 0));
        faces.push_back(IV(1, 5, 6));
        faces.push_back(IV(2, 5, 1));
        faces.push_back(IV(4, 5, 2));
        faces.push_back(IV(3, 4, 2));

        for (int i = 0; i < 5; i++)
        {
            faces.push_back(IV(faces[i][2], faces[i][1], faces[i][0]) + IV::Constant(edge));
        }
        
        for (auto face : faces)
            out << "f " << (face + IV::Ones()).transpose() << std::endl;

        
        for (int i = 0; i < edge; i++)
        {
            int j = (i + 1) % edge;
            out << "f " << j + 1 << " " << i + 1 << " " << edge + i + 1<< std::endl;
            out << "f " << j + 1 << " " << edge + i + 1 << " " << edge + j + 1 << std::endl;
        }

        out.close(); 
    }
    else if (edge == 8)
    {
        std::vector<TV> vertices;
        vertices.push_back(TV(1, 0, 0));
        vertices.push_back(TV(0.5 * std::sqrt(2), 0, 0.5 * std::sqrt(2)));
        vertices.push_back(TV(0, 0, 1));
        vertices.push_back(TV(-0.5 * std::sqrt(2), 0, 0.5 * std::sqrt(2)));
        vertices.push_back(TV(-1.0, 0, 0));
        vertices.push_back(TV(-0.5 * std::sqrt(2), 0, -0.5 * std::sqrt(2)));
        vertices.push_back(TV(0, 0, -1));
        vertices.push_back(TV(0.5 * std::sqrt(2), 0, -0.5 * std::sqrt(2)));

        for (int i = 0; i < edge; i++)
            vertices.push_back(TV(vertices[i][0], vertices[i][1] + 1, vertices[i][2]));

        std::ofstream out("prsim_oct_base.obj");
        for (int i = 0; i < edge * 2; i++)
            out << "v " << vertices[i].transpose() << std::endl;

        std::vector<IV> faces;
        faces.push_back(IV(1, 7, 0));
        faces.push_back(IV(1, 6, 7));
        faces.push_back(IV(1, 5, 6));
        faces.push_back(IV(1, 3, 5));
        faces.push_back(IV(1, 2, 3));
        faces.push_back(IV(3, 4, 5));

        for (int i = 0; i < 6; i++)
        {
            faces.push_back(IV(faces[i][2], faces[i][1], faces[i][0]) + IV::Constant(edge));
        }
        
        for (auto face : faces)
            out << "f " << (face + IV::Ones()).transpose() << std::endl;
        
        for (int i = 0; i < edge; i++)
        {
            int j = (i + 1) % edge;
            out << "f " << j + 1 << " " << i + 1 << " " << edge + i + 1<< std::endl;
            out << "f " << j + 1 << " " << edge + i + 1 << " " << edge + j + 1 << std::endl;
        }
        out.close(); 
    }
}


#endif