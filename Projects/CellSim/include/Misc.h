#ifndef MISC_H
#define MISC_H

#include "VecMatDef.h"
#include <iostream>

void saveOBJPrism(int edge)
{
    using TV = Vector<double, 3>;

    if (edge == 5)
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
}

#endif