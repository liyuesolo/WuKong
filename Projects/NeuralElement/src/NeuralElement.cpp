#include "../include/NeuralElement.h"

template<int dim>
void NeuralElement<dim>::generateBeamSceneTrainingData(const std::string& folder)
{
    std::ofstream out(folder + "data.txt");
    out << std::setprecision(10);
    if constexpr (dim == 2)
    {
        VectorXT ext_force_curr = solver.f;
        int n_force_sp = 50;
        TV scale_range = TV(0.1, 5.0);
        T df = (scale_range[1] - scale_range[0]) / T(n_force_sp);
        solver.verbose = false;
        solver.max_newton_iter = 500;
        for (int sp = 0; sp < n_force_sp; sp++)
        {
            solver.f = (scale_range[0] + T(sp) * df) * ext_force_curr;
            solver.staticSolve();
            solver.iterateQuadElementsSerial([&](const QuadEleNodes& x_deformed, 
                const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
            {
                for (int i = 0; i < 6; i++)
                    if (x_undeformed(i, 0) < 1e-6)
                        return;

                TM32 undeformed_linear = x_undeformed.block(0, 0, 3, dim);
                TM32 deformed_linear = x_deformed.block(0, 0, 3, dim);

                // std::ofstream out("test_vtx.obj");
                // for (int i = 0; i < x_undeformed.rows(); i++)
                //     out << "v " << x_deformed.row(i) << " 0" << std::endl;
                // out.close();
                // std::exit(0);

                TV dx3 = x_deformed.row(3) - 0.5 * (x_deformed.row(2) + x_deformed.row(1));
                TV dx4 = x_deformed.row(4) - 0.5 * (x_deformed.row(2) + x_deformed.row(0));
                TV dx5 = x_deformed.row(5) - 0.5 * (x_deformed.row(0) + x_deformed.row(1));

                TV e0 = x_deformed.row(2) - x_deformed.row(0);
                TV e1 = x_deformed.row(1) - x_deformed.row(0);
                TV e2 = x_deformed.row(2) - x_deformed.row(1);

                TV E0 = x_undeformed.row(2) - x_undeformed.row(0);
                TV E1 = x_undeformed.row(1) - x_undeformed.row(0);
                TV E2 = x_undeformed.row(2) - x_undeformed.row(1);

                T area_rest = std::abs(E0[0] * E1[1] - E0[1] * E1[0]);

                TV e0_normalized = e0 / E0.norm();
                TV e1_normalized = e1 / E1.norm();
                TV e2_normalized = e2 / E2.norm();

                TV dx3_normalized = dx3 / area_rest;
                TV dx4_normalized = dx4 / area_rest;
                TV dx5_normalized = dx5 / area_rest;

                // out << e0_normalized[0] << " " << e0_normalized[1] << " " << e1_normalized[0] << " " << e1_normalized[1] 
                //     << " " << e2_normalized[0] << " " << e2_normalized[1] << " "
                //      << dx3_normalized[0] << " " << dx3_normalized[1] << " " 
                //     << dx4_normalized[0] << " " << dx4_normalized[1] << " " 
                //     << dx5_normalized[0] << " " << dx5_normalized[1] << std::endl;

                for (int i = 0; i < 3; i++)
                    out << undeformed_linear(i, 0) << " " << undeformed_linear(i, 1) << " ";
                for (int i = 0; i < 3; i++)
                    out << deformed_linear(i, 0) << " " << deformed_linear(i, 1) << " ";
                for (int i = 0; i < 3; i++)
                    out << x_deformed(3+i, 0) << " " << x_deformed(i+3, 1) << " ";
                // out << dx3[0] << " " << dx3[1] << " ";
                // out << dx4[0] << " " << dx4[1] << " ";
                // out << dx5[0] << " " << dx5[1] << std::endl;
                out << std::endl;
            });
            
        }
    }
    out.close();
}


template class NeuralElement<2>;
template class NeuralElement<3>;