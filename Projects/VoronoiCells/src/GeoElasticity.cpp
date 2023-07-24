#include "../include/IntrinsicSimulation.h"
#include "../autodiff/Elasticity.h"

void IntrinsicSimulation::computeGeodesicTriangleRestShape()
{
    X_inv_geodesic.resize(triangles.size());
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        // here we use current length because it's undeformed
        T l0 = current_length[edge_map[e0]];
        T l1 = current_length[edge_map[e1]];
        T l2 = current_length[edge_map[e2]];

        T theta = std::acos((l0 * l0 - l1 * l1 + l2 * l2)/(2.0 * l2 * l0));
        TV2 ei(l0, 0);
        TV2 ej(l2 * std::cos(theta), l2 * std::sin(theta));
        TM2 delta_rest; delta_rest.col(0) = ei; delta_rest.col(1) = ej;
        X_inv_geodesic[tri_idx] = delta_rest.inverse();
        
    });
}

void IntrinsicSimulation::addGeodesicNHEnergy(T& energy)
{
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        
        // computeGeodesicNHEnergy(lambda, mu, undeformed_area[tri_idx], TV(l0, l1, l2), X_inv_geodesic[tri_idx], ei);

        T theta = std::acos((l0[0] * l0[0] - l0[1] * l0[1] + l0[2] * l0[2])/(2.0 * l0[2] * l0[0]));
        TV2 ei(l0[0], 0);
        TV2 ej(l0[2] * std::cos(theta), l0[2] * std::sin(theta));
        TV2 ek = ej - ei;

        T element_energy;
        computeGeodesicNHEnergyWithC(lambda, mu, l, ei, ej, ek, element_energy);
        // // viF^TFvi = l*2
        // Matrix<T, 3, 3> A; 
        // A(0, 0) = ei[0] * ei[0]; A(0, 1) = 2.0 * ei[0] * ei[1]; A(0, 2) = ei[1] * ei[1]; 
        // A(1, 0) = ej[0] * ej[0]; A(1, 1) = 2.0 * ej[0] * ej[1]; A(1, 2) = ej[1] * ej[1]; 
        // A(2, 0) = ek[0] * ek[0]; A(2, 1) = 2.0 * ek[0] * ek[1]; A(2, 2) = ek[1] * ek[1]; 
        // Vector<T, 3> b; b << l0*l0, l1*l1, l2*l2;
        // Vector<T, 3> C_entries = A.inverse() * b;

        energy += element_energy;
    });
}

void IntrinsicSimulation::addGeodesicNHForceEntry(VectorXT& residual)
{
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        // T l0 = current_length[edge_map[e0]];
        // T l1 = current_length[edge_map[e1]];
        // T l2 = current_length[edge_map[e2]];
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        T theta = std::acos((l0[0] * l0[0] - l0[1] * l0[1] + l0[2] * l0[2])/(2.0 * l0[2] * l0[0]));
        TV2 ei(l0[0], 0);
        TV2 ej(l0[2] * std::cos(theta), l0[2] * std::sin(theta));
        TV2 ek = ej - ei;

        Vector<T, 3> dedl;
        computeGeodesicNHEnergyWithCGradient(lambda, mu, l, ei, ej, ek, dedl);
        // computeGeodesicNHEnergyGradient(lambda, mu, undeformed_area[tri_idx], TV(l0, l1, l2), X_inv_geodesic[tri_idx], dedl);

    
        Vector<T, 4> dl0dw0, dl1dw1, dl2dw2;
        computeGeodesicLengthGradient(e0, dl0dw0);
        computeGeodesicLengthGradient(e1, dl1dw1);
        computeGeodesicLengthGradient(e2, dl2dw2);

		addForceEntry(residual, {e0[0], e0[1]}, -dedl[0] * dl0dw0);
		addForceEntry(residual, {e1[0], e1[1]}, -dedl[1] * dl1dw1);
		addForceEntry(residual, {e2[0], e2[1]}, -dedl[2] * dl2dw2);
    });
}
    
void IntrinsicSimulation::addGeodesicNHHessianEntry(std::vector<Entry>& entries)
{
    iterateTriangleSerial([&](Triangle tri, int tri_idx){
        Edge e0, e1, e2;
		getTriangleEdges(tri, e0, e1, e2);
        // T l0 = current_length[edge_map[e0]];
        // T l1 = current_length[edge_map[e1]];
        // T l2 = current_length[edge_map[e2]];
        TV l0; l0 << undeformed_length[edge_map[e0]], undeformed_length[edge_map[e1]], undeformed_length[edge_map[e2]];
        TV l; l << current_length[edge_map[e0]], current_length[edge_map[e1]], current_length[edge_map[e2]];
        T theta = std::acos((l0[0] * l0[0] - l0[1] * l0[1] + l0[2] * l0[2])/(2.0 * l0[2] * l0[0]));
        TV2 ei(l0[0], 0);
        TV2 ej(l0[2] * std::cos(theta), l0[2] * std::sin(theta));
        TV2 ek = ej - ei;

        Vector<T, 3> dedl;
        // computeGeodesicNHEnergyGradient(lambda, mu, undeformed_area[tri_idx], TV(l0, l1, l2), X_inv_geodesic[tri_idx], dedl);
        computeGeodesicNHEnergyWithCGradient(lambda, mu, l, ei, ej, ek, dedl);
        Matrix<T, 3, 3> d2edl2;
        // computeGeodesicNHEnergyHessian(lambda, mu, undeformed_area[tri_idx], TV(l0, l1, l2), X_inv_geodesic[tri_idx], d2edl2);
        computeGeodesicNHEnergyWithCHessian(lambda, mu, l, ei, ej, ek, d2edl2);
        // min e(l(w))
        // d2edw2 = dldw^T d2edl2 dldw + dedl d2ldw2
        
        Vector<T, 4> dl0dw0, dl1dw1, dl2dw2;
		Matrix<T, 4, 4> d2l0dw02, d2l1dw12, d2l2dw22;
        computeGeodesicLengthGradientAndHessian(e0, dl0dw0, d2l0dw02);
        computeGeodesicLengthGradientAndHessian(e1, dl1dw1, d2l1dw12);
        computeGeodesicLengthGradientAndHessian(e2, dl2dw2, d2l2dw22);
		std::unordered_map<int, int> index_map;
		index_map[tri[0]] = 0; index_map[tri[1]] = 1; index_map[tri[2]] = 2;
		MatrixXT dldw(3, 6); dldw.setZero();
		VectorXT row0(6); row0.setZero();
		VectorXT row1(6); row1.setZero();
		VectorXT row2(6); row2.setZero();
		addForceEntry(row0, {index_map[e0[0]], index_map[e0[1]]}, dl0dw0);
		addForceEntry(row1, {index_map[e1[0]], index_map[e1[1]]}, dl1dw1);
		addForceEntry(row2, {index_map[e2[0]], index_map[e2[1]]}, dl2dw2);
		dldw.row(0) = row0; dldw.row(1) = row1; dldw.row(2) = row2;

        MatrixXT tensor_term(6, 6); tensor_term.setZero();
		addHessianMatrixEntry<4>(tensor_term, {index_map[e0[0]], index_map[e0[1]]}, dedl[0] * d2l0dw02);
		addHessianMatrixEntry<4>(tensor_term, {index_map[e1[0]], index_map[e1[1]]}, dedl[1] * d2l1dw12);
		addHessianMatrixEntry<4>(tensor_term, {index_map[e2[0]], index_map[e2[1]]}, dedl[2] * d2l2dw22);

		
		Matrix<T, 6, 6> hessian; hessian.setZero();
		hessian += dldw.transpose() * d2edl2 * dldw;
		hessian += tensor_term;
		addHessianEntry(entries, {tri[0], tri[1], tri[2]}, hessian);
    });
}