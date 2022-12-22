#include "Projects/Foam2D/deprecated/include/ImageMatch/ImageMatchNLP.h"

void ImageMatchNLP::check_gradients(const Eigen::VectorXd &x) const {
    double eps = 1e-6;

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    int nx = info->n_free * (dims + 1);
    int ng = info->n_free * dims;

    VectorXT g = eval_g(x);
    Eigen::SparseMatrix<double> jac = eval_jac_g_sparsematrix(x);
    for (int i = 0; i < nx; i++) {
        VectorXd xp = x;
        xp(i) += eps;
        VectorXT gp = eval_g(xp);
        xp(i) += eps;
        VectorXT gp2 = eval_g(xp);

        for (int j = 0; j < ng; j++) {
            std::cout << "g[" << j << ", " << i << "] " << g(j) << " " << (gp(j) - g(j)) / eps << " " << jac.coeff(j, i)
                      << " " << (gp2(j) - g(j) - 2 * eps * jac.coeff(j, i)) / (gp(j) - g(j) - eps * jac.coeff(j, i))
                      << std::endl;
        }
    }

//    double f = eval_f(x);
//    VectorXd grad_f = eval_grad_f(x);
//    for (int i = 0; i < nx; i++) {
//        VectorXd xp = x;
//        xp(i) += eps;
//        double fp = eval_f(xp);
//        xp(i) += eps;
//        double fp2 = eval_f(xp);
//
//        std::cout << "f[" << i << "] " << f << " " << fp << " " << fp2 << " " << (fp - f) / eps << " " << grad_f(i)
//                  << " " << (fp - f - eps * grad_f(i)) << " " << (fp2 - f - 2 * eps * grad_f(i)) << " "
//                  << (fp2 - f - 2 * eps * grad_f(i)) / (fp - f - eps * grad_f(i))
//                  << std::endl;
//    }
}

double ImageMatchNLP::eval_f(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    VectorXd c_free = x.segment(0, n_free * dims);

//    std::cout << "Eval f" << std::endl;
//    for (int i = 0; i < x.rows(); i++) {
//        std::cout << x(i) << std::endl;
//    }

//    std::cout << "internal f: " << 0.0001 * objective->evaluate(c_free) << std::endl;

    return objective->evaluate(c_free);

    return 0;
}

VectorXd ImageMatchNLP::eval_grad_f(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    VectorXd c_free = x.segment(0, n_free * dims);

//    std::cout << "Eval gradf" << std::endl;
//    for (int i = 0; i < x.rows(); i++) {
//        std::cout << x(i) << std::endl;
//    }

    VectorXd grad_f(n_free * (dims + 1));
    grad_f.setZero();
    grad_f.segment(0, n_free * dims) = objective->get_dOdc(c_free);
    return grad_f;

    return VectorXd::Zero(n_free * (dims + 1));
}

VectorXd ImageMatchNLP::eval_g(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;

//    std::cout << "Eval g" << std::endl;
//    for (int i = 0; i < x.rows(); i++) {
//        std::cout << x(i) << std::endl;
//    }

    VectorXd g = energy->get_dOdx(x).segment(0, n_free * dims);

//    VectorXd xmg = x - x_guess;
//    std::cout << "X - guess norm: " << xmg.norm() << std::endl;
//    for (int i = 0; i < xmg.rows(); i++) {
//        std::cout << "x - guess[" << i << "] = " << xmg(i) << " " << x(i) << std::endl;
//    }
//    for (int i = 0; i < g.rows(); i++) {
//        std::cout << "internal g[" << i << "] = " << g(i) << std::endl;
//    }

    return g;
}

Eigen::SparseMatrix<double> ImageMatchNLP::eval_jac_g_sparsematrix(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;

//    std::cout << "Eval jac g" << std::endl;
//    for (int i = 0; i < x.rows(); i++) {
//        std::cout << x(i) << std::endl;
//    }

    Eigen::SparseMatrix<double> jac;
    energy->getHessian(x, jac);

    return jac.block(0, 0, n_free * dims, n_free * (dims + 1));
}
