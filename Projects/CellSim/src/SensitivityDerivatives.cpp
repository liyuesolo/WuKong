#include "../include/VertexModel.h"

void VertexModel::dfdpWeights(MatrixXT& dfdp)
{
    // if (dfdp.rows() != num_nodes * 3 || dfdp.cols() != 8)
        // std::cout << "wrong dfdp input dimension" << std::endl;
    
    // alpha gamma sigma Gamma we B By Bp
    
    VectorXT df_dalpha = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Lateral, 1.0, df_dalpha);
    VectorXT df_dgamma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Basal, 1.0, df_dgamma);
    VectorXT df_dsigma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Apical, 1.0, df_dsigma);

    VectorXT df_dGamma = VectorXT::Zero(num_nodes * 3);
    addEdgeContractionForceEntries(1.0, df_dGamma);

    VectorXT df_dwe = VectorXT::Zero(num_nodes * 3);
    addEdgeForceEntries(ALL, 1.0, df_dwe);

    VectorXT df_dB = VectorXT::Zero(num_nodes * 3);
    T _B = B;
    B = 1.0;
    addCellVolumePreservationForceEntries(df_dB);
    B = _B;

    VectorXT df_dBy = VectorXT::Zero(num_nodes * 3);
    T _By = By;
    By = 1.0;
    addYolkVolumePreservationForceEntries(df_dBy);
    By = _By;

    VectorXT df_dBp = VectorXT::Zero(num_nodes * 3);
    T _Bp = Bp;
    Bp = 1.0;
    addPerivitellineVolumePreservationForceEntries(df_dBp);
    Bp = _Bp;

    // dfdp.col(0) = df_dalpha;
    // dfdp.col(1) = df_dgamma;
    // dfdp.col(2) = df_dsigma;
    // dfdp.col(3) = df_dGamma;
    // dfdp.col(4) = df_dwe;
    // dfdp.col(5) = df_dB;
    // dfdp.col(6) = df_dBy;
    // dfdp.col(7) = df_dBp;

    // for (int i = 0; i < 8; i++)
    // {
    //     for (auto data : dirichlet_data)
    //         dfdp.col(i)[data.first] = 0;
    // }

    dfdp.col(0) = df_dGamma;
    for (int i = 0; i < 1; i++)
    {
        for (auto data : dirichlet_data)
            dfdp.col(i)[data.first] = 0;
    }
}

void VertexModel::dfdpWeightsFD(MatrixXT& dfdp)
{
    if (dfdp.rows() != num_nodes * 3 || dfdp.cols() != 8)
        std::cout << "wrong dfdp input dimension" << std::endl;
    
    // alpha gamma sigma Gamma we B By Bp
    T epsilon = 1e-6;

    VectorXT df_dalpha = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Lateral, alpha - epsilon, df_dalpha);
    df_dalpha *= -1.0;
    addFaceAreaForceEntries(Lateral, alpha + epsilon, df_dalpha);
    df_dalpha /= (2.0 * epsilon);

    VectorXT df_dgamma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Basal, gamma - epsilon, df_dgamma);
    df_dgamma *= -1.0;
    addFaceAreaForceEntries(Basal, gamma + epsilon, df_dgamma);
    df_dgamma /= (2.0 * epsilon);

    VectorXT df_dsigma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Apical, sigma - epsilon, df_dsigma);
    df_dsigma *= -1.0;
    addFaceAreaForceEntries(Apical, sigma + epsilon, df_dsigma);
    df_dsigma /= (2.0 * epsilon);

    VectorXT df_dGamma = VectorXT::Zero(num_nodes * 3);
    addEdgeContractionForceEntries(Gamma - epsilon, df_dGamma);
    df_dGamma *= -1.0;
    addEdgeContractionForceEntries(Gamma + epsilon, df_dGamma);
    df_dGamma /= (2.0 * epsilon);

    VectorXT df_dwe = VectorXT::Zero(num_nodes * 3);
    addEdgeForceEntries(ALL, weights_all_edges - epsilon, df_dwe);
    df_dwe *= -1.0;
    addEdgeForceEntries(ALL, weights_all_edges + epsilon, df_dwe);
    df_dwe /= (2.0 * epsilon);

    VectorXT df_dB = VectorXT::Zero(num_nodes * 3);
    B -= epsilon * B;
    addCellVolumePreservationForceEntries(df_dB);
    df_dB *= -1.0;
    B += 2.0 * epsilon * B;
    addCellVolumePreservationForceEntries(df_dB);
    B -= epsilon * B;
    df_dB  /= (2.0 * epsilon * B);

    VectorXT df_dBy = VectorXT::Zero(num_nodes * 3);
    By -= epsilon * By;
    addYolkVolumePreservationForceEntries(df_dBy);
    df_dBy *= -1.0;
    By += 2.0 * epsilon * By;
    addYolkVolumePreservationForceEntries(df_dBy);
    By -= epsilon * By;
    df_dBy /= (2.0 * epsilon * By);

    VectorXT df_dBp = VectorXT::Zero(num_nodes * 3);
    Bp -= epsilon * Bp;
    addPerivitellineVolumePreservationForceEntries(df_dBp);
    df_dBp *= -1.0;
    Bp += 2.0 * epsilon * Bp;
    addPerivitellineVolumePreservationForceEntries(df_dBp);
    Bp -= epsilon * Bp;
    df_dBp /= (2.0 * epsilon * Bp);

    dfdp.col(0) = df_dalpha;
    dfdp.col(1) = df_dgamma;
    dfdp.col(2) = df_dsigma;
    dfdp.col(3) = df_dGamma;
    dfdp.col(4) = df_dwe;
    dfdp.col(5) = df_dB;
    dfdp.col(6) = df_dBy;
    dfdp.col(7) = df_dBp;

    for (int i = 0; i < 8; i++)
    {
        for (auto data : dirichlet_data)
            dfdp.col(i)[data.first] = 0;
    }
}