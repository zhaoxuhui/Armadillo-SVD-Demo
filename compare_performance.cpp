#include <iostream>

// Armadillo相关引用
#include <armadillo>

// Eigen相关引用
#include <Eigen/Core>
#include <Eigen/Dense>

// Truncated SVD库引用
#include "svd_truncated.cpp"

using namespace std;

// Truncated SVD分解方法
void truncated_svd(Eigen::MatrixXd in_mat, Eigen::MatrixXd &U_mat, Eigen::MatrixXd &S_mat, Eigen::MatrixXd &V_mat) {
    double *in_mat_ptr = in_mat.data();
    int m = in_mat.rows();
    int n = in_mat.cols();
    double *un = new double[m * n];
    double *sn = new double[n * n];
    double *v = new double[n * n];

    // m > n, row > col
    svd_truncated_u(m, n, in_mat_ptr, un, sn, v);

    U_mat = Eigen::Map<Eigen::MatrixXd>(un, m, n);
    S_mat = Eigen::Map<Eigen::MatrixXd>(sn, n, n);
    V_mat = Eigen::Map<Eigen::MatrixXd>(v, n, n);
}

// Eigen的BDC SVD分解方法
void bdc_svd(Eigen::MatrixXd in_Mat, Eigen::MatrixXd &U, Eigen::MatrixXd &S, Eigen::MatrixXd &V) {
    // 构造SVD对象，除了BDC还有Jacobian等方法
    Eigen::BDCSVD<Eigen::MatrixXd> svd1(in_Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Eigen得到的是奇异值向量以及V，所以需要构造对角阵并对V转置
    U = svd1.matrixU();
    S = svd1.singularValues().asDiagonal();
    V = svd1.matrixV();
}


int main() {
    int row = 6000, col = 50;
    int step = 1000;
    int times = 10;
    vector<vector<double>> res_list;

    for (int i = 0; i < times; ++i) {
        vector<double> performance_list;

        row += step;
        //col += step;
        cout << row << " " << col << endl;

        performance_list.push_back(row);
        performance_list.push_back(col);

        Eigen::MatrixXd mat_eigen = Eigen::MatrixXd::Random(row, col);

        Eigen::MatrixXd U1, S1, V1;
        int t1 = clock();
        truncated_svd(mat_eigen, U1, S1, V1);
        int t2 = clock();
        double time1 = (t2 - t1) * 1.0 / CLOCKS_PER_SEC;
        Eigen::MatrixXd Restore1 = U1 * S1 * V1.transpose();
        double max1 = (mat_eigen - Restore1).cwiseAbs().maxCoeff();
        cout << "cost time truncated:" << time1 << " s" << endl;
        cout << "max error truncated:" << max1 << endl;
        performance_list.push_back(time1);
        performance_list.push_back(max1);

        arma::mat mat_arma = arma::mat(mat_eigen.data(), mat_eigen.rows(), mat_eigen.cols(),
                                       false, false);
        arma::mat U_arma;
        arma::vec s_arma;
        arma::mat V_arma;
        int t3 = clock();
        arma::svd_econ(U_arma, s_arma, V_arma, mat_arma);
        int t4 = clock();
        double time2 = (t4 - t3) * 1.0 / CLOCKS_PER_SEC;
        arma::mat Restore2 = U_arma * diagmat(s_arma) * V_arma.t();
        Eigen::MatrixXd Restore2_eigen = Eigen::Map<Eigen::MatrixXd>(Restore2.memptr(),
                                                                     Restore2.n_rows,
                                                                     Restore2.n_cols);
        double max2 = (mat_eigen - Restore2_eigen).cwiseAbs().maxCoeff();
        cout << "cost time armadillo:" << time2 << " s" << endl;
        cout << "max error armadillo:" << max2 << endl;
        performance_list.push_back(time2);
        performance_list.push_back(max2);


        Eigen::MatrixXd U2, S2, V2;
        int t5 = clock();
        bdc_svd(mat_eigen, U2, S2, V2);
        int t6 = clock();
        double time3 = (t6 - t5) * 1.0 / CLOCKS_PER_SEC;
        Eigen::MatrixXd Restore3 = U2 * S2 * V2.transpose();

        double max3 = (mat_eigen - Restore3).cwiseAbs().maxCoeff();
        cout << "cost time eigen:" << time3 << " s" << endl;
        cout << "max error eigen:" << max3 << endl;
        performance_list.push_back(time3);
        performance_list.push_back(max3);

        res_list.push_back(performance_list);
    }

    ofstream fout;
    fout.open("../est_rst.txt");
    for (int j = 0; j < res_list.size(); ++j) {
        fout << to_string(res_list[j][0]) << "\t" <<
             to_string(res_list[j][1]) << "\t" <<
             to_string(res_list[j][2]) << "\t" <<
             to_string(res_list[j][3]) << "\t" <<
             to_string(res_list[j][4]) << "\t" <<
             //             to_string(res_list[j][5]) << "\t" <<
             //             to_string(res_list[j][6]) << "\t" <<
             to_string(res_list[j][5]) << "\n";
    }
    fout.close();
}