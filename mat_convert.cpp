// Armadillo相关引用
#include <armadillo>

// Eigen相关引用
#include <Eigen/Core>
#include <Eigen/Dense>

// OpenCV相关引用
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>

using namespace std;

int main() {
    Eigen::MatrixXd mat_eigen = Eigen::MatrixXd::Random(4, 3);
    cout << "Original mat(Eigen):\n" << mat_eigen << endl;

    // Eigen转Armadillo
    arma::mat mat_arma = arma::mat(mat_eigen.data(), mat_eigen.rows(), mat_eigen.cols(),
                                   false, false);
    cout << "Converted mat(Eigen > Armadillo):\n" << mat_arma << endl;

    // Armadillo转Eigen
    Eigen::MatrixXd mat_restore_eigen = Eigen::Map<Eigen::MatrixXd>(mat_arma.memptr(),
                                                                    mat_arma.n_rows,
                                                                    mat_arma.n_cols);
    cout << "Restoreed mat(Armadillo > Eigen):\n" << mat_restore_eigen << endl;

    // OpenCV转Armadillo
    cv::Mat mat_opencv = cv::Mat::eye(4, 4, CV_64F);
    Eigen::MatrixXd mat_eigen2;
    cv2eigen(mat_opencv, mat_eigen2);
    arma::mat mat_arma2 = arma::mat(mat_eigen2.data(), mat_eigen2.rows(), mat_eigen2.cols(),
                                    false, false);
    cout << "Converted mat(OpenCV > Armadillo):\n" << mat_arma2 << endl;

    // Armadillo转OpenCV
    Eigen::MatrixXd mat_restore_eigen2 = Eigen::Map<Eigen::MatrixXd>(mat_arma2.memptr(),
                                                                     mat_arma2.n_rows,
                                                                     mat_arma2.n_cols);
    cv::Mat mat_opencv2 = cv::Mat::eye(4, 4, CV_64F);
    cv::eigen2cv(mat_restore_eigen2, mat_opencv2);
    cout << "Converted mat(Armadillo > OpenCV):\n" << mat_restore_eigen2 << endl;
    return 0;
}