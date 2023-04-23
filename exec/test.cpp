
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm/fundamental.hpp>

int main(int, char **)
{
    cv::Mat mat1 = (cv::Mat_<float>(2, 8) << 962, 721, 940, 153.60001, 1003.9681, 885.84204, 146.31325, 953.12628, 110, 245, 107, 177.60001, 314.49603, 199.06563, 62.705677, 204.24135);
    cv::Mat mat1b = (cv::Mat_<float>(2, 8) << 978, 729.60004, 955.20001, 141.60001, 1049.2417, 910.72522, 144.32259, 976.41693, 109, 248.40001, 104.4, 178.8, 333.84964, 201.55396, 77.137932, 203.04695);

    cv::Mat mat2 = (cv::Mat_<float>(3, 8) << 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1);
    cv::Mat mat2b = (cv::Mat_<float>(2, 8) << 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16);
    cv::Mat exF = (cv::Mat_<float>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0);
    cv::Mat exmat2b = exF * mat2;
    cv::Mat F1;
    cv::sfm::normalizedEightPointSolver(mat1, mat1b, F1);
    cv::Mat F2;
    cv::sfm::normalizedEightPointSolver(mat2.rowRange(0, 2), exmat2b.rowRange(0, 2), F2);

    std::cout << mat1 << std::endl;
    std::cout << mat1b << std::endl;
    std::cout << mat2 << std::endl;
    std::cout << mat2b << std::endl;

    std::cout << "F1" << std::endl;
    std::cout << F1 << std::endl;
    std::cout << F2 << std::endl;

    std::cout << "F1 * mat1" << std::endl;
    std::cout << exmat2b << std::endl;
    std::cout << exF << std::endl;
}