#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

int main(int, char **)
{
    cv::namedWindow("corners", cv::WINDOW_AUTOSIZE);
    cv::Mat objp = cv::Mat::zeros(9 * 6, 3, CV_32FC1);
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 6; j++)
        {
            objp.at<float>(i * 6 + j, 0) = j;
            objp.at<float>(i * 6 + j, 1) = i;
            objp.at<float>(i * 6 + j, 2) = 0;
        }

    std::vector<cv::String> filenames;
    std::string path = "./images/*.jpg";
    cv::glob(path, filenames);

    std::vector<cv::Mat> objps, imgps;

    for (const auto &filename : filenames)
    {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cerr << "Error: could not read image " << filename << std::endl;
            continue;
        }
        cv::resize(image, image, cv::Size(300, 400));
        cv::Mat corners;
        bool ret = cv::findChessboardCorners(image, cv::Size(6, 9), corners);
        // bool ret = false;
        if (ret)
        {
            cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 000.1));

            cv::drawChessboardCorners(image, cv::Size(6, 9), corners, ret);
            cv::imshow("corners", image);
            cv::waitKey(50);
            objps.push_back(objp);
            imgps.push_back(corners);
        }
    }

    std::vector<cv::Mat> rvec, tvec;
    cv::Mat K, distCoeffs;
    double a = cv::calibrateCamera(objps, imgps, cv::Size(300, 400), K, distCoeffs, rvec, tvec);
    std::cout << "K" << std::endl;
    std::cout << K << std::endl;
    cv::destroyAllWindows();
}
