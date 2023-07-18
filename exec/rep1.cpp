#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

int main(int, char **)
{
    cv::Mat objp = cv::Mat::zeros(9 * 6, 3, CV_32FC1);
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 6; j++)
        {
            objp.at<float>(i * 6 + j, 0) = j * 1;
            objp.at<float>(i * 6 + j, 1) = i * 1;
            objp.at<float>(i * 6 + j, 2) = 0;
        }

    std::vector<cv::String> filenames;

    // Must be changed
    std::string path = "./images/1/zoom/*.jpg";
    std::string resultPath = "./images/1/zoom/results/";
    cv::glob(path, filenames);

    std::vector<cv::Mat> objps, imgps;

    for (const auto &filename : filenames)
    {
        cv::Mat imageColor = cv::imread(filename); //, cv::IMREAD_GRAYSCALE);
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        // cv::resize(image, image, cv::Size(4000, 3000));
        // cv::resize(imageColor, imageColor, cv::Size(4000, 3000));
        if (image.empty())
        {
            std::cerr << "Error: could not read image " << filename << std::endl;
            continue;
        }

        cv::Mat corners;
        bool ret = cv::findChessboardCorners(image, cv::Size(6, 9), corners);
        if (ret)
        {
            cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 000.1));

            cv::drawChessboardCorners(imageColor, cv::Size(6, 9), corners, ret);

            // substr depends on the base path, should be the same as the path length
            cv::imwrite(resultPath + filename.substr(19), imageColor);
            cv::waitKey(50);
            objps.push_back(objp);
            imgps.push_back(corners);
        }
    }

    std::vector<cv::Mat> rvec, tvec;
    cv::Mat K, distCoeffs;
    double a = cv::calibrateCamera(objps, imgps, cv::Size(3456, 4608), K, distCoeffs, rvec, tvec);
    std::cout << "K" << std::endl;
    std::cout << K << std::endl;
    cv::destroyAllWindows();
}
