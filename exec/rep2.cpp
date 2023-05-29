#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    // Check if left mouse button is clicked and four points haven't been selected yet
    if (event == cv::EVENT_LBUTTONDOWN && ((std::vector<cv::Point2f> *)userdata)->size() < 4)
    {
        // Save the point coordinates
        ((std::vector<cv::Point2f> *)userdata)->push_back(cv::Point2f(x, y));
        std::cout << "Selected point " << ((std::vector<cv::Point2f> *)userdata)->size() << ": (" << x << ", " << y << ")" << std::endl;
    }
}

int main(int, char **)
{
    // Must be changed
    std::string imagePath = "./images/1/zoom/1.jpg";
    std::string resultPath = "./images/2/results/";
    cv::Mat image = cv::imread(imagePath);
    cv::Mat copy = image.clone();
    cv::namedWindow("Source", cv::WINDOW_NORMAL);
    cv::resizeWindow("Source", 800, 600);
    cv::namedWindow("Warped", cv::WINDOW_NORMAL);
    cv::resizeWindow("Warped", 600, 600);
    if (image.empty())
    {
        std::cerr << "Error: could not read image " << imagePath << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> points2;
    points2 = {{0, 0}, {300, 0}, {300, 300}, {0, 300}};
    std::cout << image.rows / 20 << std::endl;
    cv::setMouseCallback("Source", onMouse, &points);

    while (points.size() < 4)
    {
        if (points.size() > 0)
        {
            cv::circle(copy, points[points.size() - 1], MIN(15, image.rows / 40), cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("Source", copy);
        cv::waitKey(50);
    }
    cv::circle(copy, points[points.size() - 1], MIN(15, image.rows / 40), cv::Scalar(0, 255, 0), -1);

    cv::Mat M = cv::getPerspectiveTransform(points, points2);
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(300, 300));
    cv::imshow("Warped", warped);
    cv::imwrite(resultPath + "withPoints.jpg", copy);
    cv::imwrite(resultPath + "warped.jpg", warped);
    cv::waitKey(10000);
}