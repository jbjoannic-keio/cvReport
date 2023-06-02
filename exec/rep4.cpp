#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{

    std::string folderPath = "./images/4/Set1"; // Path to the folder with images of the same panorama

    std::vector<std::string> imagePaths;

    // Create viewer
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image", 800, 600);

    // Read all images from folder
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            imagePaths.push_back(entry.path());
        }
    }

    // create vectors to store images
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> imagesWithFeatures;                // images with features drawn on them (for visualization)
    std::vector<std::vector<cv::KeyPoint>> imagesKeypoints; // keypoints for each image
    std::vector<cv::Mat> imagesDescriptors;                 // descriptors for each image

    // create AKAZE detector, descriptor extractor. Other are possible
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

    // detect keypoints and compute descriptors for each image
    for (int i = 0; i < imagePaths.size(); i++)
    {
        cv::Mat img = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);

        cv::Mat imgWithFeatures = img.clone();

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        akaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

        imagesKeypoints.push_back(keypoints);
        imagesDescriptors.push_back(descriptors);
        std::cout << "Number of keypoints in image " << i << ": " << keypoints.size() << std::endl;

        // draw keypoints on image
        for (int j = 0; j < keypoints.size(); j++)
        {
            cv::circle(imgWithFeatures, keypoints[j].pt, 5, cv::Scalar(0, 255, 0), -1);
        }

        // draw image number on image, should be resize on low quality images
        cv::rectangle(imgWithFeatures, cv::Point(0, 0), cv::Point(400, 400), cv::Scalar(0, 0, 0), -1);
        cv::putText(imgWithFeatures, std::to_string(i), cv::Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(255, 255, 255), 5);

        images.push_back(img);
        imagesWithFeatures.push_back(imgWithFeatures);

        cv::imshow("Image", imgWithFeatures);

        std::string res_path = folderPath + "/results/" + "features_" + std::to_string(i) + ".png";
        cv::imwrite(res_path, imgWithFeatures);
        cv::waitKey(10);
    }
    cv::Mat image0 = images[0];
    cv::Mat image1 = images[1];

    std::vector<cv::KeyPoint> keypoint0 = imagesKeypoints[0];
    std::vector<cv::KeyPoint> keypoint1 = imagesKeypoints[1];

    cv::Mat descriptor0 = imagesDescriptors[0];
    cv::Mat descriptor1 = imagesDescriptors[1];

    // create brute force matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    std::vector<std::vector<cv::DMatch>> nn_matches;
    matcher.knnMatch(descriptor0, descriptor1, nn_matches, 2);
    std::vector<cv::KeyPoint> matched1, matched2, inliers1, inliers2;
    std::vector<cv::DMatch> good_matches;

    for (size_t k = 0; k < nn_matches.size(); k++)
    {
        cv::DMatch first = nn_matches[k][0];
        float dist1 = nn_matches[k][0].distance;
        float dist2 = nn_matches[k][1].distance;

        matched1.push_back(keypoint0[first.queryIdx]);
        matched2.push_back(keypoint1[first.trainIdx]);

        if (dist1 < 0.7f * dist2)
        {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[k]);
            inliers2.push_back(matched2[k]);
            good_matches.push_back(cv::DMatch(new_i, new_i, 0));
        }
    }
    cv::Mat imgMatches;
    cv::drawMatches(image0, inliers1, image1, inliers2, good_matches, imgMatches);
    cv::imshow("Image", imgMatches);
    cv::waitKey(10);
    std::string res_path = folderPath + "/results/" + "matches.png";
    cv::imwrite(res_path, imgMatches);

    // convertion from keyPoint to Point2f
    std::vector<cv::Point2f> matches1, matches2;
    for (int j = 0; j < inliers1.size(); j++)
    {
        matches1.push_back(inliers1[j].pt);
        matches2.push_back(inliers2[j].pt);
    }

    std::cout << "Number of matches inliers: " << matches1.size() << std::endl;

    // find fundamental matrix
    cv::Mat mask;
    cv::Mat F = cv::findFundamentalMat(matches1, matches2, cv::FM_RANSAC, 3, 0.99, mask); //  modifier le 3  pour prendre plus ou moins de matchs en compte

    std::vector<cv::Point2f> epiinliers1, epiinliers2;
    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0) == 1)
        {
            epiinliers1.push_back(matches1[i]);
            epiinliers2.push_back(matches2[i]);
        }
    }

    std::cout << "Number of epipolar inliers: " << epiinliers1.size() << std::endl;

    // compute epipolar lines
    std::vector<cv::Vec3f> epilines1, epilines2;
    cv::computeCorrespondEpilines(epiinliers1, 1, F, epilines1);
    cv::computeCorrespondEpilines(epiinliers2, 2, F, epilines2);

    std::cout << "Number of epipolar lines: " << epilines1.size() << std::endl;

    cv::RNG rng(0);
    cv::cvtColor(imagesWithFeatures[0], imagesWithFeatures[0], cv::COLOR_GRAY2BGR);
    cv::cvtColor(imagesWithFeatures[1], imagesWithFeatures[1], cv::COLOR_GRAY2BGR);

    // draw lines
    for (int i = 0; i < epilines1.size(); i++)
    {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::line(imagesWithFeatures[0], cv::Point(0, -epilines1[i][2] / epilines1[i][1]), cv::Point(images[0].cols, -(epilines1[i][2] + epilines1[i][0] * images[0].cols) / epilines1[i][1]), color, 2);
        cv::line(imagesWithFeatures[1], cv::Point(0, -epilines2[i][2] / epilines2[i][1]), cv::Point(images[1].cols, -(epilines2[i][2] + epilines2[i][0] * images[1].cols) / epilines2[i][1]), color, 2);
    }
    cv::imshow("Image", images[0]);
    cv::waitKey(10);
    cv::imshow("Image", images[1]);
    cv::waitKey(10);
    cv::imwrite(folderPath + "/results/" + "epilines1.png", images[0]);
    cv::imwrite(folderPath + "/results/" + "epilines2.png", images[1]);

    cv::Mat fused = cv::Mat::zeros(images[0].rows, images[0].cols + images[1].cols, CV_8UC3);
    imagesWithFeatures[0].copyTo(fused(cv::Rect(0, 0, images[0].cols, images[0].rows)));
    imagesWithFeatures[1].copyTo(fused(cv::Rect(images[0].cols, 0, images[1].cols, images[1].rows)));
    cv::imshow("Image", fused);
    cv::waitKey(10);
    cv::imwrite(folderPath + "/results/" + "epilines_fused.png", fused);

    // distance average
    double totalDistanceEpiInliers = 0.0;
    size_t numPointsEpiInliers = epiinliers1.size();

    for (size_t i = 0; i < numPointsEpiInliers; i++)
    {
        double distance1 = std::abs(epilines1[i][0] * epiinliers2[i].x + epilines1[i][1] * epiinliers2[i].y + epilines1[i][2]) / std::sqrt(epilines1[i][0] * epilines1[i][0] + epilines1[i][1] * epilines1[i][1]);
        double distance2 = std::abs(epilines2[i][0] * epiinliers1[i].x + epilines2[i][1] * epiinliers1[i].y + epilines2[i][2]) / std::sqrt(epilines2[i][0] * epilines2[i][0] + epilines2[i][1] * epilines2[i][1]);
        totalDistanceEpiInliers += sqrt(distance1 * distance1 + distance2 * distance2);
    }
    double avgDistanceEpi = totalDistanceEpiInliers / (2 * numPointsEpiInliers);
    std::cout << "Average distance with Epi inliers : " << avgDistanceEpi << std::endl;
}